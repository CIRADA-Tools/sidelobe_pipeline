import os
import argparse
import subprocess
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy import units as u
import pyink as pu


def filename(objname, survey="DECaLS-DR8", format="fits"):
    # Take Julian coords of name to eliminate white space - eliminate prefix
    name = objname.split(" ")[1]
    filename = f"{name}_{survey}.{format}"
    return filename


def load_catalogue(catalog, flag_data=True, flag_SNR=False, pandas=False, **kwargs):
    fmt = "fits" if catalog.endswith("fits") else "csv"
    rcat = Table.read(catalog, format=fmt)

    if flag_data:
        rcat = rcat[rcat["S_Code"] != "E"]
        rcat = rcat[rcat["Duplicate_flag"] < 2]

    if flag_SNR:
        rcat = rcat[rcat["Peak_flux"] >= 5 * rcat["Isl_rms"]]

    rcat["SNR"] = rcat["Total_flux"] / rcat["Isl_rms"]

    if pandas:
        rcat = rcat.to_pandas()
        if fmt == "fits":
            for col in rcat.columns[rcat.dtypes == object]:
                rcat[col] = rcat[col].str.decode("ascii")

    return rcat


def scale_data(data, log=False, minsnr=None):
    noise = pu.rms_estimate(data[data != 0], mode="mad", clip_rounds=2)
    # data - np.median(remove_zeros)

    if minsnr is not None:
        mask = data >= minsnr * noise
    else:
        mask = np.ones_like(data, dtype=bool)

    if log:
        data = np.log10(data)
    prepro = pu.minmax(data, mask=mask)
    prepro[~mask] = 0
    return prepro.astype(np.float32)


def radio_preprocess(idx, sample, path="images", **kwargs):
    try:
        radio_file = sample["filename"].loc[idx]
        radio_file = os.path.join(path, radio_file)
        radio_hdu = preprocessing.load_radio_fits(radio_file)
        radio_data = radio_hdu.data
        return idx, scale_data(radio_data, **kwargs)
    except:
        print(f"Failed on index {idx}")
        return None


def run_prepro(sample, outfile, shape=(150, 150), threads=None, **kwargs):
    with pu.ImageWriter(outfile, 0, shape, clobber=True) as pk_img:
        if threads is None:
            threads = cpu_count()
        pool = Pool(processes=threads)
        results = [
            pool.apply_async(radio_preprocess, args=(idx, sample), kwds=kwargs)
            for idx in sample.index
        ]
        for res in tqdm(results):
            out = res.get()
            if out is not None:
                pk_img.add(out[1], attributes=out[0])


def map_imbin(
    imbin_file,
    som_file,
    map_file,
    trans_file,
    som_width,
    som_height,
    numthreads=4,
    cpu=False,
    nrot=360,
    log=True,
):
    commands = [
        "Pink",
        "--map",
        imbin_file,
        map_file,
        som_file,
        "--numthreads",
        f"{numthreads}",
        "--som-width",
        f"{som_width}",
        "--som-height",
        f"{som_height}",
        "--store-rot-flip",
        trans_file,
        "--euclidean-distance-shape",
        "circular",
        "-n",
        str(nrot),
    ]
    if cpu:
        commands += ["--cuda-off"]

    if log:
        map_logfile = map_file.replace(".bin", ".log")
        with open(map_logfile, "w") as log:
            subprocess.run(commands, stdout=log)
    else:
        subprocess.run(commands)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description="Add sidelobe info to VLASS component catalogue."
    )
    parser.add_argument(
        dest="catalogue", help="VLASS component catalogue", type=str,
    )
    parser.add_argument(
        dest="outfile", help="Name for the updated component catalogue", type=str,
    )
    parser.add_argument(
        "-p",
        "--cutout_path",
        dest="cutout_path",
        help="Path to the directory containing the input fits images",
        default="images",
        type=str,
    )
    parser.add_argument(
        "-s", "--som", dest="som_file", help="The SOM binary file", type=str,
    )
    parser.add_argument(
        "-n",
        "--neuron_table",
        dest="neuron_table_file",
        help="The table of properties for each SOM neuron",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--threads",
        dest="threads",
        help="Number of threads to use for multiprocessing",
        default=cpu_count(),
        type=int,
    )
    parser.add_argument(
        "--cpu",
        dest="cpu",
        help="Run PINK in cpu mode instead of gpu mode",
        default=False,
        type=bool,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # cutout_path = "/home/adrian/CIRADA/Sample/Sidelobes/images_150pix"
    # catalogue = "/home/adrian/CIRADA/Data/vlass/CIRADA_VLASS1QL_table1_components_v1.fits"
    # som_file = "/home/adrian/CIRADA/SOM/Sidelobes/PtR3_log_SN2/SOM_B3_h10_w10_vlass.bin"
    # neuron_table_file = "/home/adrian/CIRADA/SOM/Sidelobes/PtR3_log_SN2/Psidelobe.npy"

    args = parse_args()
    catalogue = args.catalogue
    cutout_path = args.cutout_path
    som_file = args.som_file
    neuron_table_file = args.neuron_table_file
    threads = args.threads

    cat_name = ".".join(os.path.basename(catalogue).split(".")[:1])
    imbin_file = f"IMG_{cat_name}.bin"

    sample = load_catalogue(catalogue, flag_data=True, flag_SNR=False, pandas=True)
    sample["filename"] = sample["Component_name"].apply(filename, survey="VLASS")

    run_prepro(
        sample,
        imbin_file,
        shape=(150, 150),
        path=cutout_path,
        threads=threads,
        log=True,
        minsnr=2,
    )

    # Map the image binary through the SOM
    som = pu.SOM(som_file)
    som_width, som_height, ndim = som.som_shape
    map_file = imbin_file.replace("IMG", "MAP")
    trans_file = map_file.replace("MAP", "TRANSFORM")
    map_imbin(
        imbin_file,
        som_file,
        map_file,
        trans_file,
        som_width,
        som_height,
        numthreads=cpu_count(),
        cpu=args.cpu,
        nrot=360,
        log=True,
    )

    # Update the component catalogue with the sidelobe probability
    imgs = pu.ImageReader(imbin_file)
    sample = sample.iloc[imgs.records].reset_index(drop=True)
    somset = pu.SOMSet(som, map_file, trans_file)
    sample["bmu"] = somset.mapping.bmu(return_tuples=True)
    bmu = somset.mapping.bmu()

    # This formatting of the neuron table will change in the future
    Psidelobe = np.load(neuron_table_file)

    """
    # If P_sidelobe is stored as a table, convert it to an array
    bmu_y, bmu_x = np.where(Psidelobe >= 0)
    Ps_df = pd.DataFrame(
        {"bmu_y": bmu_y, "bmu_x": bmu_x, "P_sidelobe": Psidelobe.flatten()}
    )
    neuron
    Psidelobe = -np.ones((Ps_df.bmu_y.max()+1, Ps_df.bmu_x.max()+1))
    Psidelobe[Ps_df.bmu_y, Ps_df.bmu_x] = Ps_df.P_sidelobe
    """

    # If P_sidelobe is stored as an array
    sample["P_sidelobe"] = -np.ones(len(sample))
    sample.loc[sample.Peak_to_ring < 3, "P_sidelobe"] = (
        0.01 * Psidelobe[bmu[:, 0], bmu[:, 1]]
    )

    # Update the Quality_flag column
    original_cat = load_catalogue(
        catalogue, flag_data=True, flag_SNR=False, pandas=True
    )
    final_cat = pd.merge(original_cat, sample[["Component_name", "P_sidelobe"]])
    final_cat.loc[(final_cat.P_sidelobe >= 0.05), "Quality_flag"] += 8
    Table.from_pandas(final_cat).write(args.outfile)
