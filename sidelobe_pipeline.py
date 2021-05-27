import os
import argparse
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
import pyink as pu


def filename(objname, survey="DECaLS-DR8", format="fits"):
    # Take Julian coords of name to eliminate white space - eliminate prefix
    name = objname.split(" ")[1]
    filename = f"{name}_{survey}.{format}"
    return filename


def load_catalogue(catalog, flag_data=False, flag_SNR=False, pandas=False, **kwargs):
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


def load_fits(filename, ext=0):
    hdulist = fits.open(filename)
    d = hdulist[ext]
    return d


def load_radio_fits(filename, ext=0):
    hdu = load_fits(filename, ext=ext)
    wcs = WCS(hdu.header).celestial
    hdu.data = np.squeeze(hdu.data)
    hdu.header = wcs.to_header()
    return hdu


def scale_data(data, log=False, minsnr=None):
    img = np.zeros_like(data)
    noise = pu.rms_estimate(data[data != 0], mode="mad", clip_rounds=2)
    # data - np.median(remove_zeros)

    if minsnr is not None:
        mask = data >= minsnr * noise
    else:
        mask = np.ones_like(data, dtype=bool)
    data = data[mask]

    if log:
        data = np.log10(data)
    img[mask] = pu.minmax(data)
    return img.astype(np.float32)


def radio_preprocess(idx, sample, path="images", **kwargs):
    try:
        radio_file = sample["filename"].loc[idx]
        radio_file = os.path.join(path, radio_file)
        radio_hdu = load_radio_fits(radio_file)
        radio_data = radio_hdu.data
        return idx, scale_data(radio_data, **kwargs)
    except Exception as e:
        print(f"Failed on index {idx}: {e}")
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


def run_prepro_seq(sample, outfile, shape=(150, 150), **kwargs):
    with pu.ImageWriter(outfile, 0, shape, clobber=True) as pk_img:
        for idx in tqdm(sample.index):
            out = radio_preprocess(idx, sample, **kwargs)
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


def fill_duplicates(cat, cols):
    # Fill in `cols` for duplicates by searching for matches in the rest
    # of the duplicate components.
    # Need to apply this multiple times because of the duplicate flagging algorithm.
    missing_comps = cat[(cat.Duplicate_flag >= 1) & np.isnan(cat[cols[0]])]
    not_missing_comps = cat[(cat.Duplicate_flag >= 1) & ~np.isnan(cat[cols[0]])]

    missing_coords = SkyCoord(
        missing_comps["RA"].values, missing_comps["DEC"].values, unit=u.deg
    )
    not_missing_coords = SkyCoord(
        not_missing_comps["RA"].values, not_missing_comps["DEC"].values, unit=u.deg
    )

    idx1, idx2, sep, dist = search_around_sky(
        missing_coords, not_missing_coords, seplimit=2 * u.arcsec
    )
    # When multiple matches are found, choose the one with the highest SNR
    idx1u, idx1c = np.unique(idx1, return_counts=True)
    idx2u = [
        idx2[idx1 == i1][0]
        if i1c == 1
        else idx2[idx1 == i1][final_cat.iloc[idx2[idx1 == i1]]["SNR"].argmax()]
        for i1, i1c in zip(idx1u, idx1c)
    ]

    for col in cols:
        cat.loc[missing_comps.iloc[idx1].index, col] = (
            not_missing_comps[col].iloc[idx2].values
        )


def fill_all_duplicates(cat, cols):
    nan_count = 0
    while np.sum(np.isnan(cat[cols[0]])) != nan_count:
        nan_count = np.sum(np.isnan(cat[cols[0]]))
        fill_duplicates(cat, cols)


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

    sample = load_catalogue(catalogue, flag_data=False, flag_SNR=False, pandas=True)
    sample["filename"] = sample["Component_name"].apply(filename, survey="VLASS")

    # Subset on Duplicate_flag, then fill in those values later
    # Keep S_Code == "E" (951k)?
    sample = sample[sample["Duplicate_flag"] < 2].reset_index(drop=True)

    run_prepro_seq(
        sample,
        imbin_file,
        shape=(150, 150),
        path=cutout_path,
        # threads=threads,
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
    sample["Neuron_dist"] = somset.mapping.bmu_ed()
    bmu = somset.mapping.bmu()
    sample["Best_neuron_y"] = bmu[:, 0]
    sample["Best_neuron_x"] = bmu[:, 1]

    # This formatting of the neuron table will change in the future
    neuron_table = pd.read_csv(neuron_table_file)
    Psidelobe = -np.ones((neuron_table.bmu_y.max() + 1, neuron_table.bmu_x.max() + 1))
    Psidelobe[neuron_table.bmu_y, neuron_table.bmu_x] = neuron_table.P_sidelobe

    sample["P_sidelobe"] = -np.ones(len(sample))
    lowPtR = (sample.Peak_to_ring < 3) & (sample.S_Code != "E")
    sample.loc[lowPtR, "P_sidelobe"] = 0.01 * Psidelobe[bmu[:, 0], bmu[:, 1]][lowPtR]

    neuron_cols = [
        "Best_neuron_y",
        "Best_neuron_x",
        "Neuron_dist",
        "P_sidelobe",
    ]
    sample = sample[["Component_name"] + neuron_cols]

    # Update the Quality_flag column
    original_cat = load_catalogue(
        catalogue, flag_data=False, flag_SNR=False, pandas=True
    )
    final_cat = pd.merge(original_cat, sample, how="left")

    # Add the info for duplicates
    fill_all_duplicates(final_cat, cols=neuron_cols)

    final_cat.loc[(final_cat.P_sidelobe >= 0.1), "Quality_flag"] += 8

    for key in ["SNR", "filename"]:
        if key in final_cat:
            del final_cat[key]

    Table.from_pandas(final_cat).write(args.outfile)
