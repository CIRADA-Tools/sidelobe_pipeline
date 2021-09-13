# sidelobe_pipeline
-------------------
-------------------
Download Cutouts
Download all VLASS cutouts and put them in the same directory.
- The file names should follow the convention “{coordinates}_VLASS.fits”, where
“coordinates” is the portion of the “Component_name” column that trails the space
(e.g. “J032429.37-271437.7”).
- Each image should be 1.5 arcminutes in each dimension
- The SOM expects 150x150 pixels (0.6 arcseconds per pixel), but the pipeline will
regrid it if it does not match (e.g. 1 arcsecond per pixel is native to QuickLook)
Pipeline Instructions
python3 catalogue outfile -p path_to_image_cutouts -s som_file -n neuron_table
[--cpu] [--overwrite]
Parameters:
- catalogue: The name of the input catalogue (csv or fits)
- outfile: The desired name for the output catalogue.
- path_to_image_cutouts: The directory that contains all of the VLASS cutouts
- som_file: The name of the SOM binary file (SOM_B3_h10_w10_vlass.bin)
- neuron_file: The name containing the information (neuron_info.csv)
- --cpu: A Boolean flag. If set, it runs the Mapping stage in CPU mode (Warning: Slow)
- --overwrite: A Boolean flag. If set, it overwrites the Image and Mapping binaries
instead of skipping those steps if the files already exist.
Pipeline Outputs
- Image binary (preprocessed images)
- Catalogue of components that failed preprocessing (fits file)
- Catalogue of components that passed preprocessing (fits file)
- Mapping binary
- Transform binary
- Final output catalogue
Computing Considerations
The Mapping step is slow on a CPU. If this step is to be conducted on a different machine
with GPU capabilities, exit the pipeline after the preprocessing step has been completed. A
commented-out exit statement is included in the code for this purpose. Copy the IMG*bin file
to the other machine, and run:
Pink --map image_bin map_bin som_bin --som-width 10 --som-height 10 \
--store-rot-flip transform_bin --euclidean-distance-shape circular -n 360
Parameters:
- image_bin: The IMG*bin file created in the preprocessing step
- map_bin: The name for the output Mapping binary. Use the same suffix as the
IMG*bin, replacing “IMG” with “MAP” in the file names.
- transform_bin: The name for the output Transform binary. Replace “IMG” with
“TRANSFORM” in the file names.
Debug Steps
1. Check that as many cutouts as possible have been downloaded.
2. Ensure all output files have been created.
3. Compare the lengths of the input and output catalogues. They should be the same.
Future Modifications
Adding a new column
If a new column is to be added based on the neuron a component is matched to, the
information should be added to “neuron_info.csv”. It is already read in via the “neuron_table”
variable. Follow similar syntax to the Psidelobe code to add the column to the output table.
Modifying a subset of the catalogue
First set up a new catalogue that is a subset of the original catalogue. Run the pipeline as
normal to obtain the output catalogue. Use table joins, such as pandas.merge, to replace the
values of the large catalogue with the relevant columns of the smaller catalogue.
Note that the PINK binary files are ordered in a specific way, and it will be very difficult to
track which rows correspond to which binary images if it is done this way. The alternative is
to rerun the entire pipeline.
