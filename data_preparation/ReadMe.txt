This folder contains files for data preparation on two datasets.

For Massachussets dataset:
1 - png_remove_no_data.py - remove no-data patches 
2 - png_count_and_index_buildings.py - generate building index files
3 - png_insert_label_noises.py - [MAIN FUNCTION] insert incomplete label noises
4 - png_check_data.py - evaluate the quality of generated noisy labels

For Germany dataset:
1 - h5_data_clean.py - clean data according to Nikolai's visual check results 
2 - h5_generate_partition_for_planet_cad.py - generate partitions for training, test and validation
3 - h5_insert_label_noises.py - [MAIN FUNCTION] insert incomplete label noises
4 - h5_check_data.py - evaluate the quality of generated noisy labels