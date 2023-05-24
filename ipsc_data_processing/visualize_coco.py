import fiftyone as fo

# A name for the dataset
name = "all_frames_roi-0"

# The directory containing the dataset to import
dataset_dir = "/data/ipsc/well3/all_frames_roi/ytvis19"

# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset  # for example

dataset = fo.Dataset.from_dir(
    # dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    data_path='/data/ipsc/well3/all_frames_roi/ytvis19/JPEGImages',
    labels_path='/data/ipsc/well3/all_frames_roi/ytvis19/ipsc-all_frames_roi_g2_0_38-train.json',
    name=name,
)

session = fo.launch_app(dataset)