# Copyright 2025 Valeo.


"""Datasets paths."""

# LOCAL DATASETS
LOCAL_WOMD_TRAIN = "data/train/womd.tfrecord"
LOCAL_NUPLAN_TRAIN = "data/train/nuplan.tfrecord"
LOCAL_WAYMO_TRAIN = "data/train/waymo.tfrecord"
LOCAL_WOMD_VALID = "data/valid/womd.tfrecord"


def get_dataset(dataset: str) -> str | None:
    """Retrieve the dataset path from the dataset name.

    Args:
        dataset: Name of the dataset.

    Returns:
        The dataset path if recognized, else the provided string.

    """
    dataset_dict = {
        "local_womd_train": LOCAL_WOMD_TRAIN,
        "local_nuplan_train": LOCAL_NUPLAN_TRAIN,
        "local_waymo_train": LOCAL_WAYMO_TRAIN,
        "local_womd_valid": LOCAL_WOMD_VALID,
    }

    if dataset in dataset_dict:
        return dataset_dict[dataset]
    else:
        return dataset
