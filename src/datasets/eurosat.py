from src.datasets.dataset_registry import get_dataset_descriptor


def descriptor(root=None):
    return get_dataset_descriptor("eurosat", root=root)
