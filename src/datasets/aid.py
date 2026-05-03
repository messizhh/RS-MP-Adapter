from src.datasets.dataset_registry import get_dataset_descriptor


def descriptor(root=None, config=None):
    return get_dataset_descriptor("aid", root=root, config=config)
