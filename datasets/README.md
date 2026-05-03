# Datasets

Dataset roots are configured through `configs/datasets/*.yaml` or command-line overrides.

This repository does not hard-code local or server dataset paths. Class-folder loaders validate candidate roots and expected folder layouts before constructing sample lists.

Local WSL smoke tests use temporary fake datasets and do not require real EuroSAT, AID, or NWPU-RESISC45 files.
