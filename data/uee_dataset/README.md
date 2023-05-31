# The UEE Dataset

## Description

This directory contains the UEE dataset, which is described in our ACL 2023 Industry Track paper
"*Hunt for Buried Treasures: Extracting Unclaimed Embodiments from Patent Specifications*".
Refer to Section 3 "Dataset" of the paper for details.

## Preparation

Each of the training, development, and test sets of the UEE dataset
consists of annotation part and text part.
Before using the UEE dataset, merge the two parts in the following way:

```sh
sh merge.sh
```

This will create the following files, which are the merged version
of the training, development, and test set of the UEE dataset.

- `train_uee_221212.json`
- `dev_uee_221212.json`
- `test_uee_221212.json`

## Files

- `README.md`: This file.
- `train_uee_221212_ann.json`: The annotation part of the training set.
- `train_uee_221212_txt_{00,01,02,03}.json`: The text part of the training set (split into 4 parts). 
- `dev_uee_221212_ann.json`: The annotation part of the development set.
- `dev_uee_221212_txt.json`: The text part of the development set.
- `test_uee_221212_ann.json`: The annotation part of the test set.
- `test_uee_221212_txt.json`: The text part of the test set.
- `merge.sh`: The shell script to merge the annotation and text parts of each set.
- `merge-ann-and-txt-uee.py`: The python script used by `merge.sh`
- `LICENSE`: About the license of this dataset.

## License

See [`LICENSE`](LICENSE).
