# UEE Experiments for ACL 2023 Industry Track

This is the official repository for the ACL 2023 Industry Track paper
"*Hunt for Buried Treasures: Extracting Unclaimed Embodiments from Patent Specifications*."
With this repository, you can reproduce our experiments reported in the paper.[^1]

[^1]: Experimental results you get using this repository would be slightly different from the paper as we did not use fixed random seeds.

## Requirements and Installation

### Requirements

Create the following environment:

- `Python 3.7.13`
- [git-lfs](https://github.com/git-lfs/git-lfs) required to install [Rinna RoBERTa](https://huggingface.co/rinna/japanese-roberta-base)
- Install required libraries
```
pip install -r requirements.txt
# depending on the cuda version, install torch accordingly
pip install torch==1.12.1+cu113  --extra-index-url https://download.pytorch.org/whl/cu113
```

### Datasets

We need two datasets:

- The UEE dataset for all the experiments.
- The [JSNLI](https://nlp.ist.i.kyoto-u.ac.jp/?%E6%97%A5%E6%9C%AC%E8%AA%9ESNLI%28JSNLI%29%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88) dataset for the Decision (ii) experiments.

#### The UEE Dataset

1. Follow the instruction in `data/uee_dataset/README.md` to prepare files.
2. Make sure you have the following files in `data/uee_dataset/`:
    - `train_uee_221212.json`
    - `dev_uee_221212.json`
    - `test_uee_221212.json`

#### The JSNLI Dataset

1. Download the JSNLI dataset in `data/`:
    ```bash
    # Go to data directory
    cd data/
    # Download JSNLI data
    curl -LO https://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JSNLI/jsnli_1.1.zip&name=JSNLI.zip
    # Go to working directory
    cd ..
    ```
2. Expand it: `unzip jsnli_1.1.zip`
3. Make sure you have the following files in `data/jsnli_1.1/`:
    - `train_w_filtering.tsv`
    - `dev.tsv`

### Language Models

Install the base language models.

We download [Rinna RoBERTa](https://huggingface.co/rinna/japanese-roberta-base) from Hugging Face using git-lfs.
We then convert the RoBERTa to Longformer with a Python script
which is based on [this notebook](https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb) by allenai.

#### Rinna RoBERTa

1. Move to `base_model/` directory: `cd base_model/`
2. Download Rinna RoBERTa: `git clone https://huggingface.co/rinna/japanese-roberta-base` (This may take a while.)
3. You will see the model downloaded in `base_model/japanese-roberta-base`.

#### Longformer

1. Execute:
```python3
python3 convert_model_to_long.py \
	--pretrained-model base_model/japanese-roberta-base \
	--save-model-dir base_model/roberta-long-japanese-seq4096
```
2. You will see the model in `base_model/roberta-long-japanese-seq4096`.

## Experiments

In the paper, we reported our experimental results in Section 4 "Experiments".
The section consists of three subsections:
- Section 4.1 "UEE Baselines"
- Section 4.2 "Decision (i)"
- Section 4.3 "Decision (ii)"

Accordingly, this repository contains three corresponding directories:
- [`UEE_Baseline`](UEE_Baseline)
- [`Decision1`](Decision1)
- [`Decision2`](Decision2)

You will find how to run these experiments in the `README.md` files in the above directories.

## License

See [`LICENSE`](LICENSE) for the code.

For the UEE dataset, refer to [`LICENSE`](data/uee_dataset/LICENSE) in [`data/uee_dataset`](data/uee_dataset) directory.