# Condition B

## Running the Experiment

1. Make sure you have the following:
    - Rinna RoBERTa: `../../../base_model/japanese-roberta-base/`
    - Training data: `../../../data/jsnli_1.1/train_w_filtering.tsv`
    - Development data: `../../../data/jsnli_1.1/dev.tsv`
    - Test data: `../../../data/uee_dataset/test_uee_221212.json`
2. Execute: `sh run.sh` (Note: In case of memory error, lower the batch size with adjusting other hyperparameters accordingly)
3. You will find the following:
    - `model/`: Fine-tuned model
    - `report.txt`: Evaluation result on the test data
    - `pred.json`: Prediction result for the test data

    You can change the paths to these files
    by editing the following arguments in `run.sh`:
    - `--output-dir=model/`
    - `--report-file=report.txt`
    - `--prediction-file=pred.json`

## Get Mean and Standard Deviation

You can calculate mean and standard deviation
from multiple runs of this experiment in the following way.

1. Run the above experiment five times and save `report.txt`
from each run in `run1/`, `run2/`, ..., and `run5/`, so that
you have the following:

    - `run1/report.txt`
    - `run2/report.txt`
    - `run3/report.txt`
    - `run4/report.txt`
    - `run5/report.txt`

2. Execute:
```python3
python3 src/stats.py
```

3. You will see a markdown table summarizing each run's performances
and the mean and standard deviation from them.

You can change the names of the `run` directories and
the number of experiments to calculate mean and standard deviation
by editing the following lines of `src/stats.py` accordingly.

```python3
for run in ['run1', 'run2', 'run3', 'run4', 'run5']:
    getReport(os.path.join(run, 'report.txt'))
```

```python3
print(f"| Run1 | {accuracy_li[0]} | {precision_li[0]} | {recall_li[0]} | {f1_li[0]} |")
print(f"| Run2 | {accuracy_li[1]} | {precision_li[1]} | {recall_li[1]} | {f1_li[1]} |")
print(f"| Run3 | {accuracy_li[2]} | {precision_li[2]} | {recall_li[2]} | {f1_li[2]} |")
print(f"| Run4 | {accuracy_li[3]} | {precision_li[3]} | {recall_li[3]} | {f1_li[3]} |")
print(f"| Run5 | {accuracy_li[4]} | {precision_li[4]} | {recall_li[4]} | {f1_li[4]} |")
```
