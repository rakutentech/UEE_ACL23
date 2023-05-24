#!/bin/sh

python3 src/main.py \
	--base-model=../base_model/japanese-roberta-base/ \
	--train-file=../data/uee_dataset/train_uee_221212.json \
	--dev-file=../data/uee_dataset/dev_uee_221212.json \
	--test-file=../data/uee_dataset/test_uee_221212.json \
	--max-sequence-length=510 \
	--train-batch-size=128 \
	--eval-batch-size=128 \
	--output-dir=model/ \
	--report-file=report.txt \
	--prediction-file=pred.json \
	--do-train \
	--do-predict \
	--fp16 \
	--warmup-steps=100 \
	--learning-rate=5e-5 \
	--num-epochs=10 \
	--gradient-accumulation-steps=1 \
	--save-steps=57 \
	--logging-steps=57 \
	--eval-steps=57
