#!/bin/sh

python3 src/main.py \
	--base-model=../../base_model/roberta-long-japanese-seq4096/ \
	--train-file=../../data/uee_dataset/train_uee_221212.json \
	--dev-file=../../data/uee_dataset/dev_uee_221212.json \
	--test-file=../../data/uee_dataset/test_uee_221212.json \
	--max-sequence-length=4096 \
	--train-batch-size=16 \
	--eval-batch-size=16 \
	--output-dir=model/ \
	--report-file=report.txt \
	--prediction-file=pred.json \
	--do-train \
	--do-predict \
	--fp16 \
	--warmup-steps=200 \
	--learning-rate=2e-5 \
	--num-epochs=10 \
	--gradient-accumulation-steps=2 \
	--save-steps=225 \
	--logging-steps=225 \
	--eval-steps=225
