#!/bin/sh

python3 src/main.py \
	--base-model=../../../base_model/japanese-roberta-base/ \
	--train-file=../../../data/jsnli_1.1/train_w_filtering.tsv \
	--dev-file=../../../data/jsnli_1.1/dev.tsv \
	--test-file=../../../data/uee_dataset/test_uee_221212.json \
	--max-sequence-length=512 \
	--train-batch-size=32 \
	--eval-batch-size=32 \
	--output-dir=model/ \
	--report-file=report.txt \
	--prediction-file=pred.json \
	--do-train \
	--do-predict \
	--fp16 \
	--warmup-steps=500 \
	--learning-rate=3e-5 \
	--num-epochs=10 \
	--gradient-accumulation-steps=4 \
	--save-steps=4165 \
	--logging-steps=4165 \
	--eval-steps=4165
