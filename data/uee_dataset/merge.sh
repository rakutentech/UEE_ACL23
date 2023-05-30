#!/bin/sh

cat train_uee_221212_txt_part1.json train_uee_221212_txt_part2.json > train_uee_221212_txt.json

python3 merge-ann-and-txt-uee.py \
	-o train_uee_221212.json \
	-a train_uee_221212_ann.json \
	-t train_uee_221212_txt.json

python3 merge-ann-and-txt-uee.py \
	-o dev_uee_221212.json \
	-a dev_uee_221212_ann.json \
	-t dev_uee_221212_txt.json

python3 merge-ann-and-txt-uee.py \
	-o test_uee_221212.json \
	-a test_uee_221212_ann.json \
	-t test_uee_221212_txt.json
