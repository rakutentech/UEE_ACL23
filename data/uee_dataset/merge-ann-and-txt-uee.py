#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
merge-ann-and-txt-uee.py -o train_uee_221212.json \
	-a train_uee_221212_ann.json -t train_uee_221212_txt.json
'''

import sys
import argparse
import json
from collections import ChainMap

ap = argparse.ArgumentParser()
ap.add_argument('-o', '--out-file', type=str, required=True)
ap.add_argument('-a', '--ann-file', type=str, required=True)
ap.add_argument('-t', '--txt-file', type=str, required=True)
args = ap.parse_args()

with open(args.out_file, 'w') as fo, \
        open(args.ann_file) as fa, open(args.txt_file) as ft:
            for al, tl in zip(fa, ft):
                da = json.loads(al)
                dt = json.loads(tl)
                cm = ChainMap(da, dt)
                print(json.dumps(dict(cm), \
                        ensure_ascii=False), file=fo)
