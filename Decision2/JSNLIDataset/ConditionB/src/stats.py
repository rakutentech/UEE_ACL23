import os
import sys
import statistics
import math
import json

accuracy_li = list()
precision_li = list()
recall_li = list()
f1_li = list()

def getReport(report):
    with open(report) as f:
        d = None
        for l in f:
            if not l.startswith("{'eval_loss':"): continue
            l = l.replace("'", "\"")
            d = json.loads(l)
            break
        accuracy_li.append(d['eval_accuracy'])
        precision_li.append(d['eval_precision'])
        recall_li.append(d['eval_recall'])
        f1_li.append(d['eval_f1'])

for run in ['run1', 'run2', 'run3', 'run4', 'run5']:
    getReport(os.path.join(run, 'report.txt'))
    #getReport(os.path.join(run, 'prediction/report.txt'))

def getStats(li):
    mean = statistics.mean(li)
    stdev = statistics.stdev(li)
    return mean, stdev

accuracy_mean, accuracy_stdev = getStats(accuracy_li)
precision_mean, precision_stdev = getStats(precision_li)
recall_mean, recall_stdev = getStats(recall_li)
f1_mean, f1_stdev = getStats(f1_li)

print("| Run | Accuracy | Precision | Recall | F1 |")
#print("| :---: | ---: | ---: | ---: | ---: |")
print("| ---- | ---- | ---- | ---- | ---- |")
print(f"| Run1 | {accuracy_li[0]} | {precision_li[0]} | {recall_li[0]} | {f1_li[0]} |")
print(f"| Run2 | {accuracy_li[1]} | {precision_li[1]} | {recall_li[1]} | {f1_li[1]} |")
print(f"| Run3 | {accuracy_li[2]} | {precision_li[2]} | {recall_li[2]} | {f1_li[2]} |")
print(f"| Run4 | {accuracy_li[3]} | {precision_li[3]} | {recall_li[3]} | {f1_li[3]} |")
print(f"| Run5 | {accuracy_li[4]} | {precision_li[4]} | {recall_li[4]} | {f1_li[4]} |")
print(f"| Mean | {accuracy_mean:.4f} | {precision_mean:.4f} | {recall_mean:.4f} | {f1_mean:.4f} |")
print(f"| Stdev | {accuracy_stdev:.4f} | {precision_stdev:.4f} | {recall_stdev:.4f} | {f1_stdev:.4f} |")
