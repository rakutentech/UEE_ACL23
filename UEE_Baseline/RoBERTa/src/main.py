import sys
import json
import argparse
import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
        confusion_matrix,
        classification_report
        )
import transformers
from transformers import (
        T5Tokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        )
from dataset import UEEDataset
from metrics import compute_metrics

transformers.utils.logging.set_verbosity_error()

ap = argparse.ArgumentParser()
ap.add_argument('--base-model', type=str, default=None)
ap.add_argument('--finetuned-model', type=str, default=None)
ap.add_argument("--train-batch-size", type=int, default=32)
ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
ap.add_argument("--eval-batch-size", type=int, default=64)
ap.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
ap.add_argument("--warmup-steps", type=int, default=500)
ap.add_argument("--num-epochs", type=int, default=5)
ap.add_argument("--max-sequence-length", type=int, default=512)
ap.add_argument("--fp16", action="store_true")
ap.add_argument('--train-file', type=str, default=None)
ap.add_argument('--dev-file', type=str, default=None)
ap.add_argument('--test-file', type=str, default=None)
ap.add_argument('--output-dir', type=str, default='./')
ap.add_argument('--prediction-file', type=str, default=None)
ap.add_argument('--report-file', type=str, default=None)
ap.add_argument('--do-train', action='store_true')
ap.add_argument('--do-eval', action='store_true')
ap.add_argument('--do-predict', action='store_true')
ap.add_argument("--save-steps", type=int, default=10000)
ap.add_argument("--logging-steps", type=int, default=1000)
ap.add_argument("--eval-steps", type=int, default=10000)
ap.add_argument("--digit", type=int, default=4)
args = ap.parse_args()

tokenizer = T5Tokenizer.from_pretrained(args.base_model)
tokenizer.do_lower_case = True

model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        )

if args.do_train:
    train_json_list = list()
    dev_json_list = list()

    with open(args.train_file) as f:
        for l in f:
            d = json.loads(l)
            train_json_list.append(d)

    with open(args.dev_file) as f:
        for l in f:
            d = json.loads(l)
            dev_json_list.append(d)
    
    train_dataset = UEEDataset(
            train_json_list, tokenizer, args.max_sequence_length,
            )

    dev_dataset = UEEDataset(
            dev_json_list, tokenizer, args.max_sequence_length,
            )

    training_args = TrainingArguments(
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            output_dir=args.output_dir,
            logging_dir=args.output_dir,
            overwrite_output_dir=True,
            remove_unused_columns=False,
            fp16=args.fp16,
            do_train=True,
            do_eval=True,
            evaluation_strategy='epoch',
            ## need for transformers latest version
            #save_strategy='epoch',
            #logging_strategy='epoch',
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            evaluate_during_training=True,
            )

    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer
            )
    
    trainer.train()
    trainer.save_model()

if args.do_eval:
    test_json_list = list()

    with open(args.test_file) as f:
        for l in f:
            d = json.loads(l)
            test_json_list.append(d)
    
    test_dataset = UEEDataset(
            test_json_list, tokenizer, args.max_sequence_length,
            )

    training_args = TrainingArguments(
            per_device_eval_batch_size=args.eval_batch_size,
            output_dir=args.output_dir
            )

    trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

    eval_results = trainer.evaluate()
    print(eval_results)

if args.do_predict:
    test_json_list = list()

    with open(args.test_file) as f:
        for l in f:
            d = json.loads(l)
            test_json_list.append(d)
    
    test_dataset = UEEDataset(
            test_json_list, tokenizer, args.max_sequence_length,
            )

    training_args = TrainingArguments(
            per_device_eval_batch_size=args.eval_batch_size,
            output_dir=args.output_dir
            )

    trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

    predictions = trainer.predict(test_dataset)
    labels = predictions.label_ids[0] \
            if isinstance(predictions.label_ids, tuple) \
            else predictions.label_ids
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    logits = logits[0] if isinstance(logits, tuple) else logits
    preds = np.argmax(logits, axis=1).tolist()
    scores = softmax(logits, axis=1)[:,1]

    json_list = test_dataset.getJsonList()

    assert len(json_list) == len(preds)
    assert len(json_list) == len(scores)

    for i in range(len(json_list)):
        json_list[i]['pred_label'] = int(preds[i])
        json_list[i]['pred_score'] = round(float(scores[i]), args.digit)

    if args.report_file:
        with open(args.report_file, 'w') as f:
            print(confusion_matrix(labels, preds), file=f)
            print(classification_report(labels, preds, digits=4), file=f)
            print(predictions.metrics, file=f)

    if args.prediction_file:
        with open(args.prediction_file, 'w') as f:
            for d in json_list:
                print(json.dumps(d, ensure_ascii=False), file=f)
