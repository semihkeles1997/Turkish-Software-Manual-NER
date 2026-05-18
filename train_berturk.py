import sys
import os
import torch
import numpy as np
import gc
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score

os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'
sys.setrecursionlimit(10000)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Eğitim şu cihaz üzerinde yapılacak: {device}")

def read_conll(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n\n")
    sentences, labels = [], []
    for block in content:
        sentence, label_seq = [], []
        for line in block.split("\n"):
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 2:
                sentence.append(parts[0])
                label_seq.append(parts[-1])
        if sentence:
            sentences.append(sentence)
            labels.append(label_seq)
    return sentences, labels

# Tokenizer hatasını çözmek için tokenizer'ı parametre olarak alıyoruz
def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=128)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main():
    print("\n" + "="*60)
    print("🚀 BASELINE MODEL 1: BERTurk EĞİTİMİ BAŞLIYOR (V6 FİNAL VERİ SETİ)")
    print("="*60)

    dataset_name = "dataset_v6_final.conll" 
    
    sentences, labels = read_conll(dataset_name)
    unique_labels = sorted(list(set(label for label_seq in labels for label in label_seq)))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    
    data = {"tokens": sentences, "ner_tags": [[label2id[l] for l in label_seq] for label_seq in labels]}
    hf_dataset = Dataset.from_dict(data)
    hf_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)

    model_checkpoint = "dbmdz/bert-base-turkish-cased"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
    ).to(device)

    # Tokenizer'ı fn_kwargs ile haritalama fonksiyonuna güvenli bir şekilde aktarıyoruz
    tokenized_datasets = hf_dataset.map(
        tokenize_and_align_labels, 
        batched=True, 
        fn_kwargs={"tokenizer": tokenizer}
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [[unique_labels[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        true_labels = [[unique_labels[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    training_args = TrainingArguments(
        output_dir="./berturk_gold_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-05,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()

    final_metrics = trainer.evaluate()
    print("\n" + "="*60)
    print("🏆 BERTurk EĞİTİMİ TAMAMLANDI! FİNAL TEST SKORLARI:")
    print(f"F1 SKORU  : {final_metrics['eval_f1']:.4f}")
    print(f"KESİNLİK  : {final_metrics['eval_precision']:.4f}")
    print(f"DUYARLILIK: {final_metrics['eval_recall']:.4f}")
    print("="*60 + "\n")

    del model, trainer, tokenizer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()