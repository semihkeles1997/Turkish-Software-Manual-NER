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

# Olası çevre değişkeni uyarılarını engelleme
os.environ['PYARROW_IGNORE_TIMEZONE'] = '1'
sys.setrecursionlimit(10000)

# Cihaz Tespiti (Mac MPS, CUDA veya CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Eğitim şu cihaz üzerinde yapılacak: {device}")

def read_conll(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HATA: '{file_path}' bulunamadı! Lütfen dosya yolunu kontrol edin.")
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n\n")

    sentences, labels = [], []
    for block in content:
        sentence, label_seq = [], []
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split() 
            if len(parts) >= 2:
                token = parts[0]
                label = parts[-1] 
                sentence.append(token)
                label_seq.append(label)
        if sentence:
            sentences.append(sentence)
            labels.append(label_seq)
    return sentences, labels

def main():
    print("\n" + "="*60)
    print("🚀 BENCHMARK EĞİTİM SÜRECİ BAŞLIYOR (3 FARKLI MODEL)")
    print("="*60)

    # Veri setini okuma
    dataset_name = "dataset_v6_final.conll"
    print(f"Veri seti yükleniyor: {dataset_name}...")
    sentences, labels = read_conll(dataset_name)
    print(f"Toplam okunan cümle sayısı: {len(sentences)}")

    # Sınıfları belirleme
    unique_labels = sorted(list(set(label for label_seq in labels for label in label_seq)))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    print(f"Tespit Edilen Sınıflar: {unique_labels}")

    # Dataset formatına çevirme
    data = {"tokens": sentences, "ner_tags": [[label2id[l] for l in label_seq] for label_seq in labels]}
    hf_dataset = Dataset.from_dict(data)
    
    # Veriyi Eğitim (%80) ve Test (%20) olarak ayırma
    hf_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)

    # Yarıştırılacak Modellerin Listesi
    model_checkpoints = {
        "ELECTRA": "dbmdz/electra-base-turkish-cased-discriminator",
        "XLM-RoBERTa": "xlm-roberta-base",
        "DistilBERTurk": "dbmdz/distilbert-base-turkish-cased"
    }

    # Döngü halinde tüm modelleri eğitme
    for name, checkpoint in model_checkpoints.items():
        print("\n" + "="*60)
        print(f"📦 {name} MODELİ EĞİTİLİYOR ({checkpoint})")
        print("="*60)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForTokenClassification.from_pretrained(
            checkpoint,
            num_labels=len(unique_labels),
            id2label=id2label,
            label2id=label2id
        ).to(device)

        # WordPiece hizalama fonksiyonu
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=128)
            labels_list = []
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
                labels_list.append(label_ids)
            tokenized_inputs["labels"] = labels_list
            return tokenized_inputs

        tokenized_datasets = hf_dataset.map(tokenize_and_align_labels, batched=True, load_from_cache_file=False)

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

        # MAC MPS BELLEK DOSTU TRAIN ARGUMANLARI
        training_args = TrainingArguments(
            output_dir=f"./{name.lower()}_results",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=3e-05,
            per_device_train_batch_size=2,       # Bellek patlamasını önlemek için düşürüldü
            per_device_eval_batch_size=2,        # Değerlendirme de aynı şekilde korundu
            gradient_accumulation_steps=4,       # 2 x 4 = 8 Sanal Batch Size
            num_train_epochs=10,                 # Maksimum 10 epoch
            weight_decay=0.01,
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # Erken durdurma kiti
        )
        
        # Eğitimi başlat
        trainer.train()

        # Model içi final test skorlarını göster
        final_metrics = trainer.evaluate()
        print(f"\n✨ {name} Şirket İçi Final Test Skorları:")
        print(f"   F1-Score: {final_metrics['eval_f1']:.4f} | Kesinlik: {final_metrics['eval_precision']:.4f}")

        # Nihai modeli disk alanına kaydetme
        save_path = f"./saved_model_{name.lower()}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"✅ {name} başarıyla '{save_path}' klasörüne kaydedildi.")

        # BİR SONRAKİ MODELE GEÇMEDEN ÖNCE BELLEĞİ KIRBAÇLAMA (ZORUNLU TEMİZLİK)
        del model
        del trainer
        del tokenizer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "="*60)
    print("🏆 TÜM TRANSFORMER MODELLERİNİN EĞİTİMİ TAMAMLANDI!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()