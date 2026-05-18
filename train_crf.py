import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

def read_conll(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n\n")
    sentences = []
    for block in content:
        sentence = []
        for line in block.split("\n"):
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) >= 2:
                sentence.append((parts[0], parts[-1])) # (kelime, etiket)
        if sentence:
            sentences.append(sentence)
    return sentences

# CRF için kelime özelliklerini (Feature Engineering) çıkarma
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:], # Türkçe son ekler çok önemlidir!
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True # Cümle başı

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True # Cümle sonu

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def main():
    print("\n" + "="*60)
    print("🚀 BASELINE MODEL 2: CRF EĞİTİMİ BAŞLIYOR")
    print("="*60)

    # Lütfen dosya adını kendi temiz conll dosyanın adı ile değiştir!
    dataset_name = "dataset_v6_final.conll" 
    sentences = read_conll(dataset_name)
    
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]

    # Derin öğrenme kodumuzdakiyle tamamen aynı tohum (seed=42) ile veriyi bölüyoruz
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    
    print("CRF modeli eğitiliyor (Bu işlem birkaç saniye sürecek)...")
    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)
    
    print("\n" + "="*60)
    print("🏆 CRF EĞİTİMİ TAMAMLANDI! FİNAL TEST SKORLARI:")
    # İlgili olmayan etiketleri ('O') filtreleyip detaylı rapor sunuyoruz
    labels = list(crf.classes_)
    labels.remove('O')
    print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=4))
    print("="*60 + "\n")

if __name__ == "__main__":
    main()