import random

def veri_esnet_ve_gizle(girdi_dosyasi, cikti_dosyasi):
    print("🚀 Veri Çoğaltma (Augmentation) ve Gizleme (Dropout) başlatılıyor...")
    
    # Modele öğreteceğimiz yeni evrensel kelimeler
    YENI_YAZILIMLAR = ["Spotify", "Jira", "Steam", "Discord", "AutoCAD", "Figma", "AWS", "Netflix", "ChatGPT", "Slack"]
    YENI_NESNELER = ["zımbırtı", "şeye", "kırmızı butona", "mavi alana", "boşluğa", "o zımbırtıya", "ikona", "sekmesine"]

    with open(girdi_dosyasi, "r", encoding="utf-8") as f:
        lines = f.readlines()

    yeni_satirlar = []
    
    # 1. Tur: Orijinal veriyi olduğu gibi koru (Bozmuyoruz)
    yeni_satirlar.extend(lines)
    yeni_satirlar.append("\n")

    # 2. Tur: Dropout ve Augmentation uygulayarak veriyi çoğalt
    cogaltilmis_cumle = []
    
    for line in lines:
        if line.strip() == "":
            if cogaltilmis_cumle:
                yeni_satirlar.extend(cogaltilmis_cumle)
                yeni_satirlar.append("\n")
                cogaltilmis_cumle = []
            continue

        parts = line.strip().split("\t")
        if len(parts) == 2:
            word, label = parts[0], parts[1]

            # %15 İhtimalle Varlık Gizleme (Entity Dropout)
            if label in ["B-OBJECT", "B-PATH"] and random.random() < 0.15:
                cogaltilmis_cumle.append(f"[UNK]\t{label}\n")
            
            # %10 İhtimalle Kelime Değiştirme (Dictionary Injection)
            elif label == "B-PATH" and random.random() < 0.10:
                cogaltilmis_cumle.append(f"{random.choice(YENI_YAZILIMLAR)}\t{label}\n")
                
            elif label == "B-OBJECT" and random.random() < 0.10:
                cogaltilmis_cumle.append(f"{random.choice(YENI_NESNELER)}\t{label}\n")
                
            else:
                cogaltilmis_cumle.append(f"{word}\t{label}\n")
        else:
            cogaltilmis_cumle.append(line)

    with open(cikti_dosyasi, "w", encoding="utf-8") as f:
        f.writelines(yeni_satirlar)

    print(f"✅ İşlem Tamam! Veri seti esnetildi ve '{cikti_dosyasi}' olarak kaydedildi.")
    print(f"Eski Satır: {len(lines)} | Yeni Satır: {len(yeni_satirlar)}")

# Dosya adlarını kendine göre ayarla (Eğitim verini buraya gir)
girdi = "dataset_v6_final.conll"  # Şu anki en iyi ve harmanlanmış eğitim dosyan
cikti = "dataset_v7_golden.conll" # Yepyeni, makalelik devasa dosyan

if __name__ == "__main__":
    veri_esnet_ve_gizle(girdi, cikti)