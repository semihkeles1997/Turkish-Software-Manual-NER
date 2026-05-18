from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

def canli_test_baslat():
    # LÜTFEN AŞAĞIDAKİ YOLU KENDİ CHECKPOINT KLASÖRÜNE GÖRE GÜNCELLE
    model_yolu = "./berturk_gold_results/checkpoint-3896" 
    
    print("⏳ Model yükleniyor... Lütfen bekleyin.")
    tokenizer = AutoTokenizer.from_pretrained(model_yolu)
    model = AutoModelForTokenClassification.from_pretrained(model_yolu)

    # aggregation_strategy="simple" ayarı B- ve I- etiketlerini otomatik birleştirir
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    print("="*50)
    print("🚀 OTONOM ASİSTAN NER MODÜLÜ CANLI TEST 🚀")
    print(" Çıkmak için 'q' tuşuna basın.")
    print("="*50)

    while True:
        komut = input("\n🎙️ Kullanıcı Komutu: ")
        
        if komut.lower() == 'q':
            print("Sistem kapatılıyor...")
            break
            
        sonuclar = ner_pipeline(komut)
        
        print("-" * 30)
        print("🔍 AYIKLANAN GÖREVLER:")
        if not sonuclar:
            print("Uyarı: Bu cümlede teknik bir görev bulunamadı.")
        else:
            for sonuc in sonuclar:
                if sonuc['entity_group'] != 'O':
                    print(f"[{sonuc['entity_group']}] ---> {sonuc['word']}")
        print("-" * 30)

if __name__ == "__main__":
    canli_test_baslat()
