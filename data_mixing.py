import random

def read_conll_blocks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        # Cümleleri (blokları) çift boşluğa göre ayır
        content = f.read().strip().split("\n\n")
    return [block for block in content if block.strip()]

def write_conll_blocks(blocks, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for block in blocks:
            f.write(block + "\n\n")

def main():
    print("🚀 Altın Veri Seti okunuyor...")
    
    # Bir önceki adımdan çıkan devasa çoğaltılmış veri dosyamız
    golden_file = "dataset_v7_golden.conll" 
    
    blocks = read_conll_blocks(golden_file)
    
    print(f"Okunan Toplam Cümle: {len(blocks)}")
    
    # Makine öğrenmesinin ezberini bozmak için rastgele karıştırıyoruz!
    random.seed(42) # Tekrar edilebilirlik için sabit tohum
    random.shuffle(blocks)
    
    # Makaleye girecek, eğitimin yapılacağı FİNAL dosyamız
    final_file_name = "dataset_v8_ultimate.conll"
    write_conll_blocks(blocks, final_file_name)
    
    print("="*50)
    print(f"✅ İŞLEM TAMAM! Toplam {len(blocks)} cümle başarıyla harmanlandı.")
    print(f"Model eğitiminde kullanılacak ULTIMATE dosyanız: {final_file_name}")
    print("="*50)

if __name__ == "__main__":
    main()
