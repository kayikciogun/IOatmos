#!/bin/bash
# ============================================================
#  🎬 Otomatik Sinematik Ses Tasarımcısı — ÇALIŞTIR
#  Bu dosyaya çift tıklayın, videoları işler.
# ============================================================

# Scriptin bulunduğu klasöre git
cd "$(dirname "$0")"

clear
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  🎬 Otomatik Sinematik Ses Tasarımcısı — ÇALIŞTIR     ║"
echo "║                                                        ║"
echo "║  Bu pencereyi kapatmayın, işlemler otomatik ilerliyor. ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Python ortamını aktif et
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
else
    echo "❌ Kurulum bulunamadı! Lütfen önce 'Kur' dosyasına çift tıklayın."
    echo ""
    read -p "Çıkmak için Enter'a basın..."
    exit 1
fi

# Ses kütüphanesi kontrolü
if [[ ! -f "index.npz" ]]; then
    echo "⚠️  Ses kütüphanesi henüz indexlenmemiş!"
    echo "   Lütfen şimdi ses kütüphanesi klasörünüzü seçin."
    echo ""
    # macOS klasör seçme penceresi
    SFX_DIR=$(osascript -e 'try
        set theFolder to POSIX path of (choose folder with prompt "Ses kütüphanenizin bulunduğu klasörü seçin (WAV, MP3, AIFF dosyaları)")
        return theFolder
    on error
        return ""
    end try' 2>/dev/null)
    
    if [[ -n "$SFX_DIR" && -d "$SFX_DIR" ]]; then
        echo "⏳ Ses kütüphanesi indexleniyor... (Bu işlem ses sayısına göre vakit alabilir)"
        python src/clap_index.py --audio_dir "$SFX_DIR"
        echo "✅ Ses kütüphanesi hazır!"
        echo ""
    else
        echo "❌ Klasör seçilmedi. Lütfen önce ses kütüphanenizi indexleyin."
        echo ""
        read -p "Çıkmak için Enter'a basın..."
        exit 1
    fi
fi

# Inputs klasörünü kontrol et
mkdir -p inputs outputs

# Inputs klasöründe video var mı basitçe kontrol edelim
shopt -s nullglob
VIDEOS=(inputs/*.{mp4,mov,avi,mkv,MP4,MOV,AVI,MKV})
shopt -u nullglob

if [[ ${#VIDEOS[@]} -eq 0 ]]; then
    echo "⚠️  'inputs' klasöründe hiç video bulunamadı!"
    echo "   Lütfen işlenecek videoyu (mp4, mov vb.) açılan 'inputs' klasörüne sürükleyin."
    echo ""
    open inputs/
    read -p "Videoyu koyduktan sonra devam etmek için Enter'a basın..."
    
    shopt -s nullglob
    VIDEOS_RECHECK=(inputs/*.{mp4,mov,avi,mkv,MP4,MOV,AVI,MKV})
    shopt -u nullglob
    
    if [[ ${#VIDEOS_RECHECK[@]} -eq 0 ]]; then
        echo "❌ Hala video bulunamadı. Program kapatılıyor."
        exit 1
    fi
fi

echo "🚀 İşlem başlatılıyor..."
echo ""

# Ana programı çalıştır
python src/main.py

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║   ✅ İŞLEM TAMAMLANDI!                                 ║"
echo "║                                                        ║"
echo "║   Çıktılar (AAF dosyası) 'outputs' klasöründedir.      ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Outputs klasörünü Finder'da aç
open outputs/

echo "Pencereyi kapatabilirsiniz."
read -p ""
