#!/bin/bash
# ============================================================
#  🎬 Otomatik Sinematik Ses Tasarımcısı — KURULUM
#  Bu dosyaya çift tıklayın, gerisini otomatik yapar.
# ============================================================

# Scriptin bulunduğu klasöre git
cd "$(dirname "$0")"

clear
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║  🎬 Otomatik Sinematik Ses Tasarımcısı — KURULUM      ║"
echo "║                                                        ║"
echo "║  Bu pencereyi kapatmayın, kurulum otomatik ilerliyor.  ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# ────────────────────────────────────────
# 1. Homebrew
# ────────────────────────────────────────
echo "━━━ [1/6] Homebrew kontrol ediliyor... ━━━"
if ! command -v brew &>/dev/null; then
    echo "  ⏳ Homebrew kuruluyor (macOS paket yöneticisi)..."
    echo "  → Bilgisayar şifrenizi isteyebilir, bu normaldir."
    echo ""
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile 2>/dev/null
    fi
    echo "  ✅ Homebrew kuruldu!"
else
    echo "  ✅ Homebrew zaten yüklü"
fi
echo ""

# ────────────────────────────────────────
# 2. Gerekli programlar
# ────────────────────────────────────────
echo "━━━ [2/6] Gerekli programlar kuruluyor... ━━━"

for pkg in "python@3.12" "ffmpeg" "cmake" "git"; do
    if brew list "$pkg" &>/dev/null; then
        echo "  ✅ $pkg zaten yüklü"
    else
        echo "  ⏳ $pkg kuruluyor..."
        brew install "$pkg"
        echo "  ✅ $pkg kuruldu"
    fi
done
echo ""

# ────────────────────────────────────────
# 3. Python ortamı
# ────────────────────────────────────────
echo "━━━ [3/6] Python ortamı hazırlanıyor... ━━━"

if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
    echo "  ✅ Python ortamı oluşturuldu"
else
    echo "  ✅ Python ortamı zaten hazır"
fi

source .venv/bin/activate
echo ""

# ────────────────────────────────────────
# 4. Python kütüphaneleri
# ────────────────────────────────────────
echo "━━━ [4/6] Yapay zeka kütüphaneleri kuruluyor... ━━━"
echo "  (Bu adım birkaç dakika sürebilir)"
pip install --quiet --upgrade pip 2>/dev/null
pip install --quiet -r requirements.txt 2>/dev/null
pip install --quiet --no-deps laion-clap==1.1.6 2>/dev/null
pip install --quiet pyaaf2 scenedetect opencv-python ffprobe 2>/dev/null
echo "  ✅ Tüm kütüphaneler kuruldu"
echo ""

# ────────────────────────────────────────
# 5. Klasörler
# ────────────────────────────────────────
echo "━━━ [5/6] Klasörler hazırlanıyor... ━━━"
mkdir -p inputs outputs models
echo "  ✅ Klasörler hazır"
echo ""

# ────────────────────────────────────────
# 6. Ses kütüphanesi
# ────────────────────────────────────────
echo "━━━ [6/6] Ses kütüphanesi ━━━"

if [[ -f "index.npz" ]]; then
    echo "  ✅ Ses kütüphanesi zaten indexlenmiş"
else
    echo ""
    echo "  📢 Ses kütüphanenizi seçmeniz gerekiyor."
    echo "     Şimdi bir klasör seçme penceresi açılacak."
    echo ""
    read -p "  Devam etmek için Enter'a basın... "
    
    # macOS klasör seçme penceresi
    SFX_DIR=$(osascript -e 'try
        set theFolder to POSIX path of (choose folder with prompt "Ses kütüphanenizin bulunduğu klasörü seçin (WAV, MP3, AIFF dosyaları)")
        return theFolder
    on error
        return ""
    end try' 2>/dev/null)
    
    if [[ -n "$SFX_DIR" && -d "$SFX_DIR" ]]; then
        echo "  ⏳ Ses kütüphanesi indexleniyor..."
        echo "     Klasör: $SFX_DIR"
        echo "     (Dosya sayısına göre 1-10 dakika sürebilir)"
        echo ""
        python src/clap_index.py --audio_dir "$SFX_DIR"
        echo ""
        echo "  ✅ Ses kütüphanesi indexlendi!"
    else
        echo "  ⚠️  Klasör seçilmedi. Daha sonra 'Çalıştır' dosyasından seçebilirsiniz."
    fi
fi

# ────────────────────────────────────────
# Bitti!
# ────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║   ✅ KURULUM TAMAMLANDI!                               ║"
echo "║                                                        ║"
echo "║   Şimdi:                                               ║"
echo "║   1. Videoyu 'inputs' klasörüne sürükleyin             ║"
echo "║   2. 'Çalıştır' dosyasına çift tıklayın                ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# inputs klasörünü Finder'da aç
open inputs/

echo "Pencereyi kapatabilirsiniz."
read -p ""
