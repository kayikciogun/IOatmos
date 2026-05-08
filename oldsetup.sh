#!/bin/bash
# ============================================================
#  🎬 Otomatik Sinematik Ses Tasarımcısı — Kurulum Scripti
#  Tek komutla her şeyi kurar ve çalıştırır.
#  Kullanım: bash setup.sh
# ============================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo -e "${BOLD}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  🎬 Otomatik Sinematik Ses Tasarımcısı — Kurulum      ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# ────────────────────────────────────────────────────────────
# 1. Homebrew kontrolü
# ────────────────────────────────────────────────────────────
echo -e "${CYAN}[1/6]${NC} Homebrew kontrol ediliyor..."
if ! command -v brew &>/dev/null; then
    echo -e "${YELLOW}  ⏳ Homebrew kuruluyor...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Apple Silicon PATH
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    fi
    echo -e "${GREEN}  ✅ Homebrew kuruldu${NC}"
else
    echo -e "${GREEN}  ✅ Homebrew mevcut${NC}"
fi

# ────────────────────────────────────────────────────────────
# 2. Sistem bağımlılıkları (Python, FFmpeg, CMake, Git)
# ────────────────────────────────────────────────────────────
echo -e "${CYAN}[2/6]${NC} Sistem bağımlılıkları kontrol ediliyor..."

DEPS=("python3" "ffmpeg" "cmake" "git")
BREW_PKGS=("python@3.12" "ffmpeg" "cmake" "git")

for i in "${!DEPS[@]}"; do
    cmd="${DEPS[$i]}"
    pkg="${BREW_PKGS[$i]}"
    if ! command -v "$cmd" &>/dev/null; then
        echo -e "${YELLOW}  ⏳ $pkg kuruluyor...${NC}"
        brew install "$pkg"
        echo -e "${GREEN}  ✅ $pkg kuruldu${NC}"
    else
        echo -e "${GREEN}  ✅ $cmd mevcut $(command -v $cmd)${NC}"
    fi
done

# Python versiyonunu göster
PYTHON_VERSION=$(python3 --version 2>&1)
echo -e "${GREEN}  🐍 $PYTHON_VERSION${NC}"

# ────────────────────────────────────────────────────────────
# 3. Python Virtual Environment
# ────────────────────────────────────────────────────────────
echo -e "${CYAN}[3/6]${NC} Python sanal ortamı hazırlanıyor..."

if [[ ! -d ".venv" ]]; then
    echo -e "${YELLOW}  ⏳ Virtual environment oluşturuluyor...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}  ✅ .venv oluşturuldu${NC}"
else
    echo -e "${GREEN}  ✅ .venv zaten mevcut${NC}"
fi

source .venv/bin/activate

# ────────────────────────────────────────────────────────────
# 4. Python kütüphaneleri
# ────────────────────────────────────────────────────────────
echo -e "${CYAN}[4/6]${NC} Python kütüphaneleri kuruluyor..."

if [[ -f "requirements.txt" ]]; then
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    pip install --quiet --no-deps laion-clap==1.1.6
    pip install --quiet pyaaf2 scenedetect opencv-python ffprobe
    echo -e "${GREEN}  ✅ Tüm Python kütüphaneleri kuruldu${NC}"
else
    echo -e "${RED}  ❌ requirements.txt bulunamadı!${NC}"
    exit 1
fi

# ────────────────────────────────────────────────────────────
# 5. Klasör yapısı
# ────────────────────────────────────────────────────────────
echo -e "${CYAN}[5/6]${NC} Klasör yapısı hazırlanıyor..."
mkdir -p inputs outputs models
echo -e "${GREEN}  ✅ inputs/ outputs/ models/ klasörleri hazır${NC}"

# ────────────────────────────────────────────────────────────
# 6. Ses kütüphanesi indexleme
# ────────────────────────────────────────────────────────────
echo -e "${CYAN}[6/6]${NC} Ses kütüphanesi kontrolü..."

if [[ -f "index.npz" ]]; then
    echo -e "${GREEN}  ✅ CLAP index mevcut (index.npz)${NC}"
else
    echo ""
    echo -e "${YELLOW}  ⚠️  Ses kütüphanesi henüz indexlenmemiş!${NC}"
    echo ""
    echo -e "  Ses dosyalarınızın bulunduğu klasörün tam yolunu girin"
    echo -e "  (örn: ${BOLD}/Users/kullanici/SFX_Library${NC})"
    echo ""
    read -p "  📁 Ses kütüphanesi yolu (boş bırakırsan atlanır): " SFX_DIR
    
    if [[ -n "$SFX_DIR" && -d "$SFX_DIR" ]]; then
        echo -e "${YELLOW}  ⏳ Ses kütüphanesi indexleniyor (bu biraz zaman alabilir)...${NC}"
        python src/clap_index.py --audio_dir "$SFX_DIR"
        echo -e "${GREEN}  ✅ Index oluşturuldu!${NC}"
    elif [[ -n "$SFX_DIR" ]]; then
        echo -e "${RED}  ❌ Klasör bulunamadı: $SFX_DIR${NC}"
        echo -e "  Daha sonra şu komutla indexleyebilirsiniz:"
        echo -e "  ${BOLD}python src/clap_index.py --audio_dir /yol/ses/klasörü${NC}"
    else
        echo -e "  ⏭️  Atlandı. Daha sonra şu komutla indexleyin:"
        echo -e "  ${BOLD}python src/clap_index.py --audio_dir /yol/ses/klasörü${NC}"
    fi
fi

# ────────────────────────────────────────────────────────────
# Kurulum tamamlandı
# ────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  ✅ Kurulum Tamamlandı!                               ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}Kullanım:${NC}"
echo ""
echo -e "  1. Videoyu ${BOLD}inputs/${NC} klasörüne koyun"
echo -e "  2. Çalıştırın:"
echo ""
echo -e "     ${GREEN}source .venv/bin/activate${NC}"
echo -e "     ${GREEN}python src/main.py${NC}"
echo ""
echo -e "  3. Çıktılar ${BOLD}outputs/${NC} klasöründe:"
echo -e "     📋 Manifest JSON  → sahne-ses eşleştirmesi"
echo -e "     🎛️  AAF dosyası   → Logic Pro / DaVinci / Pro Tools'a import edin"
echo ""
echo -e "  ${YELLOW}İlk çalıştırmada llama.cpp ve AI modeli otomatik indirilir (~3.3 GB)${NC}"
echo ""
