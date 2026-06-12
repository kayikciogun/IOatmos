#!/bin/bash
# ============================================================
#  🎬 IOatmos — Otomatik Sinematik Ses Tasarımcısı
#  Ana Menü (CLI)
# ============================================================

cd "$(dirname "$0")"

# Renkler
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

REQUIRED_PYTHON="3.12"

function brew_shellenv_if_needed() {
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -f /usr/local/bin/brew ]]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
}

function find_python312() {
    brew_shellenv_if_needed
    local candidates=()
    if command -v brew &>/dev/null; then
        local brew_prefix
        brew_prefix="$(brew --prefix python@3.12 2>/dev/null || true)"
        if [[ -n "$brew_prefix" ]]; then
            candidates+=("$brew_prefix/bin/python3.12")
        fi
    fi
    candidates+=(
        "/opt/homebrew/bin/python3.12"
        "/opt/homebrew/opt/python@3.12/bin/python3.12"
        "/usr/local/bin/python3.12"
        "/usr/local/opt/python@3.12/bin/python3.12"
        "python3.12"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

function venv_python_version() {
    if [[ -x ".venv/bin/python" ]]; then
        .venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null
    fi
}

function venv_pip() {
    .venv/bin/python -m pip "$@"
}

function install_python_deps() {
    venv_pip install --upgrade pip
    venv_pip install -r requirements.txt
    venv_pip install --no-deps laion-clap==1.1.6
    venv_pip install pyaaf2 scenedetect opencv-python ffprobe
}

function ensure_venv() {
    local python312
    local recreated=0
    python312="$(find_python312)" || true
    if [[ -z "$python312" ]]; then
        echo -e "${RED}❌ Python ${REQUIRED_PYTHON} bulunamadı.${NC}"
        echo -e "   Kurulum menüsünden ${BLUE}[2] Kurulumu Yap${NC} seçerek python@3.12 kurulmasını sağlayın."
        return 1
    fi

    local current_ver=""
    if [[ -d ".venv" ]]; then
        current_ver="$(venv_python_version)"
    fi

    if [[ -n "$current_ver" && "$current_ver" != "$REQUIRED_PYTHON" ]]; then
        echo -e "  ${YELLOW}⚠️  Mevcut ortam Python ${current_ver} — Python ${REQUIRED_PYTHON} ile yeniden oluşturuluyor...${NC}"
        rm -rf .venv
        current_ver=""
        recreated=1
    fi

    if [[ ! -d ".venv" ]]; then
        echo -e "  ⏳ Python ortamı oluşturuluyor ($("$python312" --version | cut -d' ' -f1-2))..."
        "$python312" -m venv .venv
        echo -e "  ${GREEN}✅ Python ortamı oluşturuldu ($(venv_python_version))${NC}"
        recreated=1
    else
        echo -e "  ${GREEN}✅ Python ortamı hazır (Python $(venv_python_version))${NC}"
    fi

    if [[ "$recreated" -eq 1 ]]; then
        echo -e "  ⏳ Python kütüphaneleri kuruluyor..."
        install_python_deps
        echo -e "  ${GREEN}✅ Kütüphaneler kuruldu${NC}"
    fi
    return 0
}

function show_header() {
    clear
    echo -e "${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                        ║${NC}"
    echo -e "${CYAN}║${BOLD}      🎬 IOatmos — Otomatik Sinematik Ses Tasarımcısı   ${CYAN}║${NC}"
    echo -e "${CYAN}║                                                        ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

function run_setup() {
    echo -e "${YELLOW}━━━ [1/6] Homebrew kontrol ediliyor... ━━━${NC}"
    if ! command -v brew &>/dev/null; then
        echo -e "  ⏳ Homebrew kuruluyor (macOS paket yöneticisi)..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        if [[ -f /opt/homebrew/bin/brew ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile 2>/dev/null
        fi
        echo -e "  ${GREEN}✅ Homebrew kuruldu!${NC}"
    else
        echo -e "  ${GREEN}✅ Homebrew zaten yüklü${NC}"
    fi
    echo ""

    echo -e "${YELLOW}━━━ [2/6] Gerekli programlar kuruluyor... ━━━${NC}"
    brew_shellenv_if_needed
    for pkg in "python@3.12" "ffmpeg" "cmake" "git"; do
        if brew list "$pkg" &>/dev/null; then
            echo -e "  ${GREEN}✅ $pkg zaten yüklü${NC}"
        else
            echo -e "  ⏳ $pkg kuruluyor..."
            brew install "$pkg"
        fi
    done
    echo ""

    echo -e "${YELLOW}━━━ [3/6] Python ortamı hazırlanıyor... ━━━${NC}"
    if ! ensure_venv; then
        read -p "Devam etmek için Enter'a basın..."
        return
    fi
    echo ""

    echo -e "${YELLOW}━━━ [4/6] Yapay zeka kütüphaneleri kuruluyor... ━━━${NC}"
    install_python_deps
    echo -e "  ${GREEN}✅ Tüm kütüphaneler kuruldu${NC}"
    echo ""

    echo -e "${YELLOW}━━━ [5/6] Klasörler hazırlanıyor... ━━━${NC}"
    mkdir -p inputs outputs models
    echo -e "  ${GREEN}✅ Klasörler hazır${NC}"
    echo ""

    echo -e "${YELLOW}━━━ [6/6] Ses kütüphanesi ━━━${NC}"
    if [[ -f "index.npz" ]]; then
        echo -e "  ${GREEN}✅ Ses kütüphanesi zaten indexlenmiş${NC}"
    else
        update_index
    fi

    echo -e "\n${GREEN}✅ KURULUM TAMAMLANDI!${NC}\n"
    read -p "Ana menüye dönmek için Enter'a basın..."
}

function update_index() {
    if ! ensure_venv; then
        read -p "Devam etmek için Enter'a basın..."
        return
    fi

    echo -e "\n${CYAN}📢 Ses kütüphanenizi seçmeniz gerekiyor.${NC}"
    echo "Şimdi bir klasör seçme penceresi açılacak."
    read -p "Devam etmek için Enter'a basın... "
    
    SFX_DIR=$(osascript -e 'try
        set theFolder to POSIX path of (choose folder with prompt "Ses kütüphanenizin bulunduğu klasörü seçin (WAV, MP3, AIFF dosyaları)")
        return theFolder
    on error
        return ""
    end try' 2>/dev/null)
    
    if [[ -n "$SFX_DIR" && -d "$SFX_DIR" ]]; then
        echo -e "\n${YELLOW}⏳ Ses kütüphanesi indexleniyor... (Bu işlem ses sayısına göre vakit alabilir)${NC}"
        if .venv/bin/python src/clap_index.py --audio_dir "$SFX_DIR" --update; then
            echo -e "\n${GREEN}✅ Ses kütüphanesi hazır!${NC}"
        else
            echo -e "\n${RED}❌ Indexleme başarısız oldu. Yukarıdaki hata mesajını kontrol edin.${NC}"
        fi
    else
        echo -e "\n${RED}⚠️  Klasör seçilmedi.${NC}"
    fi
    read -p "Ana menüye dönmek için Enter'a basın..."
}

function run_pipeline() {
    if ! ensure_venv; then
        read -p "Devam etmek için Enter'a basın..."
        return
    fi

    if [[ ! -f "index.npz" ]]; then
        echo -e "${RED}⚠️  Ses kütüphanesi henüz indexlenmemiş! Lütfen önce indexleyin (Seçenek 3).${NC}"
        read -p "Devam etmek için Enter'a basın..."
        return
    fi

    # === VLM MODU SEÇİMİ ===
    echo -e "\n${CYAN}🧠 VLM Analiz Modu Seçin:${NC}"
    echo -e "  ${GREEN}[1] Online${NC}  — OpenRouter API (internet gerekli, hızlı)"
    echo -e "  ${YELLOW}[2] Local${NC}   — llama.cpp server (yerel model, offline)"
    echo ""
    read -p "Seçiminiz (1-2, varsayılan: 1): " vlm_choice

    VLM_MODE_ARG=""
    if [[ "$vlm_choice" == "2" ]]; then
        VLM_MODE_ARG="--vlm-mode local"
        echo -e "\n${YELLOW}🔧 Yerel VLM modu seçildi. models/ klasöründe GGUF + mmproj çifti gereklidir.${NC}"
        echo -e "   (Eğer model yoksa kurulum menüsünden indirebilirsiniz.)"
    else
        VLM_MODE_ARG="--vlm-mode online"
        echo -e "\n${GREEN}🌐 Online VLM modu seçildi (OpenRouter API).${NC}"
    fi
    echo ""

    mkdir -p inputs outputs
    shopt -s nullglob
    VIDEOS=(inputs/*.{mp4,mov,avi,mkv,MP4,MOV,AVI,MKV})
    shopt -u nullglob

    if [[ ${#VIDEOS[@]} -eq 0 ]]; then
        echo -e "${YELLOW}⚠️  'inputs' klasöründe hiç video bulunamadı!${NC}"
        echo "Lütfen işlenecek videoyu (mp4, mov vb.) açılan 'inputs' klasörüne sürükleyin."
        open inputs/
        read -p "Videoyu koyduktan sonra devam etmek için Enter'a basın..."
        
        shopt -s nullglob
        VIDEOS=(inputs/*.{mp4,mov,avi,mkv,MP4,MOV,AVI,MKV})
        shopt -u nullglob
        if [[ ${#VIDEOS[@]} -eq 0 ]]; then
            echo -e "${RED}❌ Hala video bulunamadı. İşlem iptal edildi.${NC}"
            read -p "Ana menüye dönmek için Enter'a basın..."
            return
        fi
    fi

    echo -e "\n${CYAN}🎬 İşlenecek videoyu seçin:${NC}"
    echo -e "  ${YELLOW}[0] Tüm videoları sırayla işle${NC}"
    for i in "${!VIDEOS[@]}"; do
        echo -e "  [$((i+1))] $(basename "${VIDEOS[$i]}")"
    done
    echo ""
    read -p "Seçiminiz (0-${#VIDEOS[@]}): " vid_choice

    local pipeline_ok=0
    if [[ "$vid_choice" == "0" ]]; then
        echo -e "\n${CYAN}🚀 Tüm videolar sırayla işleniyor...${NC}\n"
        if .venv/bin/python src/main.py ${VLM_MODE_ARG}; then
            pipeline_ok=1
        fi
    elif [[ "$vid_choice" =~ ^[0-9]+$ ]] && [[ "$vid_choice" -gt 0 && "$vid_choice" -le ${#VIDEOS[@]} ]]; then
        SELECTED_VIDEO="${VIDEOS[$((vid_choice-1))]}"
        echo -e "\n${CYAN}🚀 Seçilen video işleniyor: $(basename "$SELECTED_VIDEO")${NC}\n"
        if .venv/bin/python src/main.py ${VLM_MODE_ARG} --video "$SELECTED_VIDEO"; then
            pipeline_ok=1
        fi
    else
        echo -e "\n${RED}❌ Geçersiz seçim! İşlem iptal edildi.${NC}"
        read -p "Ana menüye dönmek için Enter'a basın..."
        return
    fi

    if [[ "$pipeline_ok" -eq 1 ]]; then
        echo -e "\n${GREEN}✅ İŞLEM TAMAMLANDI! Çıktılar 'outputs' klasöründedir.${NC}\n"
        open outputs/
    else
        echo -e "\n${RED}❌ İşlem başarısız oldu. Yukarıdaki hata mesajını kontrol edin.${NC}\n"
    fi
    read -p "Ana menüye dönmek için Enter'a basın..."
}

while true; do
    show_header
    echo -e "Lütfen yapmak istediğiniz işlemi seçin:\n"
    echo -e "  ${GREEN}[1] 🚀 Uygulamayı Çalıştır${NC} (Videoyu İşle)"
    echo -e "  ${BLUE}[2] 📦 Kurulumu Yap${NC} (İlk kez kullananlar için)"
    echo -e "  ${YELLOW}[3] 🎵 Ses Kütüphanesini Güncelle${NC} (Yeni sesler eklendiğinde)"
    echo -e "  ${RED}[4] 🚪 Çıkış${NC}"
    echo ""
    read -p "Seçiminiz (1-4): " choice

    case $choice in
        1) run_pipeline ;;
        2) run_setup ;;
        3) update_index ;;
        4) clear; echo -e "${CYAN}Görüşmek üzere! 👋${NC}\n"; exit 0 ;;
        *) echo -e "${RED}❌ Geçersiz seçim! Lütfen 1 ile 4 arasında bir rakam girin.${NC}"; sleep 2 ;;
    esac
done
