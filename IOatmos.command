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
    if [[ ! -d ".venv" ]]; then
        python3 -m venv .venv
        echo -e "  ${GREEN}✅ Python ortamı oluşturuldu${NC}"
    else
        echo -e "  ${GREEN}✅ Python ortamı zaten hazır${NC}"
    fi
    source .venv/bin/activate
    echo ""

    echo -e "${YELLOW}━━━ [4/6] Yapay zeka kütüphaneleri kuruluyor... ━━━${NC}"
    pip install --quiet --upgrade pip 2>/dev/null
    pip install --quiet -r requirements.txt 2>/dev/null
    pip install --quiet --no-deps laion-clap==1.1.6 2>/dev/null
    pip install --quiet pyaaf2 scenedetect opencv-python ffprobe 2>/dev/null
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
    if [[ ! -f ".venv/bin/activate" ]]; then
        echo -e "${RED}❌ Kurulum bulunamadı! Lütfen önce kurulumu tamamlayın (Seçenek 2).${NC}"
        read -p "Devam etmek için Enter'a basın..."
        return
    fi
    source .venv/bin/activate

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
        python src/clap_index.py --audio_dir "$SFX_DIR" --update
        echo -e "\n${GREEN}✅ Ses kütüphanesi hazır!${NC}"
    else
        echo -e "\n${RED}⚠️  Klasör seçilmedi.${NC}"
    fi
    read -p "Ana menüye dönmek için Enter'a basın..."
}

function run_pipeline() {
    if [[ ! -f ".venv/bin/activate" ]]; then
        echo -e "${RED}❌ Kurulum bulunamadı! Lütfen önce kurulumu yapın (Seçenek 2).${NC}"
        read -p "Devam etmek için Enter'a basın..."
        return
    fi
    source .venv/bin/activate

    if [[ ! -f "index.npz" ]]; then
        echo -e "${RED}⚠️  Ses kütüphanesi henüz indexlenmemiş! Lütfen önce indexleyin (Seçenek 3).${NC}"
        read -p "Devam etmek için Enter'a basın..."
        return
    fi

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

    if [[ "$vid_choice" == "0" ]]; then
        echo -e "\n${CYAN}🚀 Tüm videolar sırayla işleniyor...${NC}\n"
        python src/main.py
    elif [[ "$vid_choice" =~ ^[0-9]+$ ]] && [[ "$vid_choice" -gt 0 && "$vid_choice" -le ${#VIDEOS[@]} ]]; then
        SELECTED_VIDEO="${VIDEOS[$((vid_choice-1))]}"
        echo -e "\n${CYAN}🚀 Seçilen video işleniyor: $(basename "$SELECTED_VIDEO")${NC}\n"
        python src/main.py --video "$SELECTED_VIDEO"
    else
        echo -e "\n${RED}❌ Geçersiz seçim! İşlem iptal edildi.${NC}"
        read -p "Ana menüye dönmek için Enter'a basın..."
        return
    fi

    echo -e "\n${GREEN}✅ İŞLEM TAMAMLANDI! Çıktılar 'outputs' klasöründedir.${NC}\n"
    open outputs/
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
