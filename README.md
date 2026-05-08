# 🎬 Otomatik Sinematik Ses Tasarımcısı

**Videoyu koy, çalıştır, ses tasarımını al.** Tamamen yerel, tamamen otomatik, internet gerektirmez.

Bu sistem bir videoyu alır, sahnelere ayırır, yapay zeka ile her sahneyi analiz eder, ses kütüphanenizden en uygun sesleri bulur ve doğrudan DAW'ınıza (Logic Pro, Pro Tools, DaVinci Resolve vb.) import edebileceğiniz bir AAF dosyası oluşturur.

---

## 🧠 Nasıl Çalışır?

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐     ┌─────────┐
│  Video       │────▶│ Sahne Tespit │────▶│  VLM Analiz  │────▶│ CLAP     │────▶│  AAF    │
│  (inputs/)   │     │ (SceneDet.)  │     │ (Qwen2.5-VL) │     │ Eşleştir │     │ Export  │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────┘     └─────────┘
                          │                      │                   │               │
                     scenes.json          sound_analysis.json   manifest.json   sound_design.aaf
```

### Adım adım:

| Adım | Ne Yapıyor | Teknoloji |
|------|-----------|-----------|
| 1-2 | llama.cpp motorunu indirir ve macOS için derler | git, cmake, Metal |
| 3 | Görüntü anlama modelini indirir (~3.3 GB) | Qwen2.5-VL-3B-Instruct |
| 4 | Videoyu sahnelere ayırır | PySceneDetect (AdaptiveDetector) |
| 5 | Her sahnenin ortasından kare çıkarır | OpenCV |
| 6 | Her kareyi analiz edip ses ortamını tanımlar | llama-server (HTTP API) |
| 7 | Tanımlamaları ses kütüphanesinde arar | CLAP (text→audio similarity) |
| 8 | Manifest JSON oluşturur | Python |
| 9 | DAW'a import edilebilir AAF dosyası oluşturur | pyaaf2 |

---

## ⚡ Hızlı Başlangıç (Hiç Kod Bilmeyenler İçin)

Mac kullanıyorsanız Terminal veya kod bilmenize hiç gerek yok! Sizin için hazırladığımız **çift tıklanabilir** dosyaları kullanabilirsiniz:

### 1. Kurulum (Sadece İlk Seferde)
1. **`Kur.command`** dosyasına çift tıklayın.
2. Açılan siyah pencereyi izleyin. Sizin için gerekli tüm programları (Python, FFmpeg, Homebrew) otomatik kuracaktır. 
   *(Mac şifrenizi girmeniz istenebilir, yazdığınız şifre ekranda görünmez, yazıp Enter'a basın).*
3. Kurulumun sonunda size bir **Ses Kütüphanesi Klasörü** soracak. Bilgisayarınızdaki (WAV, MP3 vb.) ses efektlerinin olduğu klasörü seçin. Seçtiğiniz sesler indekslenecektir.
4. "KURULUM TAMAMLANDI" yazısını görünce pencereyi kapatabilirsiniz.

### 2. Videolarınızı İşleyin (Her Zaman)
1. **`inputs`** klasörünün içine işlenmesini istediğiniz videoyu sürükleyip bırakın.
2. **`Çalıştır.command`** dosyasına çift tıklayın.
3. Arkanıza yaslanın! Sistem videoyu sahnelere bölecek, yapay zeka ile izleyecek ve uygun sesleri bulacaktır.
4. İşlem bitince **`outputs`** klasörü otomatik açılır. İçindeki **`.aaf`** dosyasını doğrudan Logic Pro, Premiere, Pro Tools veya DaVinci Resolve'a sürükleyip atabilirsiniz!

> **Geliştiriciler için:** Terminal üzerinden `bash setup.sh` komutunu veya `python local_sound_designer.py` komutunu kullanarak da sistemi çalıştırabilirsiniz.

`outputs/<video_adı>_frames/` klasöründe şunları bulacaksınız:

| Dosya | Ne İçerir |
|-------|-----------|
| `scenes.json` | Tespit edilen sahne listesi (timecode'lar) |
| `scene_XX/mid.jpg` | Her sahnenin ortasından alınan kare |
| `*_sound_analysis.json` | VLM'nin ürettiği ses ortamı açıklamaları |
| `*_manifest.json` | Sahne-ses eşleştirme detayları (score, offset vb.) |
| `sound_design.aaf` | **DAW'a import edilecek dosya** |

---

## 🎛️ AAF Dosyasını DAW'da Açma

### Logic Pro
1. `File → Open` veya sürükle-bırak
2. 3 audio track görünecek: **Ambience**, **Support**, **Spot FX**
3. Her sahne için en uygun 3 ses otomatik yerleştirilmiştir

### DaVinci Resolve
1. `File → Import → AAF/EDL/XML`
2. Medya yollarını doğrulayın (ses dosyaları orijinal konumlarında kalmalı)

### Pro Tools
1. `File → Import → Session Data` → AAF seçin

> ⚠️ **Önemli**: AAF dosyası ses dosyalarını **gömer değil, referans verir** (external link). Ses dosyalarınızın orijinal konumlarında kalması gerekir.

---

## 📂 Proje Yapısı

```
TEST-CLAP/
├── local_sound_designer.py   # 🎯 Ana pipeline — tek dosya, her şeyi yapar
├── aaf_external.py           # AAF dosyası oluşturucu
├── clap_index.py             # Ses kütüphanesini CLAP ile indexle
├── clap_search.py            # CLAP ile metin→ses araması
├── inspect_aaf.py            # AAF debug aracı
├── requirements.txt          # Python bağımlılıkları
│
├── VtoF/                     # Video-to-Frames modülü
│   ├── video_analyzer.py     # Sahne tespit + kare çıkarma
│   └── frame_extractor.py    # Kare kaydetme yardımcıları
│
├── inputs/                   # 📥 Videolarınızı buraya koyun
├── outputs/                  # 📤 Tüm çıktılar burada
├── models/                   # VLM model dosyaları (otomatik indirilir)
├── llama.cpp/                # Inference motoru (otomatik indirilir)
├── sfx/                      # Ses kütüphanesi (indexlenecek)
└── index.npz                 # CLAP ses index dosyası
```

---

## 🔧 Gelişmiş Kullanım

### CLAP Presetleri

| Preset | Ne Zaman Kullanılır |
|--------|-------------------|
| `natural` ⭐ | Ambience, foley, SFX, doğa sesleri (varsayılan) |
| `natural_fast` | Kısa one-shot sesler (vurma, kırılma) |
| `music` | Müzik sample'ları, drum kit'ler |
| `music_speech` | Müzik + konuşma karışık içerik |

```bash
# Müzik kütüphanesi indexleme
python clap_index.py --audio_dir ~/Samples --index_path ./music.npz --preset music
```

### Manuel CLAP Araması

Pipeline dışında da ses arayabilirsiniz:

```bash
python clap_search.py --tags "rain on tin roof" "wind through trees" --top_k 5
```

### AAF Inspect (Debug)

Oluşturulan AAF dosyasının içeriğini görmek için:

```bash
python inspect_aaf.py outputs/<klasör>/sound_design.aaf
```

---

## 🏗️ Teknik Detaylar

### VLM (Vision-Language Model)
- **Model**: Qwen2.5-VL-3B-Instruct (Q4_K_M quantization)
- **Motor**: llama.cpp (Metal GPU hızlandırma)
- **Mod**: llama-server — model bir kere yüklenir, tüm sahneler HTTP API ile işlenir
- **Prompt**: Ses tasarımcısı rolü — `Location - TimeOfDay - MainSound - BackgroundSound` formatında çıktı

### CLAP (Contrastive Language-Audio Pretraining)
- **Model**: LAION-CLAP (630k+AudioSet+Fusion)
- **Hibrit Arama**: %70 audio embedding + %30 filename embedding
- **Her sahne için top 3 sonuç**: Ambience, Support, Spot FX katmanları

### AAF Export
- **Yöntem**: External-linked (dosyalar gömülmez, referans verilir)
- **Yapı**: 3 sabit audio track (Ambience / Support / Spot FX)
- **Uyumluluk**: Logic Pro, Pro Tools, DaVinci Resolve, Nuendo

### Akıllı Offset Sistemi
Ses dosyalarının başından alınmaz — ilk 10 saniye atlanır (fade-in ve kayıt anonsu koruması). Dosya kısaysa ortadan alınır.

---

## ❓ Sık Sorulan Sorular

**S: İnternet gerekli mi?**
H: Sadece ilk kurulumda (model indirme). Sonrasında tamamen offline çalışır.

**S: Ne kadar sürer?**
M1 MacBook'ta: ~30sn (10 sahneli video). Model yükleme bir kere yapılır, sonraki sahneler çok hızlı.

**S: Hangi video formatları desteklenir?**
MP4, MOV, AVI, MKV.

**S: Ses kütüphanem hangi formatta olmalı?**
WAV, MP3, AIFF, FLAC — hepsi desteklenir.

**S: `index.npz` ne zaman yeniden oluşturulmalı?**
Kütüphanenize yeni dosyalar eklediğinizde `--update` ile güncelleyin.

---

## 📜 Lisans

Kişisel ve ticari kullanım serbesttir.
# IOatmos
