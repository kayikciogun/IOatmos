# 🎬 IOatmos — Otomatik Sinematik Ses Tasarımcısı

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

Mac kullanıyorsanız Terminal veya kod bilmenize hiç gerek yok! Sizin için hazırladığımız **tek ve modern** ana menüyü kullanabilirsiniz:

### 1. Başlangıç
1. **`IOatmos.command`** dosyasına çift tıklayın.
2. Karşınıza modern, renkli bir ana menü çıkacaktır.

### 2. Kurulum (Sadece İlk Seferde)
1. Menüden **`[2] 📦 Kurulumu Yap`** seçeneğini seçin.
2. Açılan pencereyi izleyin. Sizin için gerekli tüm programları (Python, FFmpeg, Homebrew, Yapay Zeka Modelleri) otomatik kuracaktır.
3. Kurulumun sonunda size bir **Ses Kütüphanesi Klasörü** soracak. Bilgisayarınızdaki ses efektlerinin olduğu klasörü seçin.
4. "KURULUM TAMAMLANDI" yazısını görünce Ana Menüye dönebilirsiniz.

### 3. Videolarınızı İşleyin
1. **`inputs`** klasörünün içine işlenmesini istediğiniz videoları sürükleyip bırakın.
2. Menüden **`[1] 🚀 Uygulamayı Çalıştır`** seçeneğini seçin.
3. Karşınıza videonuzu seçebileceğiniz bir liste çıkacaktır. İster tek bir videoyu, ister hepsini topluca işleyebilirsiniz.
4. İşlem bitince **`outputs`** klasörü otomatik açılır. İçindeki **`.aaf`** dosyasını doğrudan Logic Pro, Premiere, Pro Tools veya DaVinci Resolve'a sürükleyip atabilirsiniz!

> **İpucu:** Kütüphanenize yeni sesler eklediğinizde menüden **`[3] 🎵 Ses Kütüphanesini Güncelle`** diyerek hızlıca (sadece yenileri) indeksleyebilirsiniz.

---

## 🚀 Performans ve Yenilikler

IOatmos, Apple Silicon (M-Serisi) çiplerin gücünü sonuna kadar kullanmak üzere optimize edilmiştir:

- **Ultra Hızlı İndeksleme:** `soundfile` ve PyTorch `DataLoader` (paralel işlem) altyapısı sayesinde ses kütüphanenizi saniyede 10+ dosya hızıyla indeksler.
- **Akıllı Temizlik:** Dosya isimlerindeki anlamsız kütüphane kodlarını (`3DS02`, `QP92` vb.) otomatik temizler, sadece anlamlı kelimeleri (Caption) kullanarak daha doğru eşleşme yapar.
- **MPS (GPU) Desteği:** Tüm yapay zeka analizleri Apple GPU (Metal Performance Shaders) üzerinde koşar.
- **Güvenli Okuma:** Çok uzun ses dosyalarında bile donma/çökme yaşanmaması için 120 saniyelik akıllı okuma limiti mevcuttur.
- **Dinamik Ağırlıklandırma (Dynamic Weight):** Her ses dosyasının caption kalitesine göre otomatik text/audio ağırlığı belirler. Kısa caption'lar audio-dominant (tw=0.35), zengin caption'lar text-dominant (tw=0.55).
- **UCS Kategori Sınıflandırma:** CLAP zero-shot ile ses dosyalarını otomatik UCS CatID'lerine (WATR-WATERFALL, AMB-Forest, vb.) ayırır. Su kategorilerinde minimum text weight garantisi (0.50) ile yanlış eşleşmeleri önler.
- **Caption Zenginleştirme:** Dosya adlarından akustik perspektiv (underwater, interior, distant) ve kategori bazlı context üreterek CLAP eşleşme doğruluğunu artırır.
- **Negatif Sorgu Desteği:** `--negative` ile istenmeyen ses karakterlerini (örn. "underwater") sonuçlardan çıkarabilirsiniz.

---

## 📂 Proje Yapısı

```
IOatmos/
├── IOatmos.command           # 🚀 ANA MENÜ — Her şeyi buradan yapın
├── src/
│   ├── main.py               # 🎯 Ana pipeline — video'dan AAF'a
│   ├── clap_index.py         # 🎵 Ses kütüphanesini CLAP ile indexle (zero-shot UCS classification)
│   ├── clap_search.py        # 🔍 Hibrit arama (audio + text similarity, dynamic weight)
│   ├── caption_enrichment.py # 📝 Dosya adından zenginleştirilmiş caption üret
│   ├── vlm_processor.py      # 👁️ VLM ile sahne analizi (online/local)
│   ├── aaf_exporter.py       # 🎛️ AAF dosyası oluşturucu (Logic/Pro Tools/DaVinci)
│   ├── update_caption_lengths.py  # 📏 Mevcut index'e caption uzunluğu ekle (bir kez çalıştır)
│   └── VtoF/
│       └── video_analyzer.py # 🎬 Sahne tespiti ve kare çıkarma
├── inputs/                   # 📥 Videolarınızı buraya koyun
├── outputs/                  # 📤 Tüm çıktılar (AAF, manifest, frame'ler)
├── models/                   # 🤖 VLM model dosyaları (GGUF + mmproj)
├── llama.cpp/                # ⚡ Inference motoru (llama-server)
├── sfx/                      # 🎶 Ses efekt kütüphaneniz
├── Docs/                     # 📖 Dokümantasyon
├── requirements.txt          # 📦 Python bağımlılıkları
├── index.npz                 # 💾 CLAP ses index dosyası (otomatik oluşur)
└── README.md                 # 📘 Bu dosya
```

---

## 🎛️ AAF Dosyasını DAW'da Açma

### Logic Pro
1. `File → Open` veya sürükle-bırak
2. 3 audio track görünecek: **Ambience**, **Support**, **Spot FX**

### DaVinci Resolve
1. `File → Import → AAF/EDL/XML`
2. Medya yollarını doğrulayın (ses dosyaları orijinal konumlarında kalmalı)

### Pro Tools
1. `File → Import → Session Data` → AAF seçin

> ⚠️ **Önemli**: AAF dosyası ses dosyalarını **gömer değil, referans verir** (external link). Ses dosyalarınızın orijinal konumlarında kalması gerekir.

---

## ❓ Sık Sorulan Sorular

**S: İnternet gerekli mi?**
H: Sadece ilk kurulumda (model indirme). Sonrasında tamamen offline çalışır.

**S: Ne kadar sürer?**
M1 MacBook'ta: ~30sn (10 sahneli video). Model yükleme bir kere yapılır, sonraki sahneler çok hızlı.

**S: Hangi video formatları desteklenir?**
MP4, MOV, AVI, MKV.

**S: Ses kütüphanem hangi formatta olmalı?**
WAV, MP3, AIFF, FLAC, OGG, M4A — hepsi desteklenir.

**S: `index.npz` ne zaman yeniden oluşturulmalı?**
Kütüphanenize yeni dosyalar eklediğinizde `--update` ile hızlıca güncelleyin. Caption uzunluklarını index'e eklemek için bir kez `python src/update_caption_lengths.py` çalıştırın.

**S: Dinamik weight nedir?**
Her ses dosyasının caption uzunluğuna göre otomatik text/audio ağırlığı belirler. Kısa caption'lar (örn. sadece dosya adı) audio-dominant (tw=0.35), zengin caption'lar (Boom/3DS bext metadata) text-dominant (tw=0.55). Bu, dosya başına en iyi skorlama stratejisini uygular.

---

## 📜 Lisans

Kişisel ve ticari kullanım serbesttir.
