# 🎬 IOatmos — Otomatik Sinematik Ses Tasarımcısı

**Videoyu koy, çalıştır, ses tasarımını al.** Ses eşleştirme tamamen yerel CLAP ile yapılır. VLM sahne analizi için internet bağlantısı (OpenRouter API) gereklidir.

Bu sistem bir videoyu alır, sahnelere ayırır, yapay zeka ile her sahneyi analiz eder, ses kütüphanenizden en uygun sesleri bulur ve doğrudan DAW'ınıza (Logic Pro, Pro Tools, DaVinci Resolve vb.) import edebileceğiniz bir AAF dosyası oluşturur.

---

## 🧠 Nasıl Çalışır?

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐     ┌─────────┐
│  Video       │────▶│ Sahne Tespit │────▶│  VLM Analiz  │────▶│ CLAP     │────▶│  AAF    │
│  (inputs/)   │     │ (SceneDet.)  │     │ (OpenRouter  │     │ Eşleştir │     │ Export  │
└─────────────┘     └──────────────┘     │  API)        │     └──────────┘     └─────────┘
                          │                      │                   │               │
                     scenes.json          sound_analysis.json   manifest.json   sound_design.aaf
```

### Adım adım:

| Adım | Ne Yapıyor | Teknoloji |
|------|-----------|-----------|
| 1 | Videoyu sahnelere ayırır | PySceneDetect (AdaptiveDetector) |
| 2 | Her sahnenin ortasından kare çıkarır | OpenCV |
| 3 | Her kareyi analiz edip ses ortamını tanımlar | OpenRouter API (grok-4.1-fast) |
| 4 | Tanımlamaları ses kütüphanesinde arar | CLAP (text→audio similarity) |
| 5 | Manifest JSON oluşturur | Python |
| 6 | DAW'a import edilebilir AAF dosyası oluşturur | pyaaf2 |

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
- **MPS (GPU) Desteği:** Ses indeksleme ve CLAP eşleştirme Apple GPU (Metal Performance Shaders) üzerinde koşar.
- **Güvenli Okuma:** Çok uzun ses dosyalarında bile donma/çökme yaşanmaması için 120 saniyelik akıllı okuma limiti mevcuttur.
- **Sabit Hibrit Skorlama:** CLAP aramada text ağırlığı sabit %65, audio ağırlığı %35 olarak kalibre edilmiştir. Dosya adı/caption metinleri ile LLM description arasındaki eşleşmeyi önceliklendirir.
- **Saf CLAP Eşleştirme:** Kategori filtreleme veya re-rank bonus olmadan, doğrudan CLAP embedding benzerliği ile arama yapılır. Sadece text/audio hibrit skor belirleyicidir.
- **Dosya Adı Temizleme:** Ses dosyası isimlerindeki anlamsız kütüphane kodlarını (`3DS02`, `QP92` vb.) otomatik temizleyerek CLAP caption'larını optimize eder.

---

## 📂 Proje Yapısı

```
IOatmos/
├── IOatmos.command           # 🚀 ANA MENÜ — Her şeyi buradan yapın
├── src/
│   ├── main.py               # 🎯 Ana pipeline — video'dan AAF'a
│   ├── clap_index.py         # 🎵 Ses kütüphanesini CLAP ile indexle (zero-shot UCS classification)
│   ├── clap_search.py        # 🔍 Hibrit arama (audio + text similarity, text_weight=0.65)
│   ├── vlm_processor.py      # 👁️ VLM ile sahne analizi (OpenRouter API)
│   ├── vlm_processor.py      # 👁️ VLM ile sahne analizi (OpenRouter API)
│   ├── aaf_exporter.py       # 🎛️ AAF dosyası oluşturucu (Logic/Pro Tools/DaVinci)
│   └── VtoF/
│       └── video_analyzer.py # 🎬 Sahne tespiti ve kare çıkarma
├── inputs/                   # 📥 Videolarınızı buraya koyun
├── outputs/                  # 📤 Tüm çıktılar (AAF, manifest, frame'ler)
├── models/                   # 💾 CLAP checkpoint dosyası (otomatik indirilir)
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
H: VLM sahne analizi (Adım 3) için internet bağlantısı ve OpenRouter API anahtarı gereklidir. Ses eşleştirme tamamen yerel CLAP ile yapılır, offline çalışır.

**S: Ne kadar sürer?**
M1 MacBook'ta: Sahne analizi ~5-10sn/sahne (API bağlantı hızına bağlı), ses eşleştirme ~1-2sn/sahne.

**S: Hangi video formatları desteklenir?**
MP4, MOV, AVI, MKV.

**S: Ses kütüphanem hangi formatta olmalı?**
WAV, MP3, AIFF, FLAC, OGG, M4A — hepsi desteklenir.

**S: `index.npz` ne zaman yeniden oluşturulmalı?**
Kütüphanenize yeni dosyalar eklediğinizde `--update` ile hızlıca güncelleyin. Sadece yeni dosyalar embed edilir, mevcut index korunur.

**S: CLAP aramada text/audio oranı nedir?**
Sabit `text_weight=0.65` kullanılır. Yani %65 text (caption/description) benzerliği, %35 audio spektrogram benzerliği. Bu, LLM'in çıkardığı "waves crashing surf roar" gibi metnin, ses dosyasının zenginleştirilmiş caption'ıyla eşleşmesini önceliklendirir.

---

## 📜 Lisans

Kişisel ve ticari kullanım serbesttir.
