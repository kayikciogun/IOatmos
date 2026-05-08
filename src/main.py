import cv2
import os
import subprocess
import sys
import time
import json
import random
from datetime import datetime

import numpy as np

from VtoF.video_analyzer import detect_scenes, save_scene_frames
from aaf_exporter import create_external_aaf

def check_dependencies():
    """Gerekli Python kütüphanelerini kontrol eder ve eksikleri kurar."""
    required = {
        "aaf2": "pyaaf2",
        "laion_clap": "laion-clap",
        "numpy": "numpy",
        "cv2": "opencv-python",
        "ffprobe": "ffprobe" # ffprobe sistemde olmalı ama pip'te değil, bunu kontrol edeceğiz
    }
    
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            print(f"\n[BİLGİ] '{package}' kütüphanesi eksik. Otomatik kuruluyor...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
                print(f"✅ '{package}' başarıyla kuruldu.")
            except Exception as e:
                print(f"❌ '{package}' kurulurken hata oluştu: {e}")
                print(f"Lütfen manuel kurun: pip install {package}")

# Program başlar başlamaz kütüphaneleri kontrol et
check_dependencies()

def run_cmd(cmd, cwd=None):
    """Verilen terminal komutunu çalıştırır ve hataları yakalar."""
    print(f"\n[SİSTEM KOMUTU] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"HATA: Komut başarısız oldu: {e}")
        sys.exit(1)

def setup_llamacpp():
    """Hiç bilmeyen biri için llama.cpp'yi indirip Apple Mac uyumlu derler."""
    if not os.path.exists("llama.cpp"):
        print("\n---> Adım 1: llama.cpp yapay zeka motoru indiriliyor...")
        run_cmd(["git", "clone", "https://github.com/ggml-org/llama.cpp"])
    else:
        print("\n---> Adım 1: llama.cpp zaten mevcut. Geçiliyor...")

    if not os.path.exists("llama.cpp/build/bin/llama-cli") and not os.path.exists("llama.cpp/build/bin/llama-llava-cli"):
        print("\n---> Adım 2: llama.cpp bilgisayarınızın işlemcisine göre (Mac M-Serisi) derleniyor...")
        print("     Bu işlem 1-2 dakika sürebilir, lütfen bekleyin...")
        run_cmd(["cmake", "-B", "build"], cwd="llama.cpp")
        run_cmd(["cmake", "--build", "build", "--config", "Release", "-j"], cwd="llama.cpp")
    else:
        print("\n---> Adım 2: llama.cpp zaten derlenmiş. Geçiliyor...")

def download_models():
    """Gerekli olan iki modeli (LLM ve Görüntü Adaptörü) otomatik indirir."""
    os.makedirs("models", exist_ok=True)
    base_model_path = "models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
    mmproj_path = "models/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
    
    print("\n---> Adım 3: Modeller kontrol ediliyor...")
    if not os.path.exists(base_model_path):
        print(f"     Ana model indiriliyor (yaklaşık 1.93 GB)...")
        run_cmd(["curl", "-L", "-o", base_model_path, "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"])
        
    if not os.path.exists(mmproj_path):
        print(f"     Görüntü Adaptörü indiriliyor (yaklaşık 1.34 GB)...")
        run_cmd(["curl", "-L", "-o", mmproj_path, "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"])
        
    return base_model_path, mmproj_path

def analyze_with_llamacpp(image_path, base_model, mmproj):
    """Kareyi llama.cpp üzerinden analiz eder (CLI fallback)."""
    prompt = "You are a professional sound designer. Look at this image and describe the ambient sound environment. Reply with ONLY a short filename in this exact format: Location - TimeOfDay - MainSound. Example: Beach - Sunset - Waves - Seagulls. Do NOT write sentences or explanations."
    
    if os.path.exists("llama.cpp/build/bin/llama-mtmd-cli"):
        cli_path = "llama.cpp/build/bin/llama-mtmd-cli"
    elif os.path.exists("llama.cpp/build/bin/llama-llava-cli"):
        cli_path = "llama.cpp/build/bin/llama-llava-cli"
    else:
        cli_path = "llama.cpp/build/bin/llama-cli"
    
    cmd = [
        f"./{cli_path}",
        "-m", base_model,
        "--mmproj", mmproj,
        "--image", image_path,
        "-p", prompt,
        "-n", "30",
        "-c", "2048",
        "--temp", "0.1"
    ]
    
    print(f"\n[{os.path.basename(image_path)}] llama.cpp ile analiz ediliyor...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    if prompt in output:
        output = output.split(prompt)[-1].strip()
    return output


# ============================================================
#   LLAMA SERVER — Model 1 kere yükle, tüm sahneleri işle
# ============================================================

LLAMA_SERVER_PORT = 8787
LLAMA_SERVER_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"
SOUND_DESIGNER_PROMPT = "You are a professional sound designer. Look at this image and describe the general atmospheric ambient sound environment. Focus ONLY on broad environments (e.g. Office, Forest, Hangar, City Street, Factory) and continuous background tones (e.g. Room tone, AC hum, distant traffic, wind, distant birds). DO NOT describe short transient events like 'clicking', 'footsteps', or 'paper crunch'. DO NOT describe visual details like 'sunlight'. Reply with ONLY a short filename in this exact format: BroadLocation - TimeOfDay - MainAmbience - BackgroundAmbience. Example 1: Office - Day - Room Tone - AC Hum. Example 2: Forest - Day - Windy - Distant Birds. Do NOT write sentences or explanations."


def clean_description(text):
    """LLM çıktısını temizler: [img-X] tag'leri, tekrarlar, kesik kelimeleri siler."""
    import re
    
    # 1. [img-X] tag'lerini sil
    text = re.sub(r'\[img-\d+\]', '', text)
    
    # 2. "Location" placeholder'ını sil (model prompt'tan kopyalıyor)
    text = re.sub(r'\bLocation\b', '', text)
    
    # 2. Baştaki/sondaki boşluk ve noktalama temizliği
    text = text.strip().strip('.')
    
    # 3. Tekrarlanan description'ı tespit et ve sadece ilkini al
    # "City - Day - Traffic - Buildings.City - Day - Traffic" gibi durumları yakala
    parts = re.split(r'[.\n]', text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        # İlk parça en temiz olandır
        text = parts[0]
    
    # 4. Sondaki kesik kelimeyi temizle (son - den sonra 3 harften kısa kalan kısım)
    # Örn: "Wind - Birds chir" → "Wind - Birds" (kesik)
    segments = text.split(' - ')
    if len(segments) > 1:
        last = segments[-1].strip()
        # Son segment 3 karakterden kısaysa veya küçük harfle başlıyorsa kesik
        if len(last) < 3:
            segments = segments[:-1]
        text = ' - '.join(segments)
    
    # 5. Baştaki/sondaki tire ve boşluk temizliği
    text = text.strip().strip('-').strip()
    
    return text


def start_llama_server(base_model, mmproj):
    """llama-server'ı arka planda başlatır. Model bir kere yüklenir."""
    import signal
    
    server_path = "llama.cpp/build/bin/llama-server"
    if not os.path.exists(server_path):
        print("   ⚠️ llama-server bulunamadı, CLI moduna geçiliyor...")
        return None
    
    cmd = [
        f"./{server_path}",
        "-m", base_model,
        "--mmproj", mmproj,
        "--port", str(LLAMA_SERVER_PORT),
        "-c", "2048",
        "-n", "30",
        "--temp", "0.1",
        "--log-disable",
    ]
    
    print(f"   ⏳ llama-server başlatılıyor (port {LLAMA_SERVER_PORT})...")
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    
    # Server'ın hazır olmasını bekle (health endpoint)
    import urllib.request
    import urllib.error
    
    max_wait = 120  # saniye (model yükleme uzun sürebilir)
    for i in range(max_wait):
        try:
            req = urllib.request.urlopen(f"{LLAMA_SERVER_URL}/health", timeout=2)
            data = json.loads(req.read().decode())
            if data.get("status") == "ok":
                print(f"   ✅ llama-server hazır ({i+1}sn'de yüklendi)")
                return proc
        except (urllib.error.URLError, ConnectionRefusedError, Exception):
            pass
        
        # Process ölmüş mü kontrol et
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode()
            print(f"   ❌ llama-server başlatılamadı: {stderr[-300:]}")
            return None
        
        time.sleep(1)
    
    print("   ❌ llama-server zaman aşımına uğradı")
    proc.terminate()
    return None


def query_llama_server(image_path):
    """Çalışan llama-server'a HTTP ile görüntü gönderip analiz sonucunu alır."""
    import urllib.request
    import base64
    
    # Görüntüyü base64'e çevir
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    # OpenAI /v1/chat/completions formatı - Chat/Instruct modelleri için en güvenilir yol
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": SOUND_DESIGNER_PROMPT}
                ]
            }
        ],
        "temperature": 0.2,
        "max_tokens": 30
    }).encode("utf-8")
    
    req = urllib.request.Request(
        f"{LLAMA_SERVER_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        result = json.loads(resp.read().decode())
        raw = result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"   ⚠️ Llama-server hatası: {e}")
        raw = ""
        
    return clean_description(raw)


def stop_llama_server(proc):
    """llama-server process'ini durdurur."""
    if proc is None:
        return
    try:
        import signal
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    print("   🛑 llama-server durduruldu")


def get_audio_duration(file_path):
    """Ses dosyasının süresini saniye cinsinden döndürür. FFprobe kullanır."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def search_clap_index(descriptions, index_path="./index.npz", top_k=3, text_weight=0.3):
    """
    Verilen açıklama listesi ile CLAP index'inde arama yapar.
    Her açıklama için top_k sonuç döndürür.
    
    Returns: dict[tag] -> [(path, score), ...]
    """
    import torch
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _original_load(*args, **kwargs)
    torch.load = _patched_load
    
    import laion_clap
    from clap_index import PRESETS, download_ckpt_if_needed
    
    # Index'i yükle
    if not os.path.exists(index_path):
        print(f"❌ CLAP index bulunamadı: {index_path}")
        print("   Önce 'python clap_index.py --audio_dir <ses_klasörü>' çalıştırın.")
        return None
    
    data = np.load(index_path, allow_pickle=True)
    paths = data["paths"].tolist()
    audio_emb = data["embeddings"]
    text_emb_index = data["text_embeddings"] if "text_embeddings" in data.files else None
    preset_name = str(data["preset"]) if "preset" in data.files else "natural"
    
    cfg = PRESETS[preset_name]
    has_text = text_emb_index is not None
    
    print(f"   📦 Index: {len(paths)} ses dosyası, preset={preset_name}")
    
    # Text weight ayarla
    if not has_text and text_weight > 0:
        text_weight = 0.0
    audio_w = 1.0 - text_weight
    
    # CLAP modelini yükle (text encoder için)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"   ⏳ CLAP text encoder yükleniyor ({device})...")
    model = laion_clap.CLAP_Module(
        enable_fusion=cfg["fusion"], amodel=cfg["amodel"], device=device
    )
    if cfg["model_id"] is not None:
        model.load_ckpt(model_id=cfg["model_id"])
    else:
        ckpt = download_ckpt_if_needed(cfg["ckpt_url"], cfg["ckpt_name"])
        model.load_ckpt(ckpt)
    print("   ✅ CLAP hazır")
    
    # Açıklamaları encode et
    query_emb = model.get_text_embedding(descriptions, use_tensor=False)
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    
    # Similarity hesapla
    sim_audio = query_emb @ audio_emb.T
    if has_text and text_weight > 0:
        sim_text = query_emb @ text_emb_index.T
        sim_combined = audio_w * sim_audio + text_weight * sim_text
    else:
        sim_combined = sim_audio
    
    # Her açıklama için top_k sonuç
    results = {}
    for i, desc in enumerate(descriptions):
        scores = sim_combined[i]
        order = np.argsort(-scores)[:top_k]
        results[desc] = [(paths[j], float(scores[j])) for j in order]
    
    return results


def build_manifest(video_path, analysis_results, clap_results, out_dir, video_fps, num_alternatives=3):
    """
    Sahne analizi ve CLAP arama sonuçlarından manifest JSON oluşturur.
    örnek_manifest.json formatına uygun.
    """
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    # Proje adını dosya sistemi uyumlu yap
    project_name = video_basename.replace(" ", "_").replace("(", "_").replace(")", "_")
    
    layer_names = ["Ambience", "Support", "Spot FX"]
    
    scenes_manifest = []
    total_matched = 0
    
    for scene_data in analysis_results:
        scene_id = scene_data["scene_id"]
        desc = scene_data["sound_description"].strip()
        duration = scene_data["duration_seconds"]
        start_tc = scene_data["start_timecode"]
        end_tc = scene_data["end_timecode"]
        
        # Frame hesapla
        start_seconds = sum(float(x) * mult for x, mult in 
                           zip(start_tc.split(":"), [3600, 60, 1]))
        end_seconds = sum(float(x) * mult for x, mult in 
                        zip(end_tc.split(":"), [3600, 60, 1]))
        start_frame = int(start_seconds * video_fps)
        end_frame = int(end_seconds * video_fps)
        
        # CLAP sonuçlarını al
        matches = clap_results.get(desc, []) if clap_results else []
        
        layers = []
        for track_idx in range(min(num_alternatives, len(matches))):
            source_file, score = matches[track_idx]
            
            # Ses dosyasının süresini al
            audio_dur = get_audio_duration(source_file)
            
            # Ses dosyasının başını atla (fade-in / kayıt anonsu koruması)
            # İlk 10 saniyeyi geç, dosya kısa ise mümkün olduğunca ortadan al
            SKIP_START = 10.0  # saniye — başlangıç güvenlik marjı
            
            if audio_dur and audio_dur > duration + SKIP_START:
                # Yeterli alan var: 10sn sonrasından rastgele seç
                min_start = SKIP_START
                max_start = audio_dur - duration
                source_start = round(random.uniform(min_start, max_start), 3)
            elif audio_dur and audio_dur > duration:
                # Dosya kısa ama sahne sığıyor: ortadan al
                max_start = audio_dur - duration
                source_start = round(max_start / 2, 3)  # ortadan
            else:
                source_start = 0.0
            source_end = round(source_start + duration, 3)
            
            layers.append({
                "track": track_idx + 1,
                "layer_name": layer_names[track_idx] if track_idx < len(layer_names) else f"Layer {track_idx+1}",
                "source_file": source_file,
                "source_start_offset": source_start,
                "source_end_offset": source_end,
                "audio_duration": audio_dur if audio_dur else 0.0,
                "clip_duration": round(duration, 3),
                "score": round(score, 4),
                "status": "ok" if audio_dur else "duration_unknown"
            })
            total_matched += 1
        
        scenes_manifest.append({
            "scene_id": scene_id,
            "timeline_start": start_tc,
            "timeline_end": end_tc,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration": round(duration, 3),
            "sound_description": desc,
            "layers": layers
        })
    
    manifest = {
        "project_name": project_name,
        "created_at": datetime.now().strftime("%H%M%S"),
        "video_file": os.path.abspath(video_path),
        "video_fps": video_fps,
        "num_alternatives": num_alternatives,
        "total_scenes": len(analysis_results),
        "scenes_with_sound": total_matched,
        "scenes_skipped": len(analysis_results) - len([s for s in scenes_manifest if s["layers"]]),
        "audio_source_mode": "middle",
        "scenes": scenes_manifest
    }
    
    manifest_path = os.path.join(out_dir, f"{video_basename}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    return manifest_path, manifest


def extract_and_analyze(video_path, base_model, mmproj, index_path="./index.npz"):
    """Videonun sahnelerini tespit edip, analiz edip, CLAP ile ses eşleştirip manifest oluşturur."""
    pipeline_start = time.time()
    
    print("\n---> Adım 4: Video sahneleri yapay zeka ile analiz ediliyor (Scene Detection)...")
    
    # Video isminden dinamik klasör adı oluştur ve 'outputs' klasörü içine al
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join("outputs", f"{video_basename}_frames")
    
    os.makedirs(out_dir, exist_ok=True)
    output_scenes_json = os.path.join(out_dir, "scenes.json")
    
    # Sahneleri tespit et
    detect_scenes(video_path, output_path=output_scenes_json, mode="adaptive")
    
    # Kaydedilen sahneleri oku
    with open(output_scenes_json, "r", encoding="utf-8") as f:
        scenes = json.load(f)
        
    print(f"\n---> Adım 5: Tespit edilen {len(scenes)} sahnenin tam ortalarından kareler çıkartılıyor...")
    save_scene_frames(video_path, scenes, output_dir=out_dir, max_dim=720, samples=("mid",))
    
    # Video FPS bilgisini al
    video_metadata_path = os.path.join(out_dir, "video_metadata.json")
    if os.path.exists(video_metadata_path):
        with open(video_metadata_path, "r", encoding="utf-8") as f:
            video_meta = json.load(f)
        video_fps = video_meta.get("fps", 30.0)
    else:
        video_fps = 30.0
    
    # JSON çıktısı için sonuçları tutacağımız liste
    analysis_results = []
    
    print("\n---> Adım 6: Çıkarılan sahneler için Ses Tasarımı yapılıyor...")
    
    # Server modunu dene — model 1 kere yüklenecek
    server_proc = start_llama_server(base_model, mmproj)
    use_server = server_proc is not None
    
    if use_server:
        print(f"   🚀 Server modu aktif — model bellekte kalacak")
    else:
        print(f"   📟 CLI modu — her sahne için model yeniden yüklenecek")
    
    for scene in scenes:
        scene_id = scene["scene_id"]
        frame_path = os.path.join(out_dir, f"scene_{scene_id:02d}", "mid.jpg")
        
        if not os.path.exists(frame_path):
            print(f"Uyarı: Kare dosyası bulunamadı ({frame_path}). Atlanıyor...")
            continue
        
        try:
            if use_server:
                print(f"\n[Sahne {scene_id}] analiz ediliyor (server)...")
                description = query_llama_server(frame_path)
            else:
                description = analyze_with_llamacpp(frame_path, base_model, mmproj)
            
            print("-" * 50)
            print(f"Sahne {scene_id} ({scene['start']} - {scene['end']}) Ses Analizi:\n{description}")
            print("-" * 50)
            
            # JSON verisi için listeye ekle
            analysis_results.append({
                "scene_id": scene_id,
                "start_timecode": scene.get("start"),
                "end_timecode": scene.get("end"),
                "duration_seconds": scene.get("duration"),
                "frame_path": frame_path,
                "sound_description": description
            })
            
        except Exception as e:
            print(f"Sahne {scene_id} işlenirken hata oluştu: {e}")
    
    # Server'ı kapat
    stop_llama_server(server_proc)
            
    # Ses analizi JSON olarak dışa aktar
    output_json = os.path.join(out_dir, f"{video_basename}_sound_analysis.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=4)
    
    # === Adım 7: CLAP ile ses dosyası eşleştirme ve manifest oluşturma ===
    print(f"\n---> Adım 7: CLAP veritabanında ses dosyaları aranıyor...")
    
    if not os.path.exists(index_path):
        print(f"   ⚠️  CLAP index ({index_path}) bulunamadı. Manifest oluşturulamadı.")
        print(f"   Önce 'python clap_index.py --audio_dir <ses_klasörü>' çalıştırın.")
        print(f"   Ses analizi kaydedildi: {output_json}")
        return
    
    # Her sahnenin açıklamasını topla
    descriptions = [r["sound_description"].strip() for r in analysis_results]
    
    if not descriptions:
        print("   ⚠️  Analiz sonucu boş, CLAP araması yapılamıyor.")
        return
    
    # CLAP araması yap
    clap_results = search_clap_index(descriptions, index_path=index_path, top_k=3)
    
    if clap_results is None:
        return
    
    # Sonuçları göster
    print(f"\n---> Adım 8: Manifest oluşturuluyor...")
    for desc, matches in clap_results.items():
        print(f"\n   🏷️  '{desc}'")
        for i, (path, score) in enumerate(matches):
            print(f"      {i+1}. {score:+.4f}  {os.path.basename(path)}")
    
    # Manifest oluştur
    manifest_path, manifest = build_manifest(
        video_path, analysis_results, clap_results, out_dir, video_fps
    )
    
    # === Adım 9: AAF dosyası oluştur ===
    print(f"\n---> Adım 9: External-Linked AAF dosyası oluşturuluyor...")
    try:
        aaf_path = create_external_aaf(manifest_path, verbose=True)
        print(f"   ✅ AAF dosyası oluşturuldu: {aaf_path}")
    except Exception as e:
        aaf_path = None
        print(f"   ⚠️  AAF oluşturulamadı: {e}")
    
    pipeline_duration = time.time() - pipeline_start
    
    print(f"\n{'='*60}")
    print(f"✅ Tüm işlemler başarıyla tamamlandı!")
    print(f"   📁 Çıktı klasörü: {out_dir}")
    print(f"   📋 Manifest: {manifest_path}")
    if aaf_path:
        print(f"   🎛️  AAF: {aaf_path}")
    print(f"   🎬 Toplam sahne: {manifest['total_scenes']}")
    print(f"   🔊 Eşleşen ses: {manifest['scenes_with_sound']}")
    print(f"   ⏱️  Süre: {pipeline_duration:.1f}sn")
    print(f"{'='*60}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="İşlenecek spesifik video yolu")
    args = parser.parse_args()

    print("="*60)
    print(" OTOMATİK SES TASARIMCISI (LLAMA.CPP TABANLI) ".center(60, "="))
    print("="*60)
    print("Hiçbir şey yapmanıza gerek yok, eksik olan her şey otomatik kurulacak.\n")
    
    # 1 ve 2. llama.cpp indir & derle
    setup_llamacpp()
    
    # 3. Modelleri indir
    base_model, mmproj = download_models()
    
    # 4. Videoları bul ve analiz et
    import glob
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    video_files = []
    if args.video:
        if os.path.exists(args.video):
            video_files = [args.video]
        else:
            print(f"\n[HATA] Belirtilen video bulunamadı: {args.video}")
    else:
        # inputs klasöründeki popüler video formatlarını bul
        for ext in ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"]:
            video_files.extend(glob.glob(os.path.join("inputs", ext)))
        
    if not video_files:
        print("\n[BİLGİ] 'inputs' klasöründe hiç video bulunamadı!")
        print("Lütfen analiz etmek istediğiniz videoları 'inputs' klasörünün içine kopyalayın ve programı tekrar çalıştırın.")
    else:
        print(f"\n[BİLGİ] İşlenecek video sayısı: {len(video_files)}")
        for video_path in video_files:
            print(f"\n{'='*50}")
            print(f" YENİ VİDEO İŞLENİYOR: {os.path.basename(video_path)} ")
            print(f"{'='*50}")
            extract_and_analyze(video_path, base_model, mmproj, index_path="./index.npz")
