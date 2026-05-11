import os
import subprocess
import sys
import warnings

# HF Hub uyarılarını sustur (token olmadan istek uyarıları)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

def check_dependencies():
    """Gerekli Python kütüphanelerini kontrol eder ve eksikleri kurar."""
    required = {
        "aaf2": "pyaaf2",
        "laion_clap": "laion-clap",
        "numpy": "numpy",
        "cv2": "opencv-python",
        "aiohttp": "aiohttp"
    }
    
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            print(f"\n[BİLGİ] '{package}' kütüphanesi eksik. Otomatik kuruluyor...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True,
                             capture_output=True)
                print(f"✅ '{package}' başarıyla kuruldu.")
            except Exception as e:
                print(f"❌ '{package}' kurulurken hata oluştu: {e}")
                print(f"Lütfen manuel kurun: pip install {package}")

import cv2
import time
import json
import random
import re
import torch
from datetime import datetime
from pathlib import Path
import numpy as np

from VtoF.video_analyzer import detect_scenes, save_scene_frames
from aaf_exporter import create_embedded_aaf
from vlm_processor import (
    start_llama_server, query_llama_server, stop_llama_server,
    analyze_with_llamacpp_cli, analyze_scenes_batch, SIMPLE_DESIGNER_PROMPT
)
from clap_search import smart_search, load_index as load_clap_index

# .env dosyasını yükle
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

def _model_rank(base_path: str) -> float:
    """Bir model dosyasından tahmini parametre sayısı/quality puanı çıkarır.
    Dosya adından parametre boyutunu (3B, 7B, 9B, 72B vb.) bulur.
    BF16/Q8_0/Q6_K/Q4_K_M sıralaması ile quant kalitesini düşürür.
    """
    name = Path(base_path).name.lower()

    # 1. Parametre boyutunu adından çıkar (ilk bulunan sayı+B)
    import re
    param_match = re.search(r'(\d+\.?\d*)\s*[bB]', name)
    if param_match:
        params = float(param_match.group(1))
    else:
        # Fallback: dosya boyutundan tahmin (BF16 ~2B/param GB)
        size_gb = Path(base_path).stat().st_size / 1024**3
        if size_gb > 5:
            params = 7.0
        elif size_gb > 2.5:
            params = 4.0
        elif size_gb > 1.5:
            params = 3.0
        else:
            params = 1.5

    # 2. Quantization kalitesi skoru (BF16 en iyi, Q4 en kötü)
    quant_score = 1.0
    if 'bf16' in name or 'f16' in name:
        quant_score = 1.0
    elif 'q8_0' in name:
        quant_score = 0.9
    elif 'q6_k' in name:
        quant_score = 0.85
    elif 'q5_k' in name:
        quant_score = 0.8
    elif 'q4_k' in name or 'q4_k_m' in name:
        quant_score = 0.75
    elif 'q4_0' in name:
        quant_score = 0.7
    elif 'q3_k' in name:
        quant_score = 0.65
    elif 'q2_k' in name:
        quant_score = 0.5

    # 3. Overall rank = params * quant_score
    # Örn: 9B Q6_K = 9 * 0.85 = 7.65
    # Örn: 4B BF16 = 4 * 1.0 = 4.0
    return params * quant_score


def detect_available_models():
    """models/ klasöründe mevcut GGUF + mmproj çiftlerini tespit eder."""
    MODELS_DIR = Path("models")
    candidates = []

    # 1. Alt klasörler
    for subdir in MODELS_DIR.iterdir():
        if not subdir.is_dir():
            continue
        ggufs = list(subdir.glob("*.gguf"))
        base_candidates = [f for f in ggufs if "mmproj" not in f.name.lower()]
        mmproj_candidates = [f for f in ggufs if "mmproj" in f.name.lower()]
        if base_candidates and mmproj_candidates:
            base = max(base_candidates, key=lambda p: p.stat().st_size)
            mmproj = max(mmproj_candidates, key=lambda p: p.stat().st_size)
            candidates.append({
                "base": str(base),
                "mmproj": str(mmproj),
                "name": subdir.name,
                "rank": _model_rank(str(base)),
                "type": "folder"
            })

    # 2. Root klasör
    root_ggufs = list(MODELS_DIR.glob("*.gguf"))
    root_base = [f for f in root_ggufs if "mmproj" not in f.name.lower()]
    root_mmproj = [f for f in root_ggufs if "mmproj" in f.name.lower()]
    if root_base and root_mmproj:
        base = max(root_base, key=lambda p: p.stat().st_size)
        mmproj = max(root_mmproj, key=lambda p: p.stat().st_size)
        candidates.append({
            "base": str(base),
            "mmproj": str(mmproj),
            "name": base.stem[:20],
            "rank": _model_rank(str(base)),
            "type": "root"
        })

    # RANK' e göre sırala (yüksek rank = daha iyi model)
    candidates.sort(key=lambda x: x["rank"], reverse=True)
    return candidates


def select_model_interactive(candidates, auto_select_best=False):
    """Kullanıcıya mevcut modelleri gösterir ve her işlemde seçim yapar.

    - Tek model varsa otomatik seçilir (sorulmaz).
    - Birden fazla varsa numaralı menü gösterilir.
    - auto_select_best=True yapılırsa en yüksek ranklıyı otomatik seçer
      (otomatik pipeline'lar için).
    """
    if not candidates:
        return None, None

    best = candidates[0]

    print(f"\n{'='*60}")
    print(" MEVCUT SİNEMA VLM MODELLERİ ".center(60, "="))
    print(f"{'='*60}")
    for i, c in enumerate(candidates, 1):
        marker = " ★ EN YÜKSEK KALİTE" if i == 1 else ""
        # Parametre ismini dosya adından çıkar
        param_str = re.search(r'\d+\.?\d*[Bb]', Path(c["base"]).name)
        param_info = param_str.group(0) if param_str else "?B"
        # Dosya boyutunu GB olarak hesapla
        base_size = Path(c["base"]).stat().st_size / (1024**3)
        print(f"   [{i}] {c['name']:<35} | {param_info} | {base_size:.1f}GB | rank={c['rank']:.1f}{marker}")
    print(f"{'='*60}")

    # Tek model varsa veya auto_select_best aktifse → otomatik
    if len(candidates) == 1 or auto_select_best:
        print(f"\n   📦 Otomatik seçim: {best['name']} (rank={best['rank']:.1f})")
        return best["base"], best["mmproj"]

    # Çoklu seçim (her video işlemede sor)
    while True:
        choice = input(f"   🎬 Hangi model ile devam etmek istiyorsunuz? (1-{len(candidates)}): ").strip()
        if not choice:
            print(f"   ⚠️ Boş giriş. En iyi model seçiliyor: {best['name']}")
            return best["base"], best["mmproj"]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(candidates):
                selected = candidates[idx]
                print(f"   ✅ Seçilen model: {selected['name']}")
                return selected["base"], selected["mmproj"]
            else:
                print(f"   ⚠️ Lütfen 1-{len(candidates)} arası bir sayı girin.")
        except ValueError:
            print(f"   ⚠️ Geçersiz giriş. Lütfen bir sayı girin.")


def download_models():
    """Otomatik indirme yerine önce mevcut modelleri tespit eder."""
    os.makedirs("models", exist_ok=True)

    candidates = detect_available_models()
    if candidates:
        return select_model_interactive(candidates, auto_select_best=True)

    # Fallback: Otomatik indir
    print("\n---> Adım 3: Hiç model bulunamadı. Varsayılan model indiriliyor...")
    base_model_path = "models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
    mmproj_path = "models/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
    if not os.path.exists(base_model_path):
        run_cmd(["curl", "-L", "-o", base_model_path,
                 "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"])
    if not os.path.exists(mmproj_path):
        run_cmd(["curl", "-L", "-o", mmproj_path,
                 "https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"])
    return base_model_path, mmproj_path


# ======= VLM Fonksiyonları vlm_processor.py dosyasına taşındı.

def get_audio_duration(file_path):
    """Ses dosyasının süresini saniye cinsinden döndürür.
    Önce ffprobe binary, sonra ffprobe Python modülü, son soundfile fallback.
    """
    # 1. ffprobe binary
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        pass

    # 2. ffprobe Python module fallback
    try:
        import ffprobe
        probe = ffprobe.FFProbe(file_path)
        for stream in probe.streams:
            dur = stream.duration
            if dur:
                return float(dur)
    except Exception:
        pass

    # 3. soundfile fallback
    try:
        import soundfile as sf
        info = sf.info(file_path)
        return info.duration
    except Exception:
        pass

    return None


def search_clap_index_layered(analysis_results, index_path="./index.npz", top_k=3):
    """
    Layer-based arama: Her sahne için 3 ayrı Layer (background/foreground/detail)
    için ayrı arama yapar.
    
    Döner: {scene_id: {"background": [(path, score), ...], "foreground": [...], "detail": [...]}}
    """
    import laion_clap
    from clap_index import PRESETS, download_ckpt_if_needed
    
    idx = load_clap_index(index_path)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"⏳ CLAP Text Encoder yükleniyor...")
    model = laion_clap.CLAP_Module(enable_fusion=True, amodel='HTSAT-tiny', device=device)
    ckpt = download_ckpt_if_needed(PRESETS["natural"]["ckpt_url"], PRESETS["natural"]["ckpt_name"])
    model.load_ckpt(ckpt)

    scene_layers = {}
    total_searches = 0
    
    for res in analysis_results:
        sid = res.get("scene_id", 0)
        layers = res.get("layers", [])
        
        if not layers:
            continue
        
        scene_layers[sid] = {}
        
        for layer in layers:
            ltype = layer.get("type", "general").lower().strip()
            query = layer.get("query", "").strip()
            if not query:
                continue
            
            # Normalize layer types
            if ltype in ("ambience", "bg", "env", "environment", "room"):
                ltype = "background"
            elif ltype in ("movement", "action", "foley", "movement", "object"):
                ltype = "foreground"
            elif ltype in ("texture", "detail", "fx", "spatial", "reverb"):
                ltype = "detail"
            
            search_res = smart_search(
                query=query,
                idx=idx,
                model=model,
                top_k=top_k,
                text_weight=1.0,
                layer_type=ltype,
                bm25_boost_weight=0.25,
                use_bm25=True,
            )
            
            hits = [(r["path"], r["score"]) for r in search_res["results"]]
            scene_layers[sid][ltype] = hits
            total_searches += 1
        
    print(f"   ✅ {len(scene_layers)} sahne için {total_searches} layer araması tamamlandı.")
    return scene_layers


def build_manifest_top3(video_path, analysis_results, scene_layers, out_dir, video_fps):
    """
    Layer-based manifest.
    scene_layers: {scene_id: {"background": [(path, score), ...], "foreground": [...], "detail": [...]}}
    Her layer tipi ayrı bir track'e (kanala) yerleştirilir.
    silence_required sahnesinde hiçbir ses eklenmez.
    """
    SKIP_START = 10.0

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    project_name = video_basename.replace(" ", "_").replace("(", "_").replace(")", "_")

    scenes_manifest = []
    total_matched = 0

    LAYER_TRACK_MAP = {
        "background": 1,
        "foreground": 2,
        "detail": 3,
    }

    LAYER_LABEL_MAP = {
        "background": "BG",
        "foreground": "FG",
        "detail": "FX",
    }

    for scene_data in analysis_results:
        scene_id = scene_data["scene_id"]
        duration = scene_data.get("duration_seconds") or 0.0
        start_tc = scene_data.get("start_timecode", "00:00:00.000")
        end_tc = scene_data.get("end_timecode", "00:00:00.000")
        silence = scene_data.get("silence_required", False)

        def _tc_to_sec(tc):
            try:
                h, m, s = tc.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
            except Exception:
                return 0.0

        start_sec = _tc_to_sec(start_tc)
        end_sec = _tc_to_sec(end_tc)
        start_frame = int(start_sec * video_fps)
        end_frame = int(end_sec * video_fps)

        layers = []
        used_files = set()  # Same exact file
        used_roots = set()  # Same root name (e.g. water_01.wav and water_02.wav)

        if not silence:
            layer_data = scene_layers.get(scene_id, {})
            
            for layer_type in ("background", "foreground", "detail"):
                hits = layer_data.get(layer_type, [])
                if not hits:
                    continue

                # Pick first hit that hasn't been used (exact or similar root)
                chosen = None
                for path, score in hits:
                    if path in used_files:
                        continue
                    
                    # Extract root name to avoid similar files (e.g. water_01.wav, water_02.wav)
                    filename = os.path.basename(path).lower()
                    base_name = os.path.splitext(filename)[0]
                    root = re.sub(r'([-_]v?\d+|_?\d+k|_take\d+|\d+)$', '', base_name).strip('-_ ')
                    
                    if root in used_roots:
                        continue
                    
                    chosen = (path, score, root)
                    break  # Found valid candidate, stop searching

                if not chosen:
                    continue

                path, score, root = chosen
                used_files.add(path)
                used_roots.add(root)

                audio_dur = get_audio_duration(path)
                if not audio_dur:
                    continue

                if audio_dur > duration + SKIP_START:
                    source_start = round(random.uniform(SKIP_START, audio_dur - duration), 3)
                elif audio_dur > duration:
                    source_start = round((audio_dur - duration) / 2, 3)
                else:
                    source_start = 0.0
                source_end = round(source_start + duration, 3)

                confidence = (
                    "excellent" if score > 80 else
                    "high" if score > 65 else
                    "medium" if score > 45 else
                    "low" if score > 30 else
                    "poor"
                )

                track_num = LAYER_TRACK_MAP[layer_type]
                layers.append({
                    "track": track_num,
                    "layer_name": LAYER_LABEL_MAP[layer_type],
                    "layer_type": layer_type,
                    "source_file": path,
                    "source_start_offset": source_start,
                    "source_end_offset": source_end,
                    "audio_duration": audio_dur,
                    "clip_duration": round(duration, 3),
                    "score": round(score, 2),
                    "confidence": confidence,
                    "status": "ok",
                })
                total_matched += 1

        scenes_manifest.append({
            "scene_id": scene_id,
            "timeline_start": start_tc,
            "timeline_end": end_tc,
            "duration": round(duration, 3),
            "category": scene_data.get("category", ""),
            "silence_required": silence,
            "layers": layers,
        })

    manifest = {
        "project_name": project_name,
        "created_at": datetime.now().strftime("%H%M%S"),
        "video_file": os.path.abspath(video_path),
        "video_fps": video_fps,
        "pipeline_version": "layered-v1",
        "total_scenes": len(analysis_results),
        "scenes_with_sound": total_matched,
        "silent_scenes": sum(1 for s in scenes_manifest if s["silence_required"]),
        "scenes": scenes_manifest,
    }

    manifest_path = os.path.join(out_dir, f"{video_basename}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest_path, manifest





def extract_and_analyze(video_path, base_model, mmproj, vlm_mode="online", index_path="./index.npz"):
    """Videonun sahnelerini tespit edip, analiz edip, CLAP ile ses eşleştirip manifest oluşturur."""
    pipeline_start = time.time()
    
    print("\n---> Adım 4: Video sahneleri yapay zeka ile analiz ediliyor (Scene Detection)...")
    
    # Video isminden dinamik klasör adı oluştur ve 'outputs' klasörü içine al
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join("outputs", f"{video_basename}_frames")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Çıktı klasörlerini hazırla
    json_dir = os.path.join(out_dir, "jsonlar")
    frame_dir = os.path.join(out_dir, "sahneler")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)

    # === Adım 5: Sahne Tespiti ===
    scenes_json = os.path.join(json_dir, "scenes.json")
    detect_scenes(video_path, output_path=scenes_json, min_scene_length=0.5)
    
    with open(scenes_json, "r") as f:
        scenes = json.load(f)
    
    # Kareleri çıkar (Sahneler klasörüne)
    save_scene_frames(video_path, scenes, output_dir=frame_dir, samples=["mid"])
    
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

    print("\n---> Adım 6: Sahneler için sinematik ses tasarımı yapılıyor (Faz 1)...")

    # === Adım 6: Sahne Analizi (Batch/Online veya Sequential/Local) ===
    # Tüm kare yollarını ve meta verileri topla
    scene_items = []
    for scene in scenes:
        scene_id = scene["scene_id"]
        frame_path = os.path.join(frame_dir, f"scene_{scene_id:02d}.jpg")
        blank_info = scene.get("blank_info", {}) or {}
        
        scene_items.append({
            "scene_id": scene_id,
            "frame_path": frame_path,
            "start_timecode": scene.get("start"),
            "end_timecode": scene.get("end"),
            "duration_seconds": scene.get("duration"),
            "blank_info": blank_info,
            "temporal_evolution": "static"
        })

    print(f"\n---> Adım 6: {len(scene_items)} sahne analiz ediliyor (Mod: {vlm_mode.upper()})...")
    analysis_results = analyze_scenes_batch(
        scene_items,
        mode=vlm_mode,
        base_model=base_model,
        mmproj=mmproj,
        max_workers=8,
    )

    # JSON çıktı (jsonlar klasörüne)
    output_json = os.path.join(json_dir, f"{video_basename}_sound_analysis.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=4)

    # === Adım 7: Layer-based CLAP eşleştirme ===
    scene_layers = {}
    if analysis_results:
        try:
            scene_layers = search_clap_index_layered(
                analysis_results, index_path=index_path, top_k=3
            )
        except Exception as e:
            print(f"   ⚠️ CLAP arama hatası: {e}")

    # === Manifest oluştur ===
    print(f"\n---> Adım 8: Manifest oluşturuluyor...")
    manifest_path, manifest = build_manifest_top3(
        video_path, analysis_results, scene_layers, out_dir, video_fps
    )

    # === Manifest Özetini Yazdır (Kullanıcının isteği üzerine) ===
    print(f"\n{'─'*60}")
    print(f" 📋 MANİFEST ÖZETİ (Sahne Bazlı Eşleşmeler) ".center(60, " "))
    print(f"{'─'*60}")
    for s in manifest["scenes"]:
        sid = s["scene_id"]
        layers = s["layers"]
        
        print(f"\n🎬 [Sahne {sid:02d}]")
        if s["silence_required"]:
            print(f"   🔇 SESSİZ SAHNE")
        elif not layers:
            print(f"   ⚠️ Eşleşen ses bulunamadı!")
        else:
            for l in layers:
                t = l["track"]
                score = l["score"]
                fname = os.path.basename(l["source_file"])
                label = l.get("layer_name", f"Ch{t}")
                print(f"   [{label}] {fname} (Skor: {score:.1f}/100)")
    print(f"\n{'─'*60}")
    
    # === Adım 9: AAF dosyası oluştur (48kHz embed) ===
    print(f"\n---> Adım 9: Embedded AAF dosyası oluşturuluyor (sesler 48kHz'e çevrilip gömülecek)...")
    try:
        # AAF oluştururken verbose=False yapıyoruz çünkü yukarıda detaylı özeti zaten bastık
        # Ama AAF içindeki dönüştürme loglarını görmek isterse True kalabilir.
        # Kullanıcı "manifest logları" dediği için yukarıdaki tabloyu ekledik.
        aaf_path = create_embedded_aaf(manifest_path, verbose=False) 
        if aaf_path:
            print(f"   ✅ AAF dosyası oluşturuldu: {aaf_path}")
    except Exception as e:
        aaf_path = None
        print(f"   ⚠️  AAF oluşturulamadı: {e}")
    
    pipeline_duration = time.time() - pipeline_start

    print(f"\n{'='*60}")
    print(f"✅ İŞLEM TAMAMLANDI! (IOatmos v2)")
    print(f"   📁 Çıktı klasörü: {out_dir}")
    print(f"   📋 Manifest: {manifest_path}")
    if aaf_path:
        print(f"   🎛️  AAF: {aaf_path}")
    print(f"   🎬 Toplam sahne: {manifest['total_scenes']}")
    print(f"   🔊 Eşleşen ses: {manifest['scenes_with_sound']}")
    print(f"   🔇 Sessiz sahne: {manifest.get('silent_scenes', 0)}")
    print(f"   ⏱️  Süre: {pipeline_duration:.1f}sn")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Sadece ana process'te çalıştır, subprocess worker'larda tekrar etme
    check_dependencies()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-mode", type=str, choices=["online", "local"], default="online",
                        help="VLM analiz modu: online (OpenRouter API) veya local (llama.cpp server). Default: online")
    parser.add_argument("--video", type=str, help="İşlenecek spesifik video yolu")
    parser.add_argument("--model", type=str, help="Hangi VLM modelini kullan (1, 2, 3...). Belirtilmezse menü sorar.")
    parser.add_argument("--auto-model", action="store_true", help="Her video öncesi model seçim menüsü göster")
    args = parser.parse_args()

    print("="*60)
    print(" OTOMATİK SES TASARIMCISI (LLAMA.CPP TABANLI) ".center(60, "="))
    print("="*60)
    print("Hiçbir şey yapmanıza gerek yok, eksik olan her şey otomatik kurulacak.\n")
    
    # 1 ve 2. llama.cpp indir & derle
    setup_llamacpp()
    
    # 3. Modelleri tespit et (indirme yok, sadece mevcutları bul)
    os.makedirs("models", exist_ok=True)
    model_candidates = detect_available_models()
    
    # 4. VLM Analiz Modu (CLI argümanından)
    vlm_mode = args.vlm_mode
        
    # Mevcut video seçim mantığına vlm_mode bilgisini de ekle
    # process_video fonksiyonunu güncelleyip vlm_mode parametresini ekleyeceğiz
    if not model_candidates:
        print("\n❌ Hiç VLM modeli bulunamadı!")
        print("   Lütfen models/ klasörüne GGUF + mmproj çifti yerleştirin.")
        print("   Örnek: models/qwen3.5/Qwen3.5-9B-Q6_K.gguf")
        print("          models/qwen3.5/mmproj-Qwen3.5-9B-BF16.gguf")
        sys.exit(1)

    # CLI'da --model verildiyse otomatik seç
    base_model, mmproj = None, None
    if args.model:
        try:
            idx = int(args.model) - 1
            if 0 <= idx < len(model_candidates):
                base_model = model_candidates[idx]["base"]
                mmproj = model_candidates[idx]["mmproj"]
                print(f"\n   ✅ CLI'dan seçilen model: {model_candidates[idx]['name']}")
            else:
                print(f"   ⚠️ --model {args.model} geçersiz. Mevcut modeller:")
                for i, c in enumerate(model_candidates, 1):
                    print(f"      {i}. {c['name']}")
                sys.exit(1)
        except ValueError:
            # İsim eşleştirmesi dene
            for c in model_candidates:
                if args.model.lower() in c["name"].lower():
                    base_model, mmproj = c["base"], c["mmproj"]
                    print(f"   ✅ CLI'dan eşleşen model: {c['name']}")
                    break
            if not base_model:
                print(f"   ⚠️ '{args.model}' isimli model bulunamadı.")
                sys.exit(1)
    
    # 4. Videoları bul
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
            
            # --- HER VİDEO ÖNCESİ MODEL SEÇİMİ ---
            if args.auto_model or not base_model:
                # Kullanıcıdan (veya ilk seferde) model seçsin
                base_model, mmproj = select_model_interactive(model_candidates, auto_select_best=False)
                if not base_model:
                    print("   ❌ Model seçimi iptal edildi.")
                    continue
            
            extract_and_analyze(video_path, base_model, mmproj, vlm_mode=vlm_mode, index_path="./index.npz")
