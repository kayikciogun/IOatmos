#!/usr/bin/env python3
"""
clap_index.py — Audio kütüphanesini CLAP ile embed eder.

Doğal sesler için DEFAULT preset:
  natural (630k+audioset+fusion, ESC50 ~89%, uzun ambience'lar için ideal)

Filename embedding (default ON):
  Profesyonel kütüphanelerde dosya adları içeriği betimler
  (örn. "thunder - heavy rain - weather 1.mp3").
  Bu adları text encoder'dan da geçirip ayrı saklarız.
  Search sırasında --text_weight ile ağırlık ayarlanabilir.

Kullanım:
    python clap_index.py --audio_dir ./sfx
    python clap_index.py --audio_dir ./sfx --preset natural_fast --device mps
    python clap_index.py --audio_dir ./sfx --no_filename  # eski davranış
    python clap_index.py --audio_dir ./sfx --update
"""

import argparse
import glob
import os
import re
import time
from pathlib import Path

# PyTorch 2.6+ ile laion_clap 1.1.6 uyumsuzluğu
import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

import laion_clap
import numpy as np


AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif")

# model_id mapping (laion_clap/hook.py):
#   0 = 630k non-fusion          → 630k-best.pt
#   1 = 630k+audioset non-fusion → 630k-audioset-best.pt
#   2 = 630k fusion              → 630k-fusion-best.pt
#   3 = 630k+audioset fusion     → 630k-audioset-fusion-best.pt
PRESETS = {
    "natural": {
        "model_id": None,
        "amodel": "HTSAT-tiny",
        "fusion": True,
        "ckpt_url": "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt",
        "ckpt_name": "630k-audioset-fusion-best.pt",
        "desc": "Doğal sesler/foley/ambience/SFX (630k+audioset+fusion, ESC50 ~89%)",
    },
    "natural_fast": {
        "model_id": None,
        "amodel": "HTSAT-tiny",
        "fusion": False,
        "ckpt_url": "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt",
        "ckpt_name": "630k-audioset-best.pt",
        "desc": "Doğal sesler, fusion'sız (kısa one-shot, daha hızlı)",
    },
    "music": {
        "model_id": None,
        "amodel": "HTSAT-base",
        "fusion": False,
        "ckpt_url": "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt",
        "ckpt_name": "music_audioset_epoch_15_esc_90.14.pt",
        "desc": "Müzik (sample pack, drum, synth)",
    },
    "music_speech": {
        "model_id": None,
        "amodel": "HTSAT-base",
        "fusion": False,
        "ckpt_url": "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt",
        "ckpt_name": "music_speech_audioset_epoch_15_esc_89.98.pt",
        "desc": "Müzik + konuşma",
    },
}


def find_audio_files(audio_dir: str) -> list[str]:
    files = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(audio_dir, "**", f"*{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(audio_dir, "**", f"*{ext.upper()}"), recursive=True))
    return sorted(set(os.path.abspath(f) for f in files))


def filename_to_caption(path: str) -> str:
    """
    'sfx/weather/thunder - heavy rain - weather 1.mp3' → 'thunder heavy rain weather'

    - Extension'ı at
    - '-', '_', '.' ayırıcılarını boşluğa çevir
    - Sayıları/duplicate boşlukları temizle
    - Lowercase
    """
    name = Path(path).stem  # uzantı yok
    # Tire/altçizgi/nokta → boşluk
    name = re.sub(r"[-_.]+", " ", name)
    # camelCase → boş ayır (CarEngine → Car Engine)
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    # Sayıları çıkar (genelde sadece varyant numarası: "rain 1", "rain 2")
    name = re.sub(r"\b\d+\b", "", name)
    # Çoklu boşlukları sıkıştır
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


def download_ckpt_if_needed(url: str, name: str) -> str:
    local_dir = Path(__file__).parent / "models"
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / name
    
    if target.exists():
        return str(target)
        
    # Check old locations to prevent redownload
    import laion_clap
    old_cache1 = Path.home() / ".cache" / "clap" / name
    old_cache2 = Path(laion_clap.__file__).parent / name
    
    for old_cache in [old_cache1, old_cache2]:
        if old_cache.exists():
            import shutil
            print(f"📦 Model eski konumdan 'models/' klasörüne taşınıyor...")
            shutil.move(str(old_cache), str(target))
            return str(target)

    print(f"⏳ Checkpoint indiriliyor: {name} (~1.8GB)")
    import urllib.request
    urllib.request.urlretrieve(url, target)
    print(f"✅ İndirildi: {target}")
    return str(target)


def load_existing_index(path: str):
    if not os.path.exists(path):
        return {}, None, None, None
    data = np.load(path, allow_pickle=True)
    paths = data["paths"].tolist()
    audio_emb = data["embeddings"]
    text_emb = data["text_embeddings"] if "text_embeddings" in data.files else None
    preset = str(data["preset"]) if "preset" in data.files else None
    audio_dict = dict(zip(paths, audio_emb))
    text_dict = dict(zip(paths, text_emb)) if text_emb is not None else {}
    return audio_dict, text_dict, preset, data


def build_model(preset_name: str, device_override: str = None):
    cfg = PRESETS[preset_name]
    if device_override:
        device = device_override
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"⏳ CLAP yükleniyor: preset={preset_name} ({cfg['desc']})")
    print(f"   amodel={cfg['amodel']}, fusion={cfg['fusion']}, device={device}")
    model = laion_clap.CLAP_Module(
        enable_fusion=cfg["fusion"],
        amodel=cfg["amodel"],
        device=device,
    )
    if cfg["model_id"] is not None:
        model.load_ckpt(model_id=cfg["model_id"])
    else:
        ckpt_path = download_ckpt_if_needed(cfg["ckpt_url"], cfg["ckpt_name"])
        model.load_ckpt(ckpt_path)
    print("✅ Model hazır\n")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--index_path", default="./index.npz")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--preset", default="natural", choices=list(PRESETS.keys()))
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda", "auto"],
        default="auto",
    )
    parser.add_argument(
        "--no_filename",
        action="store_true",
        help="Filename embedding'i kapat (sadece audio embedding kullan)",
    )
    parser.add_argument(
        "--text_batch_size",
        type=int,
        default=32,
        help="Filename text encoding batch boyutu",
    )
    args = parser.parse_args()

    audio_files = find_audio_files(args.audio_dir)
    if not audio_files:
        print(f"❌ {args.audio_dir} içinde ses dosyası yok.")
        return
    print(f"📁 {len(audio_files)} dosya: {args.audio_dir}\n")

    existing_audio, existing_text, prev_preset, _ = {}, {}, None, None
    if args.update and not args.force:
        existing_audio, existing_text, prev_preset, _ = load_existing_index(args.index_path)
        if existing_audio and prev_preset and prev_preset != args.preset:
            print(f"⚠️  Var olan index farklı preset ({prev_preset}). --force kullan.")
            return
        if existing_audio:
            print(f"♻️  Var olan: {len(existing_audio)} dosya")

    to_embed = [f for f in audio_files if f not in existing_audio]
    if not to_embed:
        print("✅ Yapılacak iş yok.")
        return
    print(f"⏳ Embed edilecek: {len(to_embed)} dosya\n")

    device_override = None if args.device == "auto" else args.device
    model = build_model(args.preset, device_override=device_override)

    use_filename = not args.no_filename
    if use_filename:
        captions = [filename_to_caption(f) for f in to_embed]
        # Örnek 3 caption göster
        print(f"📝 Filename → caption örnekleri:")
        for i in range(min(3, len(to_embed))):
            print(f"   {Path(to_embed[i]).name}")
            print(f"   → '{captions[i]}'")
        print()

    all_paths = list(existing_audio.keys())
    all_audio = [existing_audio[p] for p in all_paths]
    all_text = [existing_text.get(p) for p in all_paths] if use_filename else []

    # === Audio embedding ===
    t0 = time.time()
    failed = []
    for i in range(0, len(to_embed), args.batch_size):
        batch = to_embed[i : i + args.batch_size]
        try:
            emb = model.get_audio_embedding_from_filelist(x=batch, use_tensor=False)
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            for p, e in zip(batch, emb):
                all_paths.append(p)
                all_audio.append(e)
        except Exception as e:
            print(f"  ⚠️  Atlandı ({Path(batch[0]).name}...): {type(e).__name__}: {e}")
            failed.extend(batch)
            continue
        done = i + len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(to_embed) - done) / rate if rate > 0 else 0
        print(f"  audio [{done}/{len(to_embed)}] {rate:.1f}/sn, ETA {eta:.0f}sn")
    print(f"  Audio embed süresi: {time.time()-t0:.1f}sn")

    # === Filename text embedding ===
    if use_filename:
        valid_pairs = [(p, c) for p, c in zip(to_embed, captions) if p not in failed]
        if valid_pairs:
            print(f"\n⏳ Filename text embedding ({len(valid_pairs)} caption)...")
            t1 = time.time()
            new_paths = [p for p, _ in valid_pairs]
            new_caps = [c for _, c in valid_pairs]
            text_embs = []
            for i in range(0, len(new_caps), args.text_batch_size):
                batch_caps = new_caps[i : i + args.text_batch_size]
                te = model.get_text_embedding(batch_caps, use_tensor=False)
                te = te / np.linalg.norm(te, axis=1, keepdims=True)
                text_embs.append(te)
            text_embs = np.concatenate(text_embs, axis=0)
            # path → text emb mapping yapıp all_paths sırasına göre yerleştir
            text_map = dict(zip(new_paths, text_embs))
            for p in all_paths[len(existing_audio):]:  # yeni eklenenler
                all_text.append(text_map.get(p))
            print(f"  Text embed süresi: {time.time()-t1:.1f}sn")

    if not all_audio:
        print("❌ Embedding üretilemedi.")
        return

    audio_arr = np.stack(all_audio).astype(np.float32)
    save_data = {
        "paths": np.array(all_paths),
        "embeddings": audio_arr,
        "preset": args.preset,
    }
    if use_filename and all_text and all(t is not None for t in all_text):
        save_data["text_embeddings"] = np.stack(all_text).astype(np.float32)

    np.savez(args.index_path, **save_data)
    size_mb = os.path.getsize(args.index_path) / 1024 / 1024
    print(f"\n✅ Index: {args.index_path}")
    print(f"   {len(all_paths)} dosya, audio={audio_arr.shape}, "
          f"text={'evet' if 'text_embeddings' in save_data else 'hayır'}, {size_mb:.1f}MB")
    print(f"   Toplam süre: {time.time()-t0:.1f}sn")
    if failed:
        print(f"   ⚠️  {len(failed)} dosya başarısız")


if __name__ == "__main__":
    main()