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
import struct  
import re
import subprocess
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
import soundfile as sf
import librosa
from torch.utils.data import Dataset, DataLoader
from caption_enrichment import enrich_caption


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


def get_audio_duration(file_path):
    """Ses dosyasının süresini saniye cinsinden döndürür. Önce soundfile, sonra FFprobe kullanır."""
    try:
        info = sf.info(file_path)
        return info.frames / info.samplerate
    except Exception:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", file_path],
                capture_output=True, text=True, timeout=5
            )
            return float(result.stdout.strip())
        except Exception:
            return None


# UCS_HINTS kaldırıldı, caption_enrichment.py tarafından yönetiliyor.

# UCS kategori başına bir "anchor text" tanımla (Zero-shot classification için)
UCS_ANCHORS = {
    # Su / Deniz
    'WATRSurf':   'ocean waves crashing on shore above water surface sea waves surf roar',
    'WATRUndwtr': 'underwater recording submerged hydrophone deep ocean aquatic bubbles swimming',
    'WATRFall':   'waterfall cascading water falls splashing heavy water',
    'WATRFlow':   'river stream creek flowing water bubbling brook',
    'WATRWave':   'individual ocean waves splashing shore water',
    'AMBSea':     'beach seaside coastal ocean shore people seagulls seaside ambience',
    'AMBLake':    'lake water gentle lapping shore birds quiet',
    'AMBNaut':    'ship boat nautical sea docks gulls waves harbour',
    
    # Şehir / Dış Mekan
    'AMBUrbn':    'city urban street traffic pedestrians trams buses horns sirens busy downtown',
    'AMBSubn':    'suburban residential neighborhood lawnmower birds children playing quiet',
    'AMBTraf':    'highway traffic cars engines road busy motorway',
    'AMBTown':    'small town village light traffic church bell activity',
    'AMBPark':    'city park wind trees children playing fountain outdoor',
    
    # Doğa
    'AMBForst':   'forest woodland birds wind trees leaves rustling wilderness',
    'AMBDsrt':    'desert wind sand dunes arid landscape dry wind hiss',
    'AMBRurl':    'rural countryside prairie meadow nature farm',
    'AMBTrop':    'jungle rainforest tropical insects birds exotic nature',
    'AMBFarm':    'farm animals tractor rooster nature countryside',
    'ANIMBird':   'birds chirping singing forest nature wings wildlife',
    'ANIMInsct':  'insects crickets grasshoppers buzzing summer',
    
    # İç Mekan
    'AMBRoom':    'empty room tone quiet indoor air hum silence ventilation',
    'AMBHome':    'residential apartment home interior ambience daily life',
    'AMBOffc':    'office workplace typing voices phones printer activity',
    'AMBRest':    'restaurant bar cafe dining crowd dishes glasses chatter',
    'AMBPubl':    'airport mall lobby museum public indoor crowd shopping center supermarket',
    'AMBMrkt':    'outdoor market vendors crowd busy street market',
    'AMBTran':    'train station subway airport terminal transportation hub',
    'AMBSchl':    'school classroom hallway children talking bells',
    'AMBHosp':    'hospital medical monitor beeps quiet voices surgery clinic',
    'AMBRlgn':    'monks chanting temple religious scriptures church prayer',
    
    # Endüstriyel / Teknik
    'AMBUndr':    'underground car park parking garage basement enclosed reverb tunnel cave',
    'AMBInd':     'industrial factory warehouse machinery plant metal workshop',
    'AMBCnst':    'construction site jackhammer crane machinery outdoor concrete mixer',
    'AMBTech':    'server room data center fans hum computers electronics',
    'VEHInt':     'car interior cabin engine hum road noise driving',
    'VEHCar':     'car engine driving traffic street vehicle auto',
    'CRWDWalla':  'crowd murmur walla people talking indistinct background chatter',
    
    # Foley / Nesne (Yeni)
    'TOOLWork':   'hand tools impact manual work toolbox metal wood scraping',
    'WOODChopp':  'chopping wood axe hitting tree timber forest wood work',
    'METLWork':   'metal workshop clanging grinding tools factory workshop',
}

# Manuel override listesi (Path bazlı kesin eşleşmeler)
MANUAL_OVERRIDES = {
    'underground car park': 'AMBUndr',
    'underground garage':   'AMBUndr',
    'underground lab':      'AMBUndr',
    'car park':             'AMBUndr',
    'parking garage':       'AMBUndr',
}

def apply_manual_overrides(filepath: str) -> str | None:
    """Dosya yoluna göre manuel kategori ataması."""
    path_lower = filepath.lower()
    for pattern, catid in MANUAL_OVERRIDES.items():
        if pattern in path_lower:
            return catid
    return None

def _keyword_detect(filepath: str, bext_description: str = "") -> str:
    """Eski keyword tabanlı tespit (CLAP yoksa veya confidence düşükse fallback)."""
    search_text = (bext_description + " " + filepath).lower()
    
    # Not: Bu liste artık sadece fallback için. Ana tespit CLAP zero-shot ile yapılır.
    FALLBACK_PATTERNS = {
        'WATRSurf':  ['waves coming in', 'ocean waves', 'surf'],
        'WATRUndwtr':['underwater', 'uwt', 'submerged', 'hydrophone'],
        'WATRFall':  ['waterfall', 'falls'],
        'WATRFlow':  ['river', 'creek', 'stream'],
        'AMBUrbn':   ['city', 'urban', 'downtown', 'tram', 'metro'],
        'AMBTraf':   ['traffic', 'highway', 'motorway'],
        'AMBForst':  ['forest', 'woodland', 'woods'],
        'AMBSea':    ['beach', 'seaside', 'coastal'],
        'AMBRurl':   ['rural', 'countryside', 'village'],
        'CRWDWalla': ['crowd', 'walla', 'murmur'],
    }
    
    scores = {}
    for catid, patterns in FALLBACK_PATTERNS.items():
        score = sum(1 for p in patterns if p.lower() in search_text)
        if score > 0:
            scores[catid] = score
    
    return max(scores, key=scores.get) if scores else 'UNKNOWN'

def precompute_anchors(model) -> dict:
    """Tüm UCS anchor text'lerini encode et — tek seferlik."""
    import numpy as np
    texts = list(UCS_ANCHORS.values())
    catids = list(UCS_ANCHORS.keys())
    
    embs = model.get_text_embedding(texts, use_tensor=False)
    # L2 normalize
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    
    return {catid: emb for catid, emb in zip(catids, embs)}

def classify_with_clap(text: str, model, anchor_embeddings: dict) -> str:
    """CLAP text encoder ile zero-shot classification (Single-file)."""
    import numpy as np
    if not text.strip(): return 'UNKNOWN'
    
    # Text embedding
    file_emb = model.get_text_embedding([text], use_tensor=False)
    file_emb = file_emb / np.linalg.norm(file_emb)
    
    # Cosine similarity
    scores = {}
    for catid, anchor_emb in anchor_embeddings.items():
        scores[catid] = float((file_emb @ anchor_emb).item())
    
    best = max(scores, key=scores.get)
    best_score = scores[best]
    
    if best_score < 0.25:
        return 'UNKNOWN'
    
    return best

def classify_batch(texts: list[str], model, anchor_embeddings: dict) -> list[str]:
    """Tüm batch metinlerini tek seferde encode eder (Hızlı)."""
    import numpy as np
    if not texts: return []
    
    # 1. Text embedding
    embs = model.get_text_embedding(texts, use_tensor=False)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    
    # 2. Anchor matrix hazırlığı
    anchor_matrix = np.stack(list(anchor_embeddings.values())) # [n_anchors, 512]
    catids = list(anchor_embeddings.keys())
    
    # 3. Matris çarpımı ile cosine similarity (Batch Dot Product)
    scores_matrix = embs @ anchor_matrix.T # [n_files, n_anchors]
    
    best_indices = scores_matrix.argmax(axis=1)
    best_scores = scores_matrix.max(axis=1)
    
    results = []
    for i, idx in enumerate(best_indices):
        if best_scores[i] >= 0.25:
            results.append(catids[idx])
        else:
            # Anchor confidence düşükse keyword fallback dene (opsiyonel)
            results.append('UNKNOWN')
            
    return results

def get_ucs_catid_from_metadata(filepath: str) -> str | None:
    """iXML veya bext'ten UCS CatID oku."""
    try:
        with open(filepath, 'rb') as f:
            if f.read(4) != b'RIFF': return None
            f.read(4)
            if f.read(4) != b'WAVE': return None
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4: break
                chunk_size = struct.unpack('<I', f.read(4))[0]
                if chunk_id == b'iXML':
                    chunk_data = f.read(chunk_size)
                    xml = chunk_data.decode('utf-8', errors='ignore')
                    # UCS standard field
                    m = re.search(r'<IXML:CATEGORY>(.*?)</IXML:CATEGORY>', xml)
                    if m: return m.group(1).strip()
                    # Alternatif field adları
                    for tag in ['UCS_CATID', 'CATEGORY', 'SUBCATEGORY']:
                        m = re.search(f'<{tag}>(.*?)</{tag}>', xml)
                        if m and m.group(1).strip():
                            return m.group(1).strip()
                else:
                    # Chunk'ı atla (chunk_size + padding)
                    f.seek(chunk_size + (chunk_size % 2), 1)
    except Exception:
        pass
    return None

def detect_ucs_catid(filepath: str, bext_description: str = "", model=None, anchor_embeddings=None) -> str:
    """
    Library-agnostic UCS sınıflandırma (Single-file version).
    """
    # Öncelik 0: Manuel override (her zaman kazanır)
    override = apply_manual_overrides(filepath)
    if override:
        return override

    # Öncelik 1: iXML metadata
    ixml_catid = get_ucs_catid_from_metadata(filepath)
    if ixml_catid:
        return ixml_catid

    # Öncelik 2: CLAP zero-shot
    if model and anchor_embeddings:
        text = bext_description if bext_description else filename_to_caption(filepath)
        if text.strip():
            res = classify_with_clap(text, model, anchor_embeddings)
            if res != 'UNKNOWN':
                return res
    
    # Öncelik 3: Keyword fallback
    return _keyword_detect(filepath, bext_description)

def read_bext_description(filepath: str) -> str:
    """WAV dosyasından bext description chunk'ını oku."""
    try:
        with open(filepath, 'rb') as f:
            if f.read(4) != b'RIFF': return ""
            f.read(4)
            if f.read(4) != b'WAVE': return ""
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4: break
                chunk_size = struct.unpack('<I', f.read(4))[0]
                if chunk_id == b'bext':
                    # bext genelde 602 byte+ metadata içerir, description ilk 256 byte'tır
                    desc = f.read(min(256, chunk_size))
                    return desc.rstrip(b'\x00').decode('latin-1', errors='ignore').strip()
                # Chunk'ı atla (chunk_size + padding)
                f.seek(chunk_size + (chunk_size % 2), 1)
    except Exception:
        pass
    return ""

def filename_to_caption(path: str, catid: str = "") -> str:
    """
    Zenginleştirilmiş caption üretir. 
    Acoustic context ve perspective prefix'leri ekler.
    """
    bext = read_bext_description(path) if path.lower().endswith('.wav') else ""
    return enrich_caption(path, bext, catid)


def download_ckpt_if_needed(url: str, name: str) -> str:
    local_dir = Path(__file__).parent.parent / "models"
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
        return {}, None, None, None, None, {}
    data = np.load(path, allow_pickle=True)
    paths = data["paths"].tolist()
    audio_emb = data["embeddings"]
    text_emb = data["text_embeddings"] if "text_embeddings" in data.files else None
    catids = data["catids"].tolist() if "catids" in data.files else None
    preset = str(data["preset"]) if "preset" in data.files else None
    cap_lengths = data["caption_lengths"].tolist() if "caption_lengths" in data.files else None

    audio_dict = dict(zip(paths, audio_emb))
    text_dict = dict(zip(paths, text_emb)) if text_emb is not None else {}
    catid_dict = dict(zip(paths, catids)) if catids is not None else {}
    caplen_dict = dict(zip(paths, cap_lengths)) if cap_lengths is not None else {}

    return audio_dict, text_dict, catid_dict, preset, data, caplen_dict


def build_model(preset_name: str, device_override: str = None):
    cfg = PRESETS[preset_name]
    if device_override and device_override != "auto":
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


class AudioDataset(Dataset):
    def __init__(self, file_paths, max_duration=120.0):
        self.file_paths = file_paths
        self.max_duration = max_duration

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            # laion_clap expects 48kHz mono. To prevent extreme loading times
            # and OOMs, cap at max_duration.
            w, _ = librosa.load(path, sr=48000, duration=self.max_duration)
            return path, w
        except Exception:
            return path, None

def collate_fn(batch):
    paths = [item[0] for item in batch]
    waveforms = [item[1] for item in batch]
    return paths, waveforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--index_path", default="./index.npz")
    parser.add_argument("--batch_size", type=int, default=4)
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
    parser.add_argument(
        "--min_duration",
        type=float,
        default=10.0,
        help="Minimum ses süresi (saniye). Bu süreden kısa sesler atlanır.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Ses okuma paralelleştirme işlemci sayısı",
    )
    args = parser.parse_args()

    audio_files = find_audio_files(args.audio_dir)
    if not audio_files:
        print(f"❌ {args.audio_dir} içinde ses dosyası yok.")
        return
    
    print(f"📁 Bulunan dosya: {len(audio_files)}\n")

    existing_audio, existing_text, existing_catids, prev_preset, _, existing_cap_lengths = {}, {}, {}, None, None, {}
    if args.update and not args.force:
        existing_audio, existing_text, existing_catids, prev_preset, _, existing_cap_lengths = load_existing_index(args.index_path)
        if existing_audio and prev_preset and prev_preset != args.preset:
            print(f"⚠️  Var olan index farklı preset ({prev_preset}). --force kullan.")
            return
        if existing_audio:
            print(f"♻️  Var olan: {len(existing_audio)} dosya")
            
        audio_files = [f for f in audio_files if f not in existing_audio]
        if not audio_files:
            print("✅ Yeni dosya yok, yapılacak iş yok.")
            return
        print(f"🆕 Süre kontrolü yapılacak yeni dosya: {len(audio_files)}")

    # === Filtreleme (Minimum Süre) ===
    if args.min_duration > 0:
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor
        print(f"🔍 {args.min_duration} saniyeden kısa sesler ayıklanıyor...")
        
        filtered = []
        skipped_count = 0
        
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(tqdm(
                ex.map(lambda f: (f, get_audio_duration(f)), audio_files),
                total=len(audio_files), desc="Süre kontrolü"
            ))
            
        for f, d in results:
            if d is not None and d >= args.min_duration:
                filtered.append(f)
            else:
                skipped_count += 1
                
        audio_files = filtered
        if skipped_count > 0:
            print(f"   ⚠️  {skipped_count} dosya {args.min_duration}sn'den kısa olduğu için atlandı.")

    to_embed = audio_files
    if not to_embed:
        print("❌ Filtreleme sonrası işlenecek dosya kalmadı.")
        return
    print(f"⏳ Embed edilecek: {len(to_embed)} dosya\n")

    device_override = None if args.device == "auto" else args.device
    model = build_model(args.preset, device_override=device_override)

    use_filename = not args.no_filename
    if use_filename:
        # Önizleme için catid'siz caption'lar (main loop'ta catid ile zenginleşecek)
        preview_captions = [filename_to_caption(f) for f in to_embed[:3]]
        # Örnek 3 caption göster
        print(f"📝 Filename → caption örnekleri:")
        for i in range(min(3, len(to_embed))):
            print(f"   {Path(to_embed[i]).name}")
            print(f"   → '{preview_captions[i]}'")
        print()

    all_paths = list(existing_audio.keys())
    all_audio = [existing_audio[p] for p in all_paths]
    all_text = [existing_text.get(p) for p in all_paths] if use_filename else []
    all_catids = [existing_catids.get(p, 'UNKNOWN') for p in all_paths]
    all_cap_lengths = [existing_cap_lengths.get(p, -1) for p in all_paths]

    # Duration array — layer-aware search için (mevcut index'ten koru)
    existing_durations = []
    if args.update and os.path.exists(args.index_path):
        try:
            with np.load(args.index_path, allow_pickle=True) as data:
                if "durations" in data.files:
                    existing_durations = data["durations"].tolist()
                else:
                    # Durations yoksa, mevcut dosyalar kadar -1.0 (bilinmiyor) ekle
                    existing_durations = [-1.0] * len(existing_audio)
        except Exception:
            existing_durations = [-1.0] * len(existing_audio)
    else:
        existing_durations = []

    all_durations = list(existing_durations)


    # === Audio embedding ===
    t0 = time.time()
    failed = []
    
    # CLAP yüklendiyse UCS sınıflandırma için anchor'ları hesapla
    anchor_embeddings = precompute_anchors(model)
    
    dataset = AudioDataset(to_embed)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=collate_fn, 
        shuffle=False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    from tqdm import tqdm
    pbar = tqdm(
        total=len(to_embed), 
        desc="Audio Embed", 
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]", 
        unit="dosya"
    )

    for paths, waveforms in loader:
        valid_paths = []
        valid_waves = []
        for p, w in zip(paths, waveforms):
            if w is not None:
                valid_paths.append(p)
                valid_waves.append(w)
            else:
                failed.append(p)
                
        if not valid_paths:
            pbar.update(len(paths))
            continue
            
        try:
            # 1. Audio embedding
            emb = model.get_audio_embedding_from_data(x=valid_waves, use_tensor=False)
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            
            # 2. Batch UCS Classification (Çalıştırma)
            # Önce metadata kontrolü, yoksa CLAP
            final_batch_catids = []
            
            # CLAP tahmini gerekenler için batch hazırla
            clap_indices = []
            clap_texts = []
            
            for idx, p in enumerate(valid_paths):
                # 0. Manuel override (highest priority)
                override = apply_manual_overrides(p)
                if override:
                    final_batch_catids.append(override)
                else:
                    # 1. iXML/bext metadata önceliği
                    cid = get_ucs_catid_from_metadata(p)
                    if cid:
                        final_batch_catids.append(cid)
                    else:
                        # 2. CLAP için sıraya al
                        final_batch_catids.append(None) # placeholder
                        clap_indices.append(idx)
                        bext = read_bext_description(p) if p.endswith('.wav') else ""
                        clap_texts.append(bext if bext else filename_to_caption(p))
            
            # CLAP tahmini gerekiyorsa çalıştır
            if clap_indices:
                clap_results = classify_batch(clap_texts, model, anchor_embeddings)
                for i, res in zip(clap_indices, clap_results):
                    final_batch_catids[i] = res
            
            # 3. Sonuçları kaydet
            for p, e, cid in zip(valid_paths, emb, final_batch_catids):
                all_paths.append(p)
                all_audio.append(e)
                
                # Eğer CLAP UNKNOWN verdiyse keyword fallback dene
                if cid == 'UNKNOWN' or cid is None:
                    bext = read_bext_description(p) if p.endswith('.wav') else ""
                    cid = _keyword_detect(p, bext)
                
                all_catids.append(cid)
                
                # Duration bilgisini ekle
                dur = get_audio_duration(p)
                all_durations.append(float(dur) if dur is not None else -1.0)
        except Exception as e:
            tqdm.write(f"  ⚠️  Atlandı ({Path(valid_paths[0]).name}...): {type(e).__name__}: {e}")
            failed.extend(valid_paths)
            
        pbar.update(len(paths))
    
    pbar.close()
        
    print(f"  Audio embed süresi: {time.time()-t0:.1f}sn")

    # === Filename text embedding ===
    if use_filename:
        # Başarılı dosyaları ve catid'lerini bul
        success_paths = [p for p in to_embed if p not in failed]
        if success_paths:
            print(f"\n⏳ Filename text embedding ({len(success_paths)} enriched caption)...")
            t1 = time.time()
            
            # Catid-aware captions üret (all_catids listesinden en son eklenenleri al)
            new_files_start_idx = len(existing_audio)
            new_paths = all_paths[new_files_start_idx:]
            new_catids = all_catids[new_files_start_idx:]
            
            new_caps = [filename_to_caption(p, catid=c) for p, c in zip(new_paths, new_catids)]

            # Caption lengths hesapla
            new_cap_lengths = [len(cap.split()) for cap in new_caps]
            all_cap_lengths.extend(new_cap_lengths)

            # Örnek 2 zenginleştirilmiş caption göster
            print(f"📝 Enriched caption örnekleri:")
            for i in range(min(2, len(new_caps))):
                print(f"   → '{new_caps[i]}'")
            
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

    # Duration array'i hizala (bazı dosyalar failed olabilir, all_paths ile eşit uzunluk)
    while len(all_durations) < len(all_paths):
        all_durations.append(-1.0)
    all_durations = all_durations[:len(all_paths)]

    # Caption lengths hizala
    while len(all_cap_lengths) < len(all_paths):
        all_cap_lengths.append(-1)
    all_cap_lengths = all_cap_lengths[:len(all_paths)]

    save_data = {
        "paths": np.array(all_paths),
        "embeddings": audio_arr,
        "catids": np.array(all_catids),
        "preset": args.preset,
        "durations": np.array(all_durations, dtype=np.float32),
        "caption_lengths": np.array(all_cap_lengths, dtype=np.int16),
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