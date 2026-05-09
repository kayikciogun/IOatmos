#!/usr/bin/env python3
"""
clap_search.py — Index'lenmiş kütüphanede hibrit arama.

Audio + filename hibrit skor:
  combined = audio_w * audio_sim + text_w * text_sim - neg_alpha * neg_sim
  audio_w default = 0.6 (text_weight=0.4)

Kullanım:
    python clap_search.py --tags "rain on tin roof" "thunder rumble"
    python clap_search.py --tags "beach waves" --negative "underwater submerged"
    python clap_search.py --tags "forest birds" --text_weight 0.5 --show_breakdown
    python clap_search.py --tags "dog barking" --text_weight 0   # sadece audio
    python clap_search.py --tags "rain" --text_weight 1.0        # sadece filename
"""

import argparse
import os
import sys
import re
from pathlib import Path

import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

import laion_clap
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clap_index import PRESETS, download_ckpt_if_needed


def load_index(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index yok: {path}. Önce: python clap_index.py --audio_dir ...")
    data = np.load(path, allow_pickle=True)
    return {
        "paths":       data["paths"].tolist(),
        "audio":       data["embeddings"],
        "text":        data["text_embeddings"] if "text_embeddings" in data.files else None,
        "catids":      data["catids"].tolist() if "catids" in data.files else None,
        "cap_lengths": data["caption_lengths"].tolist() if "caption_lengths" in data.files else None,
        "preset":      str(data["preset"]) if "preset" in data.files else "natural",
    }


def build_model(preset: str, device: str):
    cfg = PRESETS[preset]
    model = laion_clap.CLAP_Module(
        enable_fusion=cfg["fusion"], amodel=cfg["amodel"], device=device
    )
    if cfg["model_id"] is not None:
        model.load_ckpt(model_id=cfg["model_id"])
    else:
        ckpt = download_ckpt_if_needed(cfg["ckpt_url"], cfg["ckpt_name"])
        model.load_ckpt(ckpt)
    return model, cfg


# UCS kategori → arama yapılacak catid'ler (Aşama 2)
UCS_SEARCH_SCOPE = {
    'WATR-SURF':         ['WATRSurf', 'WATRWave', 'AMBSea', 'UNKNOWN'],
    'WATR-WAVE':         ['WATRSurf', 'WATRWave', 'AMBSea', 'UNKNOWN'],
    'WATR-WATERFALL':    ['WATRFall', 'UNKNOWN'],
    'WATR-FLOW':         ['WATRFlow', 'UNKNOWN'],
    'WATR-UNDERWATER':   ['WATRUndwtr', 'UNKNOWN'],

    'AMB-URBAN':         ['AMBUrbn', 'AMBTraf', 'AMBTown', 'AMBTran', 'UNKNOWN'],
    'AMB-TRAFFIC':       ['AMBTraf', 'AMBUrbn', 'UNKNOWN'],
    'AMB-FOREST':        ['AMBForst', 'AMBRurl', 'UNKNOWN'],
    'AMB-RURAL':         ['AMBRurl', 'AMBForst', 'AMBAlpn', 'UNKNOWN'],
    'AMB-DESERT':        ['AMBDsrt', 'AMBRurl', 'UNKNOWN'],
    'AMB-SEASIDE':       ['AMBSea', 'WATRSurf', 'UNKNOWN'],
    'AMB-LAKESIDE':      ['AMBLake', 'AMBSea', 'UNKNOWN'],
    'AMB-TROPICAL':      ['AMBTrop', 'AMBForst', 'UNKNOWN'],

    'AMB-ROOM-TONE':     ['AMBRoom', 'AMBHome', 'UNKNOWN'],
    'AMB-RESIDENTIAL':   ['AMBHome', 'AMBRoom', 'UNKNOWN'],
    'AMB-OFFICE':        ['AMBOffc', 'AMBPubl', 'UNKNOWN'],
    'AMB-RESTAURANT':    ['AMBRest', 'AMBPubl', 'UNKNOWN'],
    'AMB-PUBLIC':        ['AMBPubl', 'AMBRest', 'AMBOffc', 'AMBMrkt', 'UNKNOWN'],
    'AMB-MARKET':        ['AMBMrkt', 'AMBPubl', 'UNKNOWN'],
    'AMB-TRANSPORT':     ['AMBTran', 'AMBUrbn', 'UNKNOWN'],
    'AMB-RELIGIOUS':     ['AMBRlgn', 'AMBPubl', 'UNKNOWN'],
    'AMB-UNDERGROUND':   ['AMBUndr', 'AMBTran', 'UNKNOWN'],
    'AMB-INDUSTRIAL':    ['AMBInd', 'AMBCnst', 'UNKNOWN'],
    'AMB-CONSTRUCTION':  ['AMBCnst', 'AMBInd', 'UNKNOWN'],
    'AMB-HITECH':        ['AMBTech', 'AMBOffc', 'UNKNOWN'],

    'VEH-INTERIOR':      ['VEHInt', 'UNKNOWN'],
    'VEH-CAR':           ['VEHCar', 'AMBTraf', 'UNKNOWN'],
    'CRWD-WALLA':        ['CRWDWalla', 'AMBPubl', 'UNKNOWN'],
}

def get_category_mask(ucs_category: str, catids: list) -> np.ndarray:
    """
    Verilen UCS kategorisi için hangi index satırlarının
    arama kapsamında olduğunu belirten boolean mask üret.
    """
    if catids is None:
        return None

    allowed = UCS_SEARCH_SCOPE.get(ucs_category, None)
    if allowed is None:
        return None

    mask = np.array([c in allowed for c in catids])
    
    # Güvenlik: eğer mask çok az dosya seçtiyse (< 5) filtre uygulama
    if mask.sum() < 5:
        return None

    return mask


def _compute_dynamic_weights(cap_lengths: list | None, n_files: int) -> np.ndarray:
    """
    Her dosya için text_weight hesapla.

    Thresholds:
      < 15 kelime  → 0.35  (short: sadece dosya adı, audio güvenilir)
      15-30 kelime → 0.45  (medium: kısa bext veya acoustic context ağırlıklı)
      >= 30 kelime → 0.55  (rich: zengin Boom/3DS bext, text güvenilir)
    """
    if cap_lengths is None:
        return np.full(n_files, 0.40)

    lengths = np.array(cap_lengths)
    weights = np.where(lengths >= 30, 0.55, np.where(lengths >= 15, 0.45, 0.35))
    return weights  # shape: [n_files]


def search(
    tags: list[str],
    idx: dict,
    model,
    text_weight: float = 0.4,
    neg_alpha: float = 0.0,
    negative: list[str] | None = None,
    top_k: int = 3,
    min_score: float | None = None,
    ucs_categories: list[str] | None = None,
    dynamic_weight: bool = True,
) -> list[dict]:
    """
    Hibrit arama fonksiyonu.
    ucs_categories: tags ile aynı uzunlukta, her tag için UCS kategori kısıtı.
    """
    has_text = idx["text"] is not None
    has_catids = idx.get("catids") is not None
    n_files = idx["audio"].shape[0]

    # Pozitif query embedding
    pos_emb = model.get_text_embedding(tags, use_tensor=False)
    pos_emb = pos_emb / np.linalg.norm(pos_emb, axis=1, keepdims=True)

    # Audio ve text similarity matrisleri
    sim_audio = pos_emb @ idx["audio"].T
    sim_text = (pos_emb @ idx["text"].T) if has_text and text_weight > 0 else None

    # ── Dinamik weight hesabı ──────────────────────────────────────────────────
    if dynamic_weight and has_text and idx.get("cap_lengths") is not None:
        text_w = _compute_dynamic_weights(idx["cap_lengths"], n_files)  # [n_files]
        audio_w = 1.0 - text_w
        pos_combined = audio_w * sim_audio + text_w * sim_text
    elif has_text and text_weight > 0:
        audio_w = 1.0 - text_weight
        pos_combined = audio_w * sim_audio + text_weight * sim_text
    else:
        pos_combined = sim_audio
    # ──────────────────────────────────────────────────────────────────────────

    # Negatif query embedding
    sim_neg = None
    if negative and neg_alpha > 0:
        neg_emb = model.get_text_embedding(negative, use_tensor=False)
        neg_emb = neg_emb / np.linalg.norm(neg_emb, axis=1, keepdims=True)
        neg_audio = neg_emb @ idx["audio"].T
        neg_text = (neg_emb @ idx["text"].T) if has_text and text_weight > 0 else None

        # ── Negatif: dinamik weight (pozitif ile aynı) ─────────────────────────
        if dynamic_weight and has_text and idx.get("cap_lengths") is not None:
            text_w = _compute_dynamic_weights(idx["cap_lengths"], n_files)
            audio_w = 1.0 - text_w
            neg_scores = audio_w * neg_audio + text_w * neg_text
        elif has_text and text_weight > 0:
            audio_w = 1.0 - text_weight
            neg_scores = audio_w * neg_audio + text_weight * neg_text
        else:
            neg_scores = neg_audio
        # ────────────────────────────────────────────────────────────────────────

        sim_neg = neg_scores.max(axis=0) if neg_scores.ndim > 1 else neg_scores

    output = []
    for i, tag in enumerate(tags):
        final_scores = pos_combined[i].copy()

        if sim_neg is not None:
            final_scores = final_scores - neg_alpha * sim_neg

        # UCS Kategori Filtresi (Aşama 2)
        mask = None
        if ucs_categories and i < len(ucs_categories) and has_catids:
            mask = get_category_mask(ucs_categories[i], idx["catids"])

        if mask is not None:
            masked_scores = final_scores.copy()
            masked_scores[~mask] = -np.inf
        else:
            masked_scores = final_scores

        order = np.argsort(-masked_scores)
        results = []
        for j in order:
            if masked_scores[j] == -np.inf:
                break
            if min_score is not None and final_scores[j] < min_score:
                break
            if len(results) >= top_k:
                break
            cap_len = idx["cap_lengths"][j] if idx.get("cap_lengths") is not None else -1
            tw_used = float(
                _compute_dynamic_weights(
                    [idx["cap_lengths"][j]] if idx.get("cap_lengths") is not None else None, 1
                )[0]
            )
            results.append({
                "path": idx["paths"][j],
                "score": float(final_scores[j]),
                "audio_sim": float(sim_audio[i, j]),
                "text_sim": float(sim_text[i, j]) if sim_text is not None else 0.0,
                "neg_sim": float(sim_neg[j]) if sim_neg is not None else 0.0,
                "catid": idx["catids"][j] if has_catids else "UNKNOWN",
                "cap_length": cap_len,
                "text_weight_used": tw_used,
            })
        output.append({"tag": tag, "results": results})

    return output


# Su kategorilerinde audio yanıltıcı — minimum text_weight garantisi
CATEGORY_MIN_TEXT_WEIGHT = {
    "WATR-WATERFALL": 0.50,
    "WATR-SURF": 0.50,
    "WATR-FLOW": 0.48,
    "WATR-WAVE": 0.48,
}


def smart_search(query, ucs_category, idx, model, top_k=3):
    """
    B+C Birleşik Arama Mimarisi (Soft Filter + Dynamic Weight Versiyonu):
    1. Tüm kütüphanede ara (Global Search) — dinamik weight aktif
    2. Su kategorilerinde düşük weight'li dosyaları re-score et
    3. Rerank aşamasında kategori bilgisini 'soft hint' olarak kullan
    """
    # === AŞAMA 1: Global Search (dinamik weight açık) ===
    all_results = search(
        tags=[query],
        idx=idx,
        model=model,
        dynamic_weight=True,
        top_k=50,
        ucs_categories=None,  # Hard filter KALDIRILDI
    )

    candidates = all_results[0]["results"]

    # === AŞAMA 1.5: Su kategorilerinde minimum text_weight guarantee ===
    min_tw = CATEGORY_MIN_TEXT_WEIGHT.get(ucs_category)
    if min_tw:
        recalculated = []
        for r in candidates:
            tw_used = r.get("text_weight_used", 0.4)
            if tw_used < min_tw:
                # Bu dosya dinamik weight'te düşük almış ama kategori için riskli
                # Min text_weight ile yeniden hesapla
                new_score = (1 - min_tw) * r["audio_sim"] + min_tw * r["text_sim"]
                recalculated.append({**r, "score": new_score, "text_weight_used": min_tw})
            else:
                recalculated.append(r)
        candidates = sorted(recalculated, key=lambda x: -x["score"])

    # === AŞAMA 2: Re-ranker (Soft Filter) ===
    reranked = rerank(query, ucs_category, candidates)

    final_best_score = reranked[0]["score"] if reranked else 0

    return {
        "results": reranked[:top_k],
        "source": "global",
        "confidence": "high"
        if final_best_score > 0.45
        else "medium"
        if final_best_score > 0.30
        else "low",
    }


def rerank(query: str, ucs_category: str, candidates: list) -> list:
    """
    CLAP score'u 2 sinyal ile düzelt:
    1. UCS catid match bonus (+0.04)
    2. Query keyword overlap bonus (+0.02)
    """
    # Query'den keyword'ler çıkar
    query_words = set(query.lower().split())
    
    # UCS scope — bu kategori için hangi catid'ler bekleniyor?
    expected_catids = set(UCS_SEARCH_SCOPE.get(ucs_category, []))
    
    scored = []
    for r in candidates:
        score = r["score"]
        catid = r.get("catid", "UNKNOWN")
        
        # 1. UCS catid match bonus (+0.04)
        if catid in expected_catids and catid != 'UNKNOWN':
            score += 0.04
        
        # 2. Keyword overlap bonus (+0.02 per word)
        path_words = set(
            re.sub(r'[-_.,;()]', ' ', 
            Path(r["path"]).stem).lower().split()
        )
        overlap = len(query_words & path_words)
        score += 0.02 * overlap
        
        scored.append({**r, "score": score, "original_score": r["score"]})
    
    return sorted(scored, key=lambda x: -x["score"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", default="./index.npz")
    parser.add_argument("--tags", nargs="+", required=False)
    parser.add_argument("--tags_file")
    parser.add_argument("--negative", nargs="+", default=None,
                        help="Negatif query'ler (penalize edilecek ses karakteri)")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--min_score", type=float, default=None)
    parser.add_argument("--full_path", action="store_true")
    parser.add_argument("--text_weight", type=float, default=0.4,
                        help="Filename embedding ağırlığı (0=sadece audio, 1=sadece filename)")
    parser.add_argument("--neg_alpha", type=float, default=0.0,
                        help="Negatif cezalandırma şiddeti (default 0.0, sub-indexing devrede)")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="auto")
    parser.add_argument("--show_breakdown", action="store_true",
                        help="Her sonuç için audio/text/neg skorlarını ayrı göster")
    parser.add_argument("--ucs_category", type=str, default=None,
                        help="UCS Kategori kısıtlaması (örn: WATR-SURF)")
    parser.add_argument("--dynamic_weight", action="store_true", default=True,
                        help="Dosya başına caption kalitesine göre text_weight ayarla")
    parser.add_argument("--no_dynamic_weight", dest="dynamic_weight", action="store_false",
                        help="Dinamik weight kapat, sabit text_weight kullan")
    args = parser.parse_args()

    if not args.tags and not args.tags_file:
        parser.error("--tags veya --tags_file gerekli")
    tags = args.tags or [l.strip() for l in open(args.tags_file, encoding="utf-8") if l.strip()]

    idx = load_index(args.index_path)
    has_text = idx["text"] is not None

    print(f"📦 Index: {len(idx['paths'])} dosya, preset={idx['preset']}")
    print(f"   audio={idx['audio'].shape}, filename_emb={'evet' if has_text else 'hayır'}")

    text_w = args.text_weight
    if not has_text and text_w > 0:
        print("⚠️  Index'te filename embedding yok, text_weight=0 yapılıyor")
        text_w = 0.0
    audio_w = 1.0 - text_w
    print(f"🏷️  {len(tags)} etiket | dynamic_weight={args.dynamic_weight} | neg_alpha={args.neg_alpha}")
    if not args.dynamic_weight:
        print(f"   (sabit weight: audio={1-args.text_weight:.2f} text={args.text_weight:.2f})")
    if args.negative:
        print(f"❌ Negative: {args.negative}\n")
    else:
        print()

    # Device
    if args.device != "auto":
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"⏳ Model yükleniyor ({device})...")
    model, _ = build_model(idx["preset"], device)
    print("✅ Hazır\n")

    results = []
    for tag in tags:
        res = smart_search(
            query=tag,
            ucs_category=args.ucs_category,
            idx=idx,
            model=model,
            top_k=args.top_k
        )
        results.append({"tag": tag, **res})

    print("=" * 78)
    for item in results:
        print(f"\n🏷️  '{item['tag']}' [Source: {item['source']}, Conf: {item['confidence']}]")
        print("-" * 78)
        if not item["results"]:
            print(f"  (sonuç bulunamadı)")
            continue
        for rank, r in enumerate(item["results"], 1):
            disp = r["path"] if args.full_path else Path(r["path"]).name
            if args.show_breakdown:
                neg_str = f" n={r['neg_sim']:+.3f}" if args.negative else ""
                print(f"  {rank}. {r['score']:+.4f} [Orig: {r['original_score']:+.3f}] "
                      f"(a={r['audio_sim']:+.3f} t={r['text_sim']:+.3f}{neg_str})  {disp}")
            else:
                print(f"  {rank}. {r['score']:+.4f}  {disp}")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()