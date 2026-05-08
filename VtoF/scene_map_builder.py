"""
Modül 4: Scene Map Builder
scenes.json + (OpenRouter) ai_descriptions.json → scene_map.json.
Metin açıklaması yalnızca openrouter_analysis alanından okunur.
"""

import json
import re
import unicodedata
from pathlib import Path


def duration_flexibility_for(duration: float) -> str:
    """Ses seçimi sıkılığı: sahne süresine göre."""
    try:
        d = float(duration)
    except (TypeError, ValueError):
        d = 0.0
    if d < 1.0:
        return "exact"
    if d < 5.0:
        return "strict"
    if d < 15.0:
        return "loose"
    return "any"


def resolve_sound_description(analysis: dict) -> str:
    """sound_description boşsa sound_layers veya katman açıklamalarından türet."""
    desc = (analysis.get("sound_description") or "").strip()
    if desc:
        return desc
    layers = analysis.get("sound_layers") or []
    if not isinstance(layers, list) or not layers:
        return ""
    scored = []
    for L in layers:
        if not isinstance(L, dict):
            continue
        try:
            w = float(L.get("weight", 0) or 0)
        except (TypeError, ValueError):
            w = 0.0
        t = str(L.get("description", "")).strip()
        if t:
            scored.append((w, t))
    scored.sort(key=lambda x: -x[0])
    parts = [t for _, t in scored[:3]]
    return " ".join(parts)


def _normalize_analysis_fields(analysis: dict) -> dict:
    """openrouter_analysis için güvenli varsayılanlar (eski JSON uyumu)."""
    base = {
        "sound_description": "",
        "tags": [],
        "mood": "",
        "environment": "",
        "time_of_day": "",
        "intensity": 0.5,
        "negative_elements": [],
        "primary_focus": "",
        "sound_layers": [],
    }
    if not isinstance(analysis, dict):
        return dict(base)
    out = dict(base)
    out["sound_description"] = str(analysis.get("sound_description", "")).strip()
    tags = analysis.get("tags")
    out["tags"] = list(tags) if isinstance(tags, list) else []
    out["mood"] = str(analysis.get("mood", "")).strip()
    out["environment"] = str(analysis.get("environment", "")).strip()
    out["time_of_day"] = str(analysis.get("time_of_day", "")).strip()
    try:
        out["intensity"] = max(0.0, min(1.0, float(analysis.get("intensity", 0.5))))
    except (TypeError, ValueError):
        out["intensity"] = 0.5
    ne = analysis.get("negative_elements")
    out["negative_elements"] = (
        [str(x).strip() for x in ne if x] if isinstance(ne, list) else []
    )
    out["primary_focus"] = str(analysis.get("primary_focus", "")).strip()
    sl = analysis.get("sound_layers")
    out_layers = []
    if isinstance(sl, list):
        for item in sl:
            if not isinstance(item, dict):
                continue
            try:
                w = float(item.get("weight", 1.0))
            except (TypeError, ValueError):
                w = 1.0
            out_layers.append(
                {
                    "type": str(item.get("type", "ambience")).strip() or "ambience",
                    "description": str(item.get("description", "")).strip(),
                    "weight": max(0.0, min(1.0, w)),
                }
            )
    out["sound_layers"] = out_layers
    return out


def build_scene_map_from_frames(
    scenes_path="data/scenes.json",
    frame_records=None,
    output_path="data/scene_map.json",
):
    """
    OpenRouter kapalıyken: sahne JSON + kare yolları; sound_description/tags boş.
    frame_records: [{"scene_id", "frame_path", ...}] veya None.
    """
    print(f"🗺️ Sahne haritası (kare yolları, açıklama yok) oluşturuluyor...")

    with open(scenes_path, encoding="utf-8") as f:
        scenes = json.load(f)

    by_id = {}
    if frame_records:
        for r in frame_records:
            by_id[r["scene_id"]] = r

    scene_map = []
    for scene in scenes:
        sid = scene["scene_id"]
        fr = by_id.get(sid, {})
        blank_info = scene.get("blank_info") or {}
        is_blank = bool(blank_info.get("is_blank", False))
        dur = scene["duration"]

        scene_map.append(
            {
                "scene_id": sid,
                "start": scene["start"],
                "end": scene["end"],
                "start_frame": scene.get("start_frame"),
                "end_frame": scene.get("end_frame"),
                "duration": dur,
                "frame_path": fr.get("frame_path"),
                "sound_description": "",
                "tags": [],
                "mood": "",
                "environment": "",
                "time_of_day": "",
                "intensity": 0.5,
                "negative_elements": [],
                "primary_focus": "",
                "sound_layers": [],
                "target_duration": dur,
                "duration_flexibility": duration_flexibility_for(dur),
                "loop_acceptable": dur > 5.0,
                "is_blank": is_blank,
                "blank_info": blank_info if blank_info else None,
            }
        )
        print(f"   Sahne {sid}: frame_path={'evet' if fr.get('frame_path') else 'hayır'}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scene_map, f, indent=2, ensure_ascii=False)

    print(f"✅ Sahne haritası hazır → {output_path}")
    return scene_map


def build_scene_map(
    scenes_path="data/scenes.json",
    descriptions_path="data/ai_descriptions.json",
    output_path="data/scene_map.json",
    frame_records=None,
):
    """
    Sahne bilgileri ile AI açıklamalarını birleştirir.

    Args:
        scenes_path: scenes.json dosya yolu
        descriptions_path: ai_descriptions.json dosya yolu
        output_path: Çıktı scene_map.json dosyası
        frame_records: Opsiyonel kare kayıtları (frame_path; isteğe bağlı)

    Returns:
        Birleştirilmiş scene map listesi
    """
    print(f"🗺️ Sahne haritası oluşturuluyor...")

    max_tags = 10
    try:
        from src.utils import load_config

        max_tags = int(load_config().get("sound_matching", {}).get("max_scene_tags", 10))
    except Exception:
        pass
    if max_tags < 3:
        max_tags = 10

    with open(scenes_path) as f:
        scenes = json.load(f)

    with open(descriptions_path) as f:
        descriptions = json.load(f)

    # Description'ları scene_id'ye göre map'le
    desc_map = {d["scene_id"]: d for d in descriptions}

    scene_map = []
    for scene in scenes:
        sid = scene["scene_id"]
        desc = desc_map.get(sid, {})

        analysis = desc.get("openrouter_analysis") or {}

        # Eğer error varsa boş analysis kullan
        if "error" in analysis or "raw_response" in analysis:
            # Raw response'u parse etmeyi dene
            if "raw_response" in analysis:
                parsed = try_parse_raw_response(analysis["raw_response"])
                if parsed:
                    analysis = parsed
                else:
                    analysis = {}
            else:
                analysis = {}

        analysis = _normalize_analysis_fields(analysis)
        tags = extract_tags_from_analysis(analysis, max_tags=max_tags)

        # Blank sahne bilgisini de map'e taşı (sound_matcher ve exporter için)
        blank_info = scene.get("blank_info") or {}
        is_blank = bool(blank_info.get("is_blank", False))

        fr_by_id = {}
        if frame_records:
            for r in frame_records:
                fr_by_id[r["scene_id"]] = r
        fr = fr_by_id.get(sid, {})

        dur = scene["duration"]
        sound_desc = resolve_sound_description(analysis)

        scene_entry = {
            "scene_id": sid,
            "start": scene["start"],
            "end": scene["end"],
            "start_frame": scene.get("start_frame"),
            "end_frame": scene.get("end_frame"),
            "duration": dur,
            "frame_path": fr.get("frame_path"),
            "sound_description": sound_desc,
            "tags": tags,
            "mood": analysis.get("mood", ""),
            "environment": analysis.get("environment", ""),
            "time_of_day": analysis.get("time_of_day", ""),
            "intensity": analysis.get("intensity", 0.5),
            "negative_elements": analysis.get("negative_elements") or [],
            "primary_focus": analysis.get("primary_focus", ""),
            "sound_layers": analysis.get("sound_layers") or [],
            "target_duration": dur,
            "duration_flexibility": duration_flexibility_for(dur),
            "loop_acceptable": dur > 5.0,
            "is_blank": is_blank,
            "blank_info": blank_info if blank_info else None,
        }

        scene_map.append(scene_entry)

        print(f"   Sahne {sid}: {len(tags)} tag oluşturuldu")

    # JSON'a kaydet
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scene_map, f, indent=2, ensure_ascii=False)

    print(f"✅ Sahne haritası hazır → {output_path}")
    return scene_map


def try_parse_raw_response(raw_text):
    """
    Raw AI response'undan JSON çıkarmayı dener.

    Args:
        raw_text: AI'dan gelen ham metin

    Returns:
        Parse edilmiş dict veya None
    """
    try:
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass

    return None


def _clean_tag_unicode(tag: str) -> str:
    """
    Unicode harfleri koru (Türkçe vb.); gereksiz noktalama azalt.
    <2sec gibi ifadelerde anlam kaybını azaltmak için < sayı → 'under'.
    """
    s = unicodedata.normalize("NFKC", tag.lower().strip())
    s = re.sub(r"<\s*(\d)", r"under \1", s)
    s = re.sub(r">\s*(\d)", r"over \1", s)
    parts = []
    for ch in s:
        if ch.isalnum():
            parts.append(ch)
        elif ch in (" ", "-", "'", "\u2019"):  # ASCII ve tipografik apostrof
            parts.append(ch if ch != "\u2019" else "'")
        elif ch.isspace():
            parts.append(" ")
    out = "".join(parts)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def extract_tags_from_analysis(analysis, max_tags: int = 10):
    """
    AI analysis'ten keyword'leri çıkar.

    Args:
        analysis: AI analiz sonucu dictionary
        max_tags: Üst sınır (config: sound_matching.max_scene_tags)

    Returns:
        Tag listesi
    """
    tags = []

    if "tags" in analysis and isinstance(analysis["tags"], list):
        tags.extend([str(t).lower().strip() for t in analysis["tags"] if t])

    cleaned = []
    for tag in tags:
        clean_tag = _clean_tag_unicode(tag)
        if len(clean_tag) > 2:
            cleaned.append(clean_tag)

    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "from",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "this",
        "that",
        "these",
        "those",
    }

    unique_tags = []
    seen = set()
    for tag in cleaned:
        if tag not in stop_words and tag not in seen:
            unique_tags.append(tag)
            seen.add(tag)

    return unique_tags[:max_tags]


if __name__ == "__main__":
    print("Pipeline: app.py veya build_scene_map / build_scene_map_from_frames import edin.")
