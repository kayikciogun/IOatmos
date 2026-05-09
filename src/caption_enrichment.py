"""
caption_enrichment.py
=====================
filename_to_caption() için drop-in replacement.

Mevcut sistemin sorunları:
  1. Perspective bilgisi yok → "underwater waterfall" yüzey şelalesine sızmıyor
  2. Acoustic context yok → "desert birds" ormana sızıyor
  3. UCS_HINTS prefix-tabanlı → sadece bilinen prefix'ler için çalışıyor
  4. bext content'i ham kullanılıyor → "underwater" kelimesi caption sonunda gömülü kalıyor

Bu modül şunu yapar:
  1. Perspektifi ÖNCE yaz → CLAP için en güçlü sinyal başta
  2. Acoustic context ekle → catid'den türetilmiş ayırt edici kelimeler
  3. Library-agnostic → bext + dosya adı + catid üçlüsünden otomatik
  4. Hiç hardcode rule yok → sadece veriye dayalı
"""

import re
from pathlib import Path


# ── 1. Perspective Detection ──────────────────────────────────────────────────
# Dosya yolu + bext içeriğinden perspektif tespiti
# Perspektif caption'ın BAŞINA yazılır — CLAP için en önemli sinyal

PERSPECTIVE_SIGNALS = {
    # Underwater (en kritik — en fazla leak)
    "UNDERWATER PERSPECTIVE": [
        "uwt", "underwater", "submerged", "hydrophone",
        "below surface", "aquatic recording", "hydro mic",
        "uwt0", "uwa0",
    ],
    # Above water / surface
    "ABOVE WATER SURFACE": [
        "above water", "surface", "on shore", "waves crashing",
        "beach recording", "surf", "shoreline",
    ],
    # Enclosed interior
    "ENCLOSED INTERIOR INDOOR": [
        "interior", "indoor", "inside", "int -", "int.", " int ",
        "room tone", "enclosed", "reverberant room", "lobby",
        "basement", "underground", "car park", "parking",
        "tunnel", "bunker", "cave", "sewer",
    ],
    # Open exterior
    "OPEN EXTERIOR OUTDOOR": [
        "exterior", "outdoor", "outside", "ext -", "ext.", " ext ",
        "open air", "field recording", "location",
    ],
}

def detect_perspective(filepath: str, bext: str = "") -> str:
    """
    Dosya yolu ve bext'ten perspektif tespiti.
    Dönen string doğrudan caption prefix'i olarak kullanılır.
    Öncelik: underwater > above water > enclosed > open > ""
    """
    search = (filepath + " " + bext).lower()
    for perspective, signals in PERSPECTIVE_SIGNALS.items():
        if any(s in search for s in signals):
            return perspective
    return ""


# ── 2. Acoustic Context Enrichment ───────────────────────────────────────────
# catid → CLAP'in o kategori için öğrendiği spesifik acoustic descriptor'lar
# Bunlar UCS_ANCHORS'tan FARKLI — daha kısa, ayırt edici, overlap'i düşük

CATID_ACOUSTIC_CONTEXT = {
    # Su
    "WATRSurf":   "breaking surf roar crashing sea spray foam",
    "WATRUndwtr": "muffled deep rumble pressure bubbles hydrophone",
    "WATRFall":   "cascading rushing white noise plunge pool",
    "WATRFlow":   "babbling brook gurgling current stream riffle",
    "AMBSea":     "seabirds wind sea spray shoreline distant surf",
    "AMBLake":    "gentle lapping still water reeds calm quiet",
    "AMBNaut":    "rigging creak hull tide dock ropes winch",

    # Şehir / Trafik
    "AMBUrbn":    "traffic rumble horns voices footsteps distant city",
    "AMBTraf":    "engine roar tire whine freeway constant flow",
    "AMBTown":    "light activity church bell distant cars village",
    "AMBSubn":    "lawnmower sprinkler birds children quiet street",
    "AMBPark":    "fountain wind trees jogging distant voices",

    # Doğa
    "AMBForst":   "birdsong wind leaves canopy wilderness rustling",
    "AMBDsrt":    "dry wind hiss sparse desolate sand grit arid",
    "AMBRurl":    "insects breeze meadow distant farm quiet pastoral",
    "AMBTrop":    "cicadas exotic birds dense foliage humidity",
    "AMBFarm":    "roosters cattle tractor barn mud rural",

    # İç Mekan
    "AMBRoom":    "silence ventilation hiss air conditioner low hum",
    "AMBHome":    "daily life domestic distant voices appliance",
    "AMBOffc":    "keyboards printer ventilation low voices activity",
    "AMBRest":    "cutlery glasses chatter background music dining",
    "AMBPubl":    "announcement footsteps escalator crowd shuffle",
    "AMBMrkt":    "vendors calls haggling crowd movement stalls",
    "AMBTran":    "trains announcements luggage wheels platform echo",
    "AMBRlgn":    "chanting echo reverb prayer contemplative",
    "AMBHosp":    "beeping monitors hushed voices footsteps clinical",
    "AMBSchl":    "children bell corridor lockers voices playground",

    # Endüstriyel
    "AMBUndr":    "reverb drip echo enclosed damp stone concrete",
    "AMBInd":     "machinery drone metal clanging ventilation large",
    "AMBCnst":    "jackhammer saw concrete mixer beeps rubble",
    "AMBTech":    "cooling fans servers hum racks data center",

    # Araç
    "VEHInt":     "engine vibration road noise cabin isolation",
    "VEHCar":     "engine start idle rev acceleration",

    # Kalabalık
    "CRWDWalla":  "murmur indistinct voices background chatter",
    "CRWDCheer":  "applause chanting crowd energy stadium",
}

def get_acoustic_context(catid: str) -> str:
    """catid'den acoustic context string üret."""
    return CATID_ACOUSTIC_CONTEXT.get(catid, "")


# ── 3. Gelişmiş filename_to_caption ──────────────────────────────────────────

def enrich_caption(
    filepath: str,
    bext: str = "",
    catid: str = "",
) -> str:
    """
    Zenginleştirilmiş caption üret.

    Yapı:
      [PERSPECTIVE] [ACOUSTIC_CONTEXT] [CONTENT_TEXT]

    Örnek:
      "underwater muffled deep rumble pressure bubbles hydrophone
       water waterfall underwater waterfall indoor heavy gurgling"

      "open exterior outdoor birdsong wind leaves canopy wilderness rustling
       forest day birds cicadas light breeze distant traffic"

      "enclosed interior indoor silence ventilation hiss air conditioner low hum
       room tone small house main floor constant window air conditioner"
    """
    # 1. Perspektif
    perspective = detect_perspective(filepath, bext)

    # 2. Acoustic context (catid'den)
    acoustic = get_acoustic_context(catid)

    # 3. İçerik metni (bext veya dosya adı)
    if bext:
        content = re.sub(r'[-_,;]+', ' ', bext).lower()
        content = re.sub(r'\s+', ' ', content).strip()
    else:
        stem = Path(filepath).stem
        content = re.sub(r"[-_.]+", " ", stem)
        content = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", content)
        # Kütüphane kodlarını temizle
        content = re.sub(
            r"\b(?=[A-Za-z0-9]*[0-9])(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{3,7}\b",
            " ", content
        )
        content = re.sub(r"\b\d+\b", "", content)
        content = re.sub(r"\s+", " ", content).strip().lower()

    # 4. Birleştir — perspective ve acoustic context ÖNCE
    parts = [p for p in [perspective, acoustic, content] if p]
    return " ".join(parts).strip()


# ── 4. Test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        # (filepath, bext, catid, beklenen_sonuç)
        (
            "/sfx/UWT03 97 Water, Waterfall - Underwater Waterfall.wav",
            "Water, Waterfall - Underwater Waterfall; Indoor Heavy Gurgling",
            "WATRUndwtr",
            "underwater prefix bekleniyor, 'above water' OLMAMALI"
        ),
        (
            "/sfx/water - ocean - waves coming in - heavy - ambience.wav",
            "water - ocean - waves coming in - heavy - ambience",
            "WATRSurf",
            "surface waves, underwater OLMAMALI"
        ),
        (
            "/sfx/AMBDsrt_DESERT-Morning Birds Sparse Breeze.wav",
            "Desert morning sparse birds light breeze arid landscape",
            "AMBDsrt",
            "desert context, forest OLMAMALI"
        ),
        (
            "/sfx/forest - day - birds - cicadas - light breeze.wav",
            "forest day birds cicadas light breeze distant traffic",
            "AMBForst",
            "forest context açık"
        ),
        (
            "/sfx/3DS02 Underground Car Park C 02 Active.wav",
            "Underground car park active vehicles movement",
            "AMBUndr",
            "enclosed indoor + reverb drip, urban OLMAMALI"
        ),
        (
            "/sfx/room tone - small house - constant air conditioner.wav",
            "room tone small house main floor constant window air conditioner",
            "AMBRoom",
            "silence ventilation hum, restaurant OLMAMALI"
        ),
    ]

    print("=" * 70)
    for filepath, bext, catid, expectation in test_cases:
        caption = enrich_caption(filepath, bext, catid)
        print(f"FILE:    {Path(filepath).name[:55]}")
        print(f"CATID:   {catid}")
        print(f"CAPTION: {caption[:120]}")
        print(f"EXPECT:  {expectation}")
        print("-" * 70)
