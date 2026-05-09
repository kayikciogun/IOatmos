"""
IOatmos / ATM Design - UCS v8.2.1 Pipeline Module
Universal Category System entegrasyonu için yardımcı sözlük ve fonksiyonlar.
"""

# 3B LLM prompt için: vocabulary subset (37 kategori)
PROMPT_VOCABULARY = {
    # Outdoor urban
    'AMB-URBAN':        {'cat': 'AMBIENCE',  'sub': 'URBAN',           'catid': 'AMBUrbn',   'desc': 'Dense city, traffic, pedestrians, horns'},
    'AMB-SUBURBAN':     {'cat': 'AMBIENCE',  'sub': 'SUBURBAN',        'catid': 'AMBSubn',   'desc': 'Quiet residential street, lawnmowers, birds'},
    'AMB-TOWN':         {'cat': 'AMBIENCE',  'sub': 'TOWN',            'catid': 'AMBTown',   'desc': 'Small town village, light activity'},
    'AMB-TRAFFIC':      {'cat': 'AMBIENCE',  'sub': 'TRAFFIC',         'catid': 'AMBTraf',   'desc': 'Pure traffic, highway, no pedestrians'},
    'AMB-PARK':         {'cat': 'AMBIENCE',  'sub': 'PARK',            'catid': 'AMBPark',   'desc': 'City park, light human activity'},

    # Outdoor nature
    'AMB-FOREST':       {'cat': 'AMBIENCE',  'sub': 'FOREST',          'catid': 'AMBForst',  'desc': 'Trees, birds, wind in trees'},
    'AMB-RURAL':        {'cat': 'AMBIENCE',  'sub': 'RURAL',           'catid': 'AMBRurl',   'desc': 'Countryside, away from people'},
    'AMB-ALPINE':       {'cat': 'AMBIENCE',  'sub': 'ALPINE',          'catid': 'AMBAlpn',   'desc': 'Mountain ambience'},
    'AMB-DESERT':       {'cat': 'AMBIENCE',  'sub': 'DESERT',          'catid': 'AMBDsrt',   'desc': 'Quiet desert, insects'},
    'AMB-SEASIDE':      {'cat': 'AMBIENCE',  'sub': 'SEASIDE',         'catid': 'AMBSea',    'desc': 'Beach with people, oceanside'},
    'AMB-LAKESIDE':     {'cat': 'AMBIENCE',  'sub': 'LAKESIDE',        'catid': 'AMBLake',   'desc': 'Lake scenes'},
    'AMB-TROPICAL':     {'cat': 'AMBIENCE',  'sub': 'TROPICAL',        'catid': 'AMBTrop',   'desc': 'Jungle, rainforest'},
    'AMB-FARM':         {'cat': 'AMBIENCE',  'sub': 'FARM',            'catid': 'AMBFarm',   'desc': 'Farm with animals, tractors'},

    # Indoor public
    'AMB-OFFICE':       {'cat': 'AMBIENCE',  'sub': 'OFFICE',          'catid': 'AMBOffc',   'desc': 'Workplace, typing, phones'},
    'AMB-RESTAURANT':   {'cat': 'AMBIENCE',  'sub': 'RESTAURANT & BAR','catid': 'AMBRest',   'desc': 'Restaurants, bars, dining'},
    'AMB-PUBLIC':       {'cat': 'AMBIENCE',  'sub': 'PUBLIC PLACE',    'catid': 'AMBPubl',   'desc': 'Stores, lobbies, malls, museums'},
    'AMB-MARKET':       {'cat': 'AMBIENCE',  'sub': 'MARKET',          'catid': 'AMBMrkt',   'desc': 'Busy market, vendors, crowds'},
    'AMB-SCHOOL':       {'cat': 'AMBIENCE',  'sub': 'SCHOOL',          'catid': 'AMBSchl',   'desc': 'Classroom, hallway, school bells'},
    'AMB-HOSPITAL':     {'cat': 'AMBIENCE',  'sub': 'HOSPITAL',        'catid': 'AMBHosp',   'desc': 'Hospital, surgery, ER'},
    'AMB-RELIGIOUS':    {'cat': 'AMBIENCE',  'sub': 'RELIGIOUS',       'catid': 'AMBRlgn',   'desc': 'Church, temple, services'},
    'AMB-TRANSPORT':    {'cat': 'AMBIENCE',  'sub': 'TRANSPORTATION',  'catid': 'AMBTran',   'desc': 'Train station, airport, bus station'},

    # Indoor private
    'AMB-RESIDENTIAL':  {'cat': 'AMBIENCE',  'sub': 'RESIDENTIAL',     'catid': 'AMBHome',   'desc': 'Apartment, house with some activity'},
    'AMB-ROOM-TONE':    {'cat': 'AMBIENCE',  'sub': 'ROOM TONE',       'catid': 'AMBRoom',   'desc': 'Empty room, no activity, just air'},

    # Industrial
    'AMB-CONSTRUCTION': {'cat': 'AMBIENCE',  'sub': 'CONSTRUCTION',    'catid': 'AMBCnst',   'desc': 'Jackhammers, cranes, road construction'},
    'AMB-INDUSTRIAL':   {'cat': 'AMBIENCE',  'sub': 'INDUSTRIAL',      'catid': 'AMBInd',    'desc': 'Warehouse, factory, plant'},
    'AMB-HITECH':       {'cat': 'AMBIENCE',  'sub': 'HITECH',          'catid': 'AMBTech',   'desc': 'Data center, control room, lab'},

    # Special acoustic
    'AMB-UNDERGROUND':  {'cat': 'AMBIENCE',  'sub': 'UNDERGROUND',     'catid': 'AMBUndr',   'desc': 'Cave, sewer, bunker, tunnel, parking'},
    'AMB-UNDERWATER':   {'cat': 'AMBIENCE',  'sub': 'UNDERWATER',      'catid': 'AMBUndwtr', 'desc': 'Deep underwater bubbles'},
    'AMB-NAUTICAL':     {'cat': 'AMBIENCE',  'sub': 'NAUTICAL',        'catid': 'AMBNaut',   'desc': 'Ships at sea, shipyard'},

    # Water foreground
    'WATR-SURF':        {'cat': 'WATER',     'sub': 'SURF',            'catid': 'WATRSurf',  'desc': 'Ocean roar, constant waves (above water)'},
    'WATR-WAVE':        {'cat': 'WATER',     'sub': 'WAVE',            'catid': 'WATRWave',  'desc': 'Distinct individual waves'},
    'WATR-WATERFALL':   {'cat': 'WATER',     'sub': 'WATERFALL',       'catid': 'WATRFall',  'desc': 'Waterfalls'},
    'WATR-FLOW':        {'cat': 'WATER',     'sub': 'FLOW',            'catid': 'WATRFlow',  'desc': 'River, creek, stream'},
    'WATR-UNDERWATER':  {'cat': 'WATER',     'sub': 'UNDERWATER',      'catid': 'WATRUndwtr','desc': 'Underwater swimming, currents'},

    # Vehicle foreground
    'VEH-INTERIOR':     {'cat': 'VEHICLES',  'sub': 'INTERIOR',        'catid': 'VEHInt',    'desc': 'Car/truck interior ambience'},
    'VEH-CAR':          {'cat': 'VEHICLES',  'sub': 'CAR',             'catid': 'VEHCar',    'desc': 'Cars (engine, by, idle)'},

    # Crowd foreground
    'CRWD-WALLA':       {'cat': 'CROWDS',    'sub': 'WALLA',           'catid': 'CRWDWalla', 'desc': 'Clean crowd murmur, no words'},
}


# Anti-pattern map: hangi kategori hangisinin antitezi
# CLAP retrieval'ında negative_query için kullanılır
ANTI_PATTERN_MAP = {
    'AMB-URBAN':        'AMB-UNDERGROUND',    # urban ↔ underground (parking/indoor)
    'AMB-SUBURBAN':     'AMB-INDUSTRIAL',     # quiet residential ↔ industrial
    'AMB-TOWN':         'AMB-FOREST',
    'AMB-TRAFFIC':      'AMB-FOREST',
    'AMB-PARK':         'AMB-INDUSTRIAL',

    'AMB-FOREST':       'AMB-URBAN',          # nature ↔ urban
    'AMB-RURAL':        'AMB-URBAN',
    'AMB-ALPINE':       'AMB-MARKET',
    'AMB-DESERT':       'AMB-TROPICAL',       # dry quiet ↔ wet noisy
    'AMB-SEASIDE':      'AMB-OFFICE',         # outdoor beach ↔ indoor office
    'AMB-LAKESIDE':     'AMB-URBAN',
    'AMB-TROPICAL':     'AMB-DESERT',
    'AMB-FARM':         'AMB-OFFICE',

    'AMB-OFFICE':       'AMB-FOREST',         # indoor work ↔ outdoor nature
    'AMB-RESTAURANT':   'AMB-ROOM-TONE',      # crowded eating ↔ silent room
    'AMB-PUBLIC':       'AMB-FOREST',
    'AMB-MARKET':       'AMB-ROOM-TONE',
    'AMB-SCHOOL':       'AMB-ROOM-TONE',
    'AMB-HOSPITAL':     'AMB-FOREST',
    'AMB-RELIGIOUS':    'AMB-MARKET',         # quiet sacred ↔ noisy commercial
    'AMB-TRANSPORT':    'AMB-FOREST',

    'AMB-RESIDENTIAL':  'AMB-CONSTRUCTION',   # quiet home ↔ noisy construction
    'AMB-ROOM-TONE':    'AMB-CONSTRUCTION',   # silence ↔ heavy machinery

    'AMB-CONSTRUCTION': 'AMB-ROOM-TONE',
    'AMB-INDUSTRIAL':   'AMB-FOREST',
    'AMB-HITECH':       'AMB-FOREST',

    'AMB-UNDERGROUND':  'AMB-SEASIDE',        # closed underground ↔ open beach
    'AMB-UNDERWATER':   'AMB-DESERT',         # wet submerged ↔ dry open
    'AMB-NAUTICAL':     'AMB-INDUSTRIAL',

    'WATR-SURF':        'AMB-OFFICE',         # ocean ↔ indoor office
    'WATR-WAVE':        'AMB-ROOM-TONE',
    'WATR-WATERFALL':   'AMB-OFFICE',
    'WATR-FLOW':        'AMB-URBAN',
    'WATR-UNDERWATER':  'AMB-DESERT',

    'VEH-INTERIOR':     'AMB-FOREST',         # inside car ↔ outdoor nature
    'VEH-CAR':          'AMB-RELIGIOUS',      # noisy car ↔ quiet church

    'CRWD-WALLA':       'AMB-ROOM-TONE',      # crowd ↔ silence
}


# Rafine Negatif Overrides: 
# Otomatik haritalama bazen çok geniş kalabilir (örn: seaside -> underwater dediğimizde seaside içinde ocean geçtiği için seaside'ı da vurur)
# Buraya nokta atışı negatif metinleri giriyoruz.
NEGATIVE_QUERY_OVERRIDES = {
    'WATR-SURF':   'submerged hydrophone below surface deep water recording',
    'WATR-WAVE':   'submerged hydrophone below surface deep water recording',
    'AMB-SEASIDE': 'submerged hydrophone below surface deep water recording',
    'AMB-URBAN':   'indoor enclosed parking underground garage quiet',
}

# Kategori → izin verilen/yasaklanan dosya pattern'leri (Metadata Filtering)
CATEGORY_FILTERS = {
    'WATR-SURF': {
        'exclude_patterns': ['UWT', 'underwater', 'submerged', 'hydrophone', 'below water'],
        'require_patterns': []
    },
    'WATR-WAVE': {
        'exclude_patterns': ['UWT', 'underwater', 'submerged', 'hydrophone'],
        'require_patterns': []
    },
    'AMB-SEASIDE': {
        'exclude_patterns': ['UWT', 'underwater', 'submerged', 'hydrophone'],
        'require_patterns': []
    },
    'AMB-URBAN': {
        'exclude_patterns': ['underground', 'car park', 'basement', 'UWT', 'garage'],
        'require_patterns': []
    },
    'AMB-ROOM-TONE': {
        'exclude_patterns': ['UWT', 'outdoor', 'exterior', 'traffic', 'birds'],
        'require_patterns': []
    },
    'AMB-FOREST': {
        'exclude_patterns': ['traffic', 'car', 'city', 'urban', 'industrial'],
        'require_patterns': []
    },
}

# UCS Sub-Index Map: Kategoriye göre izin verilen UCS CatID'ler
UCS_CATEGORY_MAP = {
    'WATR-SURF':    ['WATRSurf', 'WATRWave', 'AMBSea'],
    'WATR-WAVE':    ['WATRWave', 'WATRSurf'],
    'AMB-SEASIDE':  ['AMBSea', 'WATRSurf'],
    'AMB-URBAN':    ['AMBUrbn', 'AMBTown', 'AMBTraf'],
    'AMB-ROOM-TONE':['AMBRoom', 'AMBHome'],
    'AMB-UNDERGROUND': ['AMBUndr'],
    'AMB-FOREST':   ['AMBForst', 'AMBRurl', 'AMBAlpn'],
}


def get_negative_for(positive_key: str) -> str:
    """Bir pozitif kategori için anti-pattern negative kategoriyi döndür."""
    return ANTI_PATTERN_MAP.get(positive_key, 'AMB-ROOM-TONE')  # fallback


def category_to_clap_query(key: str, time: str = 'Day', perspective: str = '') -> str:
    """UCS kategori key'ini CLAP-friendly text query'ye çevir.
    Örn: 'AMB-FOREST' → 'forest ambience trees birds wind'
    """
    if key not in PROMPT_VOCABULARY:
        return key.lower().replace('-', ' ')
    item = PROMPT_VOCABULARY[key]
    # CatID'yi human-readable'a çevir + description'dan keyword
    parts = [item['sub'].lower(), item['cat'].lower()]
    parts.append(item['desc'].lower().split(',')[0])  # ilk açıklama parçası
    if time:
        parts.append(time.lower())
    if perspective:
        parts.append(perspective.lower())
    return ' '.join(parts)


# 3B LLM için kompakt prompt
SIMPLE_DESIGNER_PROMPT = """Describe sound in image. Output 2 lines:

POSITIVE: CATEGORY - Time - Perspective - Sound
NEGATIVE: Different CATEGORY - opposite perspective - different sound

CATEGORY (use ONE):
AMB-URBAN, AMB-SUBURBAN, AMB-TOWN, AMB-TRAFFIC, AMB-PARK,
AMB-FOREST, AMB-RURAL, AMB-ALPINE, AMB-DESERT, AMB-SEASIDE,
AMB-LAKESIDE, AMB-TROPICAL, AMB-FARM, AMB-OFFICE, AMB-RESTAURANT,
AMB-PUBLIC, AMB-MARKET, AMB-SCHOOL, AMB-HOSPITAL, AMB-RELIGIOUS,
AMB-TRANSPORT, AMB-RESIDENTIAL, AMB-ROOM-TONE, AMB-CONSTRUCTION,
AMB-INDUSTRIAL, AMB-HITECH, AMB-UNDERGROUND, AMB-UNDERWATER,
AMB-NAUTICAL, WATR-SURF, WATR-WAVE, WATR-WATERFALL, WATR-FLOW,
WATR-UNDERWATER, VEH-INTERIOR, VEH-CAR, CRWD-WALLA

TIME: Day or Night
PERSPECTIVE: Indoor Close, Indoor Empty, Outdoor Distant, Outdoor Close

Empty scene = AMB-ROOM-TONE or AMB-RESIDENTIAL
Negative CATEGORY must be DIFFERENT from positive.
Never use "no" or "quiet" in sound description.

Examples:

POSITIVE: AMB-ROOM-TONE - Day - Indoor Empty - Refrigerator Hum, Air
NEGATIVE: AMB-TRAFFIC - Day - Outdoor Distant - Cars, Engines

POSITIVE: AMB-TRAFFIC - Day - Outdoor Distant - Highway Cars, Engines
NEGATIVE: AMB-FOREST - Day - Outdoor Distant - Birds, Wind in Trees

POSITIVE: AMB-FOREST - Day - Outdoor Distant - Birds, Wind in Leaves
NEGATIVE: AMB-URBAN - Day - Outdoor Close - Traffic, Crowds

POSITIVE: WATR-SURF - Day - Outdoor Close - Ocean Waves Crashing on Shore
NEGATIVE: AMB-OFFICE - Day - Indoor Close - Typing, Phones

POSITIVE: AMB-CONSTRUCTION - Day - Outdoor Close - Hammers, Heavy Machinery
NEGATIVE: AMB-ROOM-TONE - Day - Indoor Empty - Quiet Air

Now describe:"""


def parse_llm_output(text: str) -> dict:
    """LLM çıktısını parse et: POSITIVE ve NEGATIVE satırlarını ayır."""
    result = {'positive': None, 'negative': None, 'pos_category': None, 'neg_category': None}
    for line in text.strip().split('\n'):
        line = line.strip()
        if line.upper().startswith('POSITIVE:'):
            content = line.split(':', 1)[1].strip()
            result['positive'] = content
            cat = content.split(' - ')[0].strip().upper()
            if cat in PROMPT_VOCABULARY:
                result['pos_category'] = cat
        elif line.upper().startswith('NEGATIVE:'):
            content = line.split(':', 1)[1].strip()
            result['negative'] = content
            cat = content.split(' - ')[0].strip().upper()
            if cat in PROMPT_VOCABULARY:
                result['neg_category'] = cat
    return result


def validate_negative(parsed: dict) -> dict:
    """Negative'in geçerli olup olmadığını kontrol et, gerekirse düzelt."""
    if not parsed['pos_category']:
        return {**parsed, 'negative_status': 'invalid_positive'}

    # 1. Negative kategorisi pozitif ile aynıysa REDDET
    if parsed['neg_category'] == parsed['pos_category']:
        # Anti-pattern map'ten otomatik düzelt
        forced_neg = get_negative_for(parsed['pos_category'])
        return {**parsed, 'neg_category': forced_neg, 'negative_status': 'auto_corrected_same'}

    # 2. Negative kategorisi vocabulary dışındaysa REDDET ve düzelt
    if not parsed['neg_category']:
        forced_neg = get_negative_for(parsed['pos_category'])
        return {**parsed, 'neg_category': forced_neg, 'negative_status': 'auto_corrected_invalid'}

    # 3. Tehlikeli kelimeler kontrolü
    danger = ['no ', 'without', 'quiet', 'silent', 'muffled']
    if parsed['negative'] and any(w in parsed['negative'].lower() for w in danger):
        forced_neg = get_negative_for(parsed['pos_category'])
        return {**parsed, 'neg_category': forced_neg, 'negative_status': 'auto_corrected_danger'}

    return {**parsed, 'negative_status': 'ok'}


if __name__ == '__main__':
    # Test
    sample = """POSITIVE: AMB-FOREST - Day - Outdoor Distant - Birds, Wind in Leaves
NEGATIVE: AMB-URBAN - Day - Outdoor Close - Traffic, Crowds"""

    parsed = parse_llm_output(sample)
    validated = validate_negative(parsed)
    print("Parsed:", parsed)
    print("Validated:", validated)
    print()
    print("CLAP query (positive):", category_to_clap_query(parsed['pos_category']))
    print("CLAP query (negative):", category_to_clap_query(parsed['neg_category']))
