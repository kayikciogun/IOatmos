import os
import re
import json
import time
import subprocess
import urllib.request
import urllib.error
import base64

# ============================================================
#   VLM (Vision Language Model) Ayarları & Promptlar
# ============================================================

DESCRIPTION_CATEGORY_HINTS = {
    'forest': 'AMB-FOREST',
    'city': 'AMB-URBAN',
    'traffic': 'AMB-URBAN',
    'highway': 'AMB-TRAFFIC',
    'airport': 'AMB-PUBLIC',
    'beach': 'AMB-SEASIDE',
    'ocean': 'AMB-SEASIDE',
    'desert': 'AMB-DESERT',
    'tunnel': 'AMB-UNDERGROUND',
    'cave': 'AMB-UNDERGROUND',
    'construction': 'AMB-CONSTRUCTION',
    'fireplace': 'AMB-RESIDENTIAL',
    'lake': 'AMB-LAKESIDE',
    'river': 'AMB-RURAL',
}

def infer_category_from_description(desc: str) -> str:
    """Description içindeki kelimelere bakarak kategori tahmin et (Fallback)."""
    desc_lower = desc.lower()
    for keyword, category in DESCRIPTION_CATEGORY_HINTS.items():
        if keyword in desc_lower:
            return category
    return 'AMB-ROOM-TONE'

LLAMA_SERVER_PORT = 8787
LLAMA_SERVER_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"

# Kullanıcının son güncellediği rafine prompt
SIMPLE_DESIGNER_PROMPT = """Look at the image. Output exactly 2 lines:

CATEGORY: [pick ONE from list]
DESCRIPTION: [natural language description of what you hear, 10-15 words]

CATEGORY list:
AMB-URBAN, AMB-TRAFFIC, AMB-SUBURBAN, AMB-PARK,
AMB-FOREST, AMB-RURAL, AMB-SEASIDE, AMB-LAKESIDE, AMB-FARM, AMB-DESERT,
AMB-OFFICE, AMB-RESTAURANT, AMB-PUBLIC, AMB-MARKET, AMB-SCHOOL,
AMB-RESIDENTIAL, AMB-ROOM-TONE,
AMB-CONSTRUCTION, AMB-INDUSTRIAL,
AMB-UNDERGROUND, AMB-NAUTICAL,
WATR-SURF, WATR-WATERFALL, WATR-FLOW,
VEH-INTERIOR, CRWD-WALLA

Critical rules:
- CATEGORY must be ONE from the list above.
- DESCRIPTION must be a natural language description of audible sounds.
- Do not describe visual-only elements like shadows, light, or colors.

Examples:
CATEGORY: AMB-URBAN
DESCRIPTION: Dense city street with trams, buses, cars and pedestrians in daytime traffic

CATEGORY: AMB-DESERT  
DESCRIPTION: Quiet open desert with distant wind and sand movement

CATEGORY: AMB-ROOM-TONE
DESCRIPTION: Empty quiet room with subtle refrigerator hum and air conditioning

Now describe:"""

def parse_vlm_response(raw_text: str) -> dict:
    """VLM yanıtını parse eder. CATEGORY ve DESCRIPTION alanlarını ayırır."""
    if not raw_text or not raw_text.strip():
        return {'category': 'AMB-ROOM-TONE', 'sound_description': '[no response]', 'silence_required': False}

    # Temizlik (Resim tagleri, Düşünce tagleri ve Markdown temizliği)
    raw_text = re.sub(r'\[img-\d+\]', '', raw_text)
    raw_text = re.sub(r'<思考>.*?</思考>', '', raw_text, flags=re.DOTALL)
    raw_text = re.sub(r'<thinking>.*?</thinking>', '', raw_text, flags=re.DOTALL)
    raw_text = raw_text.replace('**', '').replace('*', '') # Markdown temizliği
    
    category = None
    description = ""
    
    cat_match = re.search(r'CATEGORY:\s*([\w-]+)', raw_text, re.IGNORECASE)
    if cat_match:
        category = cat_match.group(1).strip().upper()
        
    desc_match = re.search(r'DESCRIPTION:\s*(.*)', raw_text, re.IGNORECASE)
    if desc_match:
        description = desc_match.group(1).strip()
    
    # Fallback: Eğer regex DESCRIPTION bulamadıysa ama metin varsa
    if not description:
        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
        for line in lines:
            if "CATEGORY:" in line.upper(): continue
            if "DESCRIPTION:" in line.upper(): continue
            if not any(skip in line.lower() for skip in ["thinking", "analyze"]):
                description = line.strip()
                break

    # Fallback: Eğer CATEGORY bulunamadıysa description'dan tahmin et
    if not category or category == "UNKNOWN":
        category = infer_category_from_description(description)

    return {
        'category': category,
        'sound_description': description,
        'silence_required': False
    }

def start_llama_server(base_model, mmproj):
    """llama-server'ı arka planda başlatır."""
    server_path = "llama.cpp/build/bin/llama-server"
    if not os.path.exists(server_path):
        print("   ⚠️ llama-server bulunamadı!")
        return None
    
    # Port temizliği
    try:
        subprocess.run(f"lsof -ti:{LLAMA_SERVER_PORT} | xargs kill -9", shell=True, stderr=subprocess.DEVNULL)
        time.sleep(1)
    except: pass

    cmd = [
        f"./{server_path}", "-m", base_model, "--mmproj", mmproj,
        "--port", str(LLAMA_SERVER_PORT), "-c", "4096", "-ngl", "99",
        "-fa", "on", "-t", "8", "--temp", "0.2", "--log-disable"
    ]
    
    print(f"   ⏳ VLM Sunucusu başlatılıyor (port {LLAMA_SERVER_PORT})...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, preexec_fn=os.setsid)
    
    max_wait = 120
    for i in range(max_wait):
        try:
            req = urllib.request.urlopen(f"{LLAMA_SERVER_URL}/health", timeout=2)
            if json.loads(req.read().decode()).get("status") == "ok":
                print(f"   ✅ VLM Sunucusu hazır ({i+1}sn)")
                return proc
        except: pass
        if proc.poll() is not None: break
        time.sleep(1)
    
    print("   ❌ VLM Sunucusu başlatılamadı!")
    return None

def query_llama_server(image_path, retries=3):
    """Görüntüyü sunucuya gönderir."""
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = json.dumps({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": SIMPLE_DESIGNER_PROMPT}
            ]}],
            "temperature": 0.2, "max_tokens": 200,
        }).encode("utf-8")

        req = urllib.request.Request(f"{LLAMA_SERVER_URL}/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"})
        
        for _ in range(retries):
            try:
                resp = urllib.request.urlopen(req, timeout=90)
                msg = json.loads(resp.read().decode())["choices"][0]["message"]
                raw = msg.get("content", "").strip() or msg.get("reasoning_content", "").strip()
                return parse_vlm_response(raw)
            except: time.sleep(2)
    except Exception as e:
        print(f"   ⚠️ VLM Sorgu hatası: {e}")
    return parse_vlm_response("")

def stop_llama_server(proc):
    """Sunucuyu durdurur."""
    if not proc: return
    try:
        os.killpg(os.getpgid(proc.pid), 15)
        proc.wait(timeout=5)
    except:
        try: proc.kill()
        except: pass
    print("   🛑 VLM Sunucusu durduruldu")

def analyze_with_llamacpp_cli(image_path, base_model, mmproj):
    """CLI Fallback (Yavaş mod)."""
    cli_path = "llama.cpp/build/bin/llama-cli"
    cmd = [f"./{cli_path}", "-m", base_model, "--mmproj", mmproj, "--image", image_path, "-p", SIMPLE_DESIGNER_PROMPT, "-n", "150", "--temp", "0.2"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return parse_vlm_response(result.stdout)
