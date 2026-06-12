import os
import re
import json
import time
import subprocess
import urllib.request
import base64
import asyncio
import aiohttp
import warnings
from PIL import Image
from pathlib import Path

# .env dosyasını yükle
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================
#   Online (OpenRouter) Ayarları
# ============================================================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip().strip('"').strip("'")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.7-plus"

# ============================================================
#   Local (llama.cpp Server) Ayarları
# ============================================================

LLAMA_SERVER_PORT = 8787
LLAMA_SERVER_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"

# ============================================================
#   Prompt
# ============================================================

SIMPLE_DESIGNER_PROMPT = """You are tagging a shot for a professional sound-effects library.

The image is only a clue. Output THREE DIFFERENT search queries — each targeting distinct audible layers for the same scene. No visual description.

Return ONLY valid JSON, no markdown:

{
  "bed": "...",
  "near": "...",
  "far": "..."
}

- "bed": the base ambience bed (room tone, street ambience, forest atmo — the constant background layer).
- "near": mid-field audible sources (footsteps, walla, machinery, water, birds, ventilation, construction activity).
- "far": distant texture layer (traffic rumble, city drone, wind, industrial hum, thunder, ocean surf).

Each field is a comma-separated keyword query, lowercase, like real library filenames.
Make all three queries DIFFERENT from each other — near and far should NOT repeat the same keywords as bed.

USE THIS VOCABULARY:

Time: day, night, dawn, dusk
Space: interior, exterior, int, ext
Bed: ambience, atmosphere, atmo, room tone, roomtone, environment
Place: urban, city, suburban, rural, industrial, office, restaurant, hospital, airport, alley, street, park, forest, beach, warehouse, laboratory
Distance: distant, far, close, nearby, background, bg
Dynamics: active, busy, quiet, calm, sparse, heavy, light, medium
Traffic: traffic, city traffic, highway, motorway, vehicles passing, traffic rumble, traffic drone
Mechanical: HVAC, air conditioning, ventilation, refrigerator, computer fan, machinery, construction, industrial hum, generator
Nature: wind, breeze, gust, rain, thunder, stream, river, ocean, waterfall
Life: birds, chirping, insects, wildlife, crowd, walla, voices, chatter, footsteps, movement
Texture: rumble, hum, drone, hiss, buzz, rustle, splash, drip, echo, reverb

Good examples:
{
  "bed": "day, exterior, urban, street ambience",
  "near": "footsteps passing, walla crowd, bicycle pass by",
  "far": "city traffic distant, wind light, birds chirping"
}
{
  "bed": "night, interior, room tone, ventilation hum",
  "near": "computer fan, keyboard typing, chair movement, footsteps",
  "far": "traffic muffled distant, city drone, siren faint"
}
{
  "bed": "day, exterior, forest, ambience, wind in trees",
  "near": "stream water, birds chirping, rustle leaves",
  "far": "thunder distant, rain light, traffic bg"
}

Rules:
- bed: always time + int/ext + place + room tone/ambience (4-6 tags).
- near: 3-5 audible foreground elements (humans, animals, moving things).
- far: 2-4 distant/bg texture sources.
- No dialogue, music, or sharp one-shots (gunshot, door slam, car horn).
- 10-18 words per field; lowercase; standard library compound terms.
- If unsure, pick generic ambience elements for that location.
"""

# ============================================================
#   Yanıt Parser
# ============================================================

def _empty_response():
    return {"queries": [], "description": ""}

def _parse_json_response(raw: str) -> dict:
    """Parse JSON: {"bed": "...", "near": "...", "far": "..."} or {"description": "..."}"""
    try:
        obj_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not obj_match:
            return _empty_response()
        data = json.loads(obj_match.group())

        # New format: bed/near/far three-queries
        queries = []
        for key in ("bed", "near", "far"):
            val = str(data.get(key, "")).strip()
            if val:
                queries.append(val)
        if queries:
            return {"queries": queries, "description": " | ".join(queries)}

        # Legacy format: single description
        for key in ("description", "sound_description", "query"):
            if key in data and data[key]:
                desc = str(data[key]).strip()
                return {"queries": [desc], "description": desc}
    except Exception:
        pass
    return _empty_response()

def parse_vlm_response(raw_text: str) -> dict:
    """VLM yanıtını parse eder — multi-query veya tek description."""
    if not raw_text or not raw_text.strip():
        return _empty_response()

    raw_text = re.sub(r'\[img-\d+\]', '', raw_text)
    raw_text = re.sub(r'<思考>.*?</思考>', '', raw_text, flags=re.DOTALL)
    raw_text = re.sub(r'<thinking>.*?</thinking>', '', raw_text, flags=re.DOTALL)
    raw_text = raw_text.replace('**', '').replace('*', '')

    json_result = _parse_json_response(raw_text)
    if json_result.get("queries"):
        return json_result

    if raw_text.strip():
        return {"queries": [raw_text.strip()], "description": raw_text.strip()}

    return _empty_response()

# ============================================================
#   Online (OpenRouter) Batch
# ============================================================

def _image_to_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

async def _call_openrouter_async(session, image_path: str, retries: int = 3) -> dict:
    if not OPENROUTER_API_KEY:
        print("   ❌ HATA: OPENROUTER_API_KEY bulunamadı!")
        return _empty_response()

    img_b64 = _image_to_b64(image_path)
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = f"image/{'jpeg' if ext in ('jpg','jpeg') else ext}"

    payload = {
        "model": MODEL,
        "reasoning": {"effort": "high"},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": SIMPLE_DESIGNER_PROMPT}
            ]
        }],
        "temperature": 0.7,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "IOatmos",
    }

    for attempt in range(retries):
        try:
            async with session.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60) as resp:
                if resp.status == 401:
                    print(f"   ❌ YETKİ HATASI (401): API anahtarı geçersiz")
                    return _empty_response()
                if resp.status != 200:
                    err_text = await resp.text()
                    print(f"   ⚠️ API Hatası ({resp.status}): {err_text[:200]}")
                    continue

                data = await resp.json()
                if "choices" not in data:
                    print(f"   ⚠️ Yanıt formatı hatalı")
                    continue

                raw = data["choices"][0]["message"]["content"].strip()
                return _parse_json_response(raw)
        except Exception as e:
            if attempt == retries - 1:
                print(f"   ❌ Sahne {image_path} hatası: {str(e)}")
            await asyncio.sleep(2)

    return _empty_response()

async def _analyze_online_batch(scene_frame_paths: list[dict], max_workers: int = 16) -> list[dict]:
    if OPENROUTER_API_KEY:
        masked = OPENROUTER_API_KEY[:10] + "..." + OPENROUTER_API_KEY[-4:]
        print(f"   🔑 API Anahtarı aktif: {masked}")

    connector = aiohttp.TCPConnector(limit=max_workers)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_call_openrouter_async(session, item["frame_path"]) for item in scene_frame_paths]
        print(f"   ⏳ {len(tasks)} istek gönderildi...")
        responses = await asyncio.gather(*tasks)

        results = []
        for item, parsed in zip(scene_frame_paths, responses):
            results.append({**item, **parsed})
            desc = parsed.get("description", "")
            if desc:
                print(f"   ✅ Scene {item['scene_id']:02d}: {desc[:80]}")
            else:
                print(f"   ⚠️ Scene {item['scene_id']:02d} BOŞ YANIT")

        return results

# ============================================================
#   Local (llama.cpp Server)
# ============================================================

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
            "temperature": 0.3, "max_tokens": 1000,
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

def _analyze_local_batch(scene_frame_paths: list[dict], base_model: str, mmproj: str, max_workers: int = 8) -> list[dict]:
    """Local VLM ile batch sahne analizi (llama-server)."""
    proc = start_llama_server(base_model, mmproj)
    if not proc:
        print("   ⚠️ Local sunucu başlatılamadı, CLI fallback kullanılıyor...")
        results = []
        for item in scene_frame_paths:
            res = analyze_with_llamacpp_cli(item["frame_path"], base_model, mmproj)
            results.append({**item, **res})
            desc = res.get("description", "")
            if desc:
                print(f"   ✅ Scene {item['scene_id']:02d}: {desc[:80]}")
            else:
                print(f"   ⚠️ Scene {item['scene_id']:02d} BOŞ YANIT")
        return results

    try:
        results = []
        for item in scene_frame_paths:
            res = query_llama_server(item["frame_path"])
            results.append({**item, **res})
            desc = res.get("description", "")
            if desc:
                print(f"   ✅ Scene {item['scene_id']:02d}: {desc[:80]}")
            else:
                print(f"   ⚠️ Scene {item['scene_id']:02d} BOŞ YANIT")
        return results
    finally:
        stop_llama_server(proc)

# ============================================================
#   Ana API
# ============================================================

def analyze_scenes_batch(scene_frame_paths: list[dict], mode: str = "online", **kwargs) -> list[dict]:
    if mode == "online":
        return asyncio.run(_analyze_online_batch(scene_frame_paths))
    elif mode == "local":
        base_model = kwargs.get("base_model")
        mmproj = kwargs.get("mmproj")
        if not base_model or not mmproj:
            print("   ❌ Local mod için base_model ve mmproj gerekli!")
            results = []
            for item in scene_frame_paths:
                results.append({**item, **_empty_response()})
            return results
        return _analyze_local_batch(scene_frame_paths, base_model, mmproj)
    else:
        results = []
        for item in scene_frame_paths:
            results.append({**item, **_empty_response()})
        return results
