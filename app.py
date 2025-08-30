import os, re
from typing import List, Dict
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

# ----------------- Settings -----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID")
MAX_PAGES = int(os.getenv("MAX_PAGES", "4"))

if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY ni .env faylida ko'rsating")
if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    raise SystemExit("GOOGLE_API_KEY va GOOGLE_CSE_ID (cx) ni .env faylida ko'rsating")

client = OpenAI(api_key=OPENAI_API_KEY)

GOOGLE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
UA = "LexUZ-Live-RAG/0.1 (contact@example.com)"
CHAT_MODEL = "gpt-4o-mini"     
DEFAULT_K = 2                  
PER_PAGE_CHARS = 5000          

# ----------------- FastAPI -----------------
app = FastAPI(title="Lex.uz AI Bot", version="0.2")

# ---  HTML chat page (Uz UI) ---
HTML = """<!doctype html>
<html lang="uz">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lex.uz AI Bot</title>
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#f6f7f9;margin:0}
  .wrap{max-width:900px;margin:40px auto;padding:0 16px}
  .card{background:#fff;border-radius:16px;box-shadow:0 8px 24px rgba(0,0,0,.06);padding:20px}
  h1{margin:0 0 8px;font-size:28px}
  p.sub{margin:0 0 16px;color:#4b5563}
  .row{display:flex;gap:10px}
  input[type=text]{flex:1;padding:14px 16px;border:1px solid #e5e7eb;border-radius:12px;font-size:16px}
  button{padding:14px 18px;border:0;border-radius:12px;background:#16a34a;color:#fff;font-weight:600;cursor:pointer}
  button:disabled{opacity:.6;cursor:not-allowed}
  .answer{white-space:pre-wrap;line-height:1.6;margin-top:16px}
  .sources{margin-top:12px}
  .src{display:block;margin:4px 0;color:#2563eb;text-decoration:none;word-break:break-all}
  .muted{color:#6b7280;font-size:14px;margin-top:8px}
  .footer{margin-top:22px;font-size:13px;color:#6b7280}
</style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Lex.uz AI Bot</h1>
      <p class="sub">Savolingizni yozing (masalan: <i>"Mehnat kodeksiga ko‘ra ishga qabul qilish tartibi?"</i>). Bot javobni <b>lex.uz</b> manbalariga tayangan holda beradi.</p>
      <div class="row">
        <input id="q" type="text" placeholder="Savolni kiriting..." autofocus>
        <button id="send">Yuborish</button>
      </div>
      <div id="status" class="muted"></div>
      <div id="out" class="answer"></div>
      <div id="srcs" class="sources"></div>
      <!-- <div class="footer">⚠️ Ma’lumot maqsadida. Huquqiy maslahat emas.</div> -->
    </div>
  </div>
<script>
const q = document.getElementById('q');
const btn = document.getElementById('send');
const out = document.getElementById('out');
const srcs = document.getElementById('srcs');
const statusEl = document.getElementById('status');

async function ask() {
  const question = q.value.trim();
  if (!question) { q.focus(); return; }
  out.textContent = "";
  srcs.innerHTML = "";
  statusEl.textContent = "Izlanmoqda... (lex.uz dan sahifalar olinmoqda)";
  btn.disabled = true;

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question: question, k: %d})
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Xatolik');

    out.textContent = data.answer || "Javob topilmadi.";
    statusEl.textContent = "Manbalar:";
    (data.sources || []).forEach(s => {
      const a = document.createElement('a');
      a.className = 'src';
      a.href = s.url;
      a.target = '_blank';
      a.rel = 'noopener';
      a.textContent = (s.title || s.url) + " — " + s.url;
      srcs.appendChild(a);
    });
    if (!data.sources || data.sources.length === 0) {
      statusEl.textContent = "Mos manbalar topilmadi.";
    }
  } catch (e) {
    statusEl.textContent = "Xatolik: " + (e.message || e);
  } finally {
    btn.disabled = false;
  }
}
btn.addEventListener('click', ask);
q.addEventListener('keydown', (e)=>{ if (e.key==='Enter') ask(); });
</script>
</body>
</html>
""".replace("%d", str(DEFAULT_K))

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

# ----------------- API (inner) -----------------
class AskIn(BaseModel):
    question: str
    k: int = DEFAULT_K

def is_lex(url: str) -> bool:
    try:
        host = url.split("/")[2].lower()
    except Exception:
        return False
    return "lex.uz" in host

def search_lex_google(query: str, k: int) -> List[Dict]:
    """Google Custom Search orqali lex.uz bo'yicha topish."""
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": f"site:lex.uz {query}",
        "num": min(k, MAX_PAGES),
        "safe": "off",
        "hl": "uz",
    }
    r = requests.get(GOOGLE_ENDPOINT, params=params, timeout=20, headers={"User-Agent": UA})
    if r.status_code != 200:
        raise HTTPException(502, f"Google CSE xatosi: {r.text[:200]}")
    js = r.json()
    items = js.get("items", []) or []
    out = []
    for it in items:
        link = it.get("link", "")
        if is_lex(link):
            out.append({
                "name": it.get("title", "") or it.get("htmlTitle",""),
                "url": link,
                "snippet": it.get("snippet", "")
            })
    return out[:min(k, MAX_PAGES)]

def fetch_text(url: str) -> str:
    """Sahifani yuklab, asosiy matnni ajratib olish."""
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    candidates = [
        {"class_": "document-text"},
        {"id": "content"},
        {"class_": "content"},
        {"role": "main"},
        {}
    ]
    text = ""
    for sel in candidates:
        el = soup.find(**sel)
        if el:
            text = el.get_text("\n", strip=True)
            if len(text) > 800:
                break
    if not text:
        text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:PER_PAGE_CHARS]

def build_messages(question: str, pages: List[Dict[str, str]]) -> list:
    system = (
        "Siz O'zbekiston qonunchiligi bo'yicha yordamchi huquqiy konsultantsiz. "
        "Faqat berilgan lex.uz manbalaridan foydalaning. "
        "Har bir fikrda manbani shu ko'rinishda ko'rsating: [Sarlavha — URL]. "
        "Mavjud bo'lsa, hujjatning qabul qilingan sanasi, oxirgi tahriri va amal qilish holatini ko'rsating. "
        "Juda lo'nda yozing. Agar manbalarda javob aniq bo'lmasa, aynan nimasi yetishmayotganini ayting. "
        "Bu huquqiy maslahat EMAS. Javobni o'zbek tilida yozing."
    )
    ctx_parts = []
    for i, p in enumerate(pages, 1):
        body = p['text'][:PER_PAGE_CHARS]
        ctx_parts.append(
            f"### Manba {i}\nSarlavha: {p['name']}\nURL: {p['url']}\n\n{body}\n"
        )
    user = f"Savol: {question}\n\nQuyidagi manbalargina mavjud:\n\n" + "\n\n---\n\n".join(ctx_parts)
    return [{"role":"system","content":system},{"role":"user","content":user}]

@app.post("/ask")
def ask(inp: AskIn):
    # 1) Search
    results = search_lex_google(inp.question, inp.k)
    if not results:
        return {"answer": "Ushbu savol bo'yicha mos lex.uz sahifalari topilmadi.", "sources": []}

    # 2) Get pages
    pages = []
    for it in results:
        try:
            text = fetch_text(it["url"])
            pages.append({"name": it["name"], "url": it["url"], "text": text})
        except Exception:
            continue
    if not pages:
        return {"answer": "lex.uz sahifalarini yuklab bo'lmadi (tarmoq / format muammosi).", "sources": []}

    # 3) Reply with LLM
    messages = build_messages(inp.question, pages)
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.1,
            max_tokens=500,
            messages=messages
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "You exceeded your current quota" in msg:
            return {
                "answer": "LLM kvotasi tugagan. Quyidagi manbalarni bevosita o'qing:",
                "sources": [{"title": p["name"], "url": p["url"]} for p in pages]
            }
        raise HTTPException(502, f"OpenAI xatosi: {e}")

    # 4) Result
    return {
        "answer": answer,
        "sources": [{"title": p["name"], "url": p["url"]} for p in pages]
    }
