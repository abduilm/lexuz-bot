# app.py — Local RAG with clean answer + lex.uz links at the end
from dotenv import load_dotenv
load_dotenv()

import os, json, re
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI

# ---------- env ----------
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL  = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")  # must match my index build
INDEX_DIR          = os.getenv("INDEX_DIR", "./index_store2")
TOP_K              = int(os.getenv("TOP_K", "8"))       # 12 edi
MIN_SIM            = float(os.getenv("MIN_SIM", "0.22")) # 0.18 edi
MAX_CTX_CHARS      = int(os.getenv("MAX_CTX_CHARS", "700"))
ESCALATE_SIM       = float(os.getenv("ESCALATE_SIM", "0.30")) # redundant?
CHAT_FALLBACK      = os.getenv("OPENAI_CHAT_MODEL_FALLBACK", "gpt-4o") # redundant?

if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY not found. Put it in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- load index ----------
IDX = Path(INDEX_DIR)
E_PATH = IDX / "embeddings.npy"
M_PATH = IDX / "meta.jsonl"

if not E_PATH.exists() or not M_PATH.exists():
    raise SystemExit(f"Index not found in {INDEX_DIR}. Build it with index_dataset.py first.")

EMB = np.load(E_PATH)
METAS: List[Dict] = [json.loads(x) for x in M_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]

if EMB.size == 0 or not METAS:
    raise SystemExit("Your index is empty. Rebuild with more docs/chunks.")

def _l2n(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        n = np.linalg.norm(X)
        return X / n if n > 0 else X
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

EMB = _l2n(EMB).astype(np.float32)

# ---------- helpers ----------
EDU_KEYWORDS = [
    "ta'lim","ta’lim","oliy ta'lim","maktab","kollej","litsey","universitet","abituriyent",
    "davlat ta'lim standarti","akkreditatsiya","o'quv reja","qabul","grant","stipendiya",
    "talaba","o‘quvchi","o’quvchi","o‘qituvchi","pedagog","malaka oshirish",
    "TTJ","yotoqxona","DS","DTM","my.maktab.uz","talablar","litsenziya","litsenziyalash",
    "образование","школа","колледж","лицей","университет","абитуриент","приём","аккредитация"
]

def has_edu_kw(s: str) -> bool:
    s = (s or "").lower()
    return any(k in s for k in EDU_KEYWORDS)

def emb_text(text: str) -> np.ndarray:
    r = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[text])
    v = np.array(r.data[0].embedding, dtype=np.float32)
    return _l2n(v)

def cosine_search(query: str, top_n: int) -> List[Tuple[float, Dict]]:
    qv = emb_text(query)
    sims = EMB @ qv
    order = np.argsort(-sims)
    results = []
    for idx in order[: max(top_n*4, top_n)]:
        m = METAS[idx]
        sim = float(sims[idx])
        if sim < MIN_SIM:
            continue
        boost = 0.0
        if (m.get("doc_url") or "").startswith(("https://lex.uz","http://lex.uz")):
            boost += 0.08
        if has_edu_kw(m.get("chunk_text","")) or has_edu_kw(m.get("doc_title","")):
            boost += 0.05
        if m.get("source_type") in {"parsed_lex","jsonl"}:
            boost += 0.03
        results.append((sim + boost, m))
    results.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for score, m in results:
        key = (m.get("doc_url",""), m.get("doc_title",""))
        if key in seen:
            continue
        seen.add(key)
        out.append((score, m))
        if len(out) >= top_n:
            break
    return out

def build_messages(question: str, picks: List[Dict]) -> List[Dict]:
    system = (
        "Siz O'zbekiston ta'lim tizimi bo'yicha yordamchi huquqiy konsultantsiz. "
        "Faqat berilgan parchalar matniga tayaning. "
        "Javobni 3–7 bandli aniq, amaliy nuqtalarda yozing. "
        "Hech qanday sarlavha qo'ymang (masalan, 'Manbalar', 'Manbalar bo'yicha' kabilar YO'Q). "
        "Matn ichida hech qanday havola yoki [sarlavha — URL] keltirmang. "
        "Hujjat nomi/raqami/sanasini zudlik bilan eslatish mumkin, ammo URL yozmang. "
        "Faqat javobni yozing; manbalarni men alohida ko'rsataman. O'zbek tilida yozing."
    )
    ctx = []
    for i, m in enumerate(picks, 1):
        title = m.get("doc_title") or ""
        url   = m.get("doc_url") or ""
        # text  = m.get("chunk_text") or ""
        text  = (m.get("chunk_text") or "")[:MAX_CTX_CHARS]  #  <<< limit per chunk
        ctx.append(f"### P {i}\nSarlavha: {title}\nURL: {url}\n\n{text}\n")
    user = (
        f"Savol: {question}\n\n"
        "Quyidagi parchalar mavjud (faqat ulardan foydalaning). "
        "Faqat javobni bering, manba bo'limlari yoki havolalar YO'Q:\n\n" +
        "\n\n---\n\n".join(ctx)
    )
    return [{"role":"system","content":system},{"role":"user","content":user}]

def strip_unwanted(answer: str) -> str:
    patts = [
        r"\n+manbalar bo['’`]?yicha:.*\Z",
        r"\n+manbalar\s*:.*\Z",
    ]
    out = answer
    for p in patts:
        out = re.sub(p, "", out, flags=re.IGNORECASE|re.DOTALL)
    return out.strip()

def only_lex_sources(picks: List[Dict], max_links: int = 10) -> List[Dict]:
    urls = []
    seen = set()
    for m in picks:
        u = (m.get("doc_url") or "").strip()
        if not u:
            continue
        if not u.startswith(("https://lex.uz","http://lex.uz")):
            continue
        if u in seen:
            continue
        seen.add(u)
        urls.append({"title": u, "url": u})
        if len(urls) >= max_links:
            break
    return urls

# ---------- FastAPI ----------
app = FastAPI(title="LexUZ Local RAG", version="1.1")

# NOTE: not an f-string! (avoids {{ }} escaping hell)
HTML = """<!doctype html>
<html lang="uz">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
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
  .label{margin-top:14px;color:#374151;font-weight:600}
</style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Lex.uz AI Bot</h1>
      <p class="sub">Savolingizni yozing (masalan: <i>"OTMga qabulda minimal o'tish ballari?"</i>).</p>
      <div class="row">
        <input id="q" type="text" placeholder="Savolni kiriting..." autofocus>
        <button id="send">Yuborish</button>
      </div>
      <div id="status" class="muted"></div>
      <div id="out" class="answer"></div>
      <div id="linksLabel" class="label" style="display:none;">Rasmiy manbalar (lex.uz):</div>
      <div id="srcs" class="sources"></div>
    </div>
  </div>
<script>
const q=document.getElementById('q'),btn=document.getElementById('send');
const out=document.getElementById('out'),srcs=document.getElementById('srcs'),
      statusEl=document.getElementById('status'),linksLabel=document.getElementById('linksLabel');
async function ask(){
  const question=q.value.trim(); if(!question){q.focus();return;}
  out.textContent=""; srcs.innerHTML=""; linksLabel.style.display='none';
  statusEl.textContent="Izlanmoqda (mahalliy indeks)..."; btn.disabled=true;
  try{
    const res=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:question})});
    const data=await res.json();
    if(!res.ok) throw new Error(data.detail||'Xatolik');
    out.textContent=data.answer||"Javob topilmadi.";
    if((data.sources||[]).length){
      linksLabel.style.display='block';
      srcs.innerHTML='';
      (data.sources||[]).forEach(s=>{
        const a=document.createElement('a');
        a.className='src'; a.href=s.url||'#'; a.target='_blank'; a.rel='noopener';
        a.textContent=(s.url||'');
        srcs.appendChild(a);
      });
      statusEl.textContent="";
    } else {
      statusEl.textContent="";
    }
  }catch(e){ statusEl.textContent="Xatolik: "+(e.message||e); }finally{ btn.disabled=false; }
}
btn.addEventListener('click',ask); q.addEventListener('keydown',(e)=>{ if(e.key==='Enter') ask(); });
</script>
</body></html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

class AskIn(BaseModel):
    question: str

@app.post("/ask")
def ask(inp: AskIn):
    ranked = cosine_search(inp.question, TOP_K)
    if not ranked:
        return {"answer": "Chummadim. Savolni aniqroq yozib ko'ring.", "sources": []}
    picks = [m for _, m in ranked]

    # cheap by default; escalate if best similarity is weak
    best_sim = float(ranked[0][0])
    model_to_use = OPENAI_CHAT_MODEL if best_sim >= ESCALATE_SIM else CHAT_FALLBACK

    try:
        messages = build_messages(inp.question, picks)
        resp = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            temperature=0.0,
            max_tokens=450, # 900 edi
            messages=messages
        )
        answer = (resp.choices[0].message.content or "").strip()
        answer = strip_unwanted(answer)
    except Exception as e:
        raise HTTPException(502, f"LLM xatosi: {e}")

    srcs = only_lex_sources(picks, max_links=10)
    return {"answer": answer, "sources": srcs}
