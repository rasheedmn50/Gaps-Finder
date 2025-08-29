#!/usr/bin/env python3
# Streamlit Paper Gap Analyzer — High-Rigor Mode + Narrative/Executive Reports
# -----------------------------------------------------------------------------
# • Upload up to 50 PDFs/DOCX/TXT and analyze shared research gaps.
# • High-Rigor mode adds: evidence quotes + page markers + section, section-aware prompts,
#   HDBSCAN clustering + re-ranking, reviewer approval UI, domain checklist tags.
# • Produces DOCX downloads: Narrative (paragraphs only) or Executive Brief (very short).
#
# Local run:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Streamlit Cloud:
#   Set secrets -> OPENAI_API_KEY = "sk-..."
#
from __future__ import annotations

import io
import json
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import streamlit as st

# Optional deps (handled in requirements)
try:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadWarning
    warnings.filterwarnings("ignore", category=PdfReadWarning)
except Exception:
    PdfReader = None

try:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except Exception:
    Document = None

try:
    from sklearn.preprocessing import normalize as sk_normalize
except Exception:
    sk_normalize = None

try:
    import hdbscan  # optional
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

from openai import OpenAI

# -------- Streamlit UI config --------
st.set_page_config(page_title="Paper Gap Analyzer", page_icon="🧠", layout="wide")

DEFAULT_MODEL = "gpt-5"
DEFAULT_EMBEDDING = "text-embedding-3-large"
MAX_PAPERS = 50
MAX_CHARS_PER_PAPER = 80_000

# -------- Sidebar --------
st.sidebar.title("⚙️ Settings")
api_key_input = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="sk-…")
api_key = api_key_input or st.secrets.get("OPENAI_API_KEY", "")

model = st.sidebar.text_input("Chat Model", value=DEFAULT_MODEL)
embedding_model = st.sidebar.text_input("Embedding Model", value=DEFAULT_EMBEDDING)
limit_chars = st.sidebar.number_input("Max chars per paper", 20000, 180000, MAX_CHARS_PER_PAPER, 2000)

st.sidebar.markdown("---")
rigor = st.sidebar.toggle("High-Rigor mode", value=True,
                          help="Adds evidence quotes, sections, HDBSCAN clustering, reviewer approval, and checklist tags.")
sim_threshold = st.sidebar.slider("Similarity threshold (fallback clustering)", 0.70, 0.90, 0.82, 0.01)

# Checklist tags (health/ML focused)
CHECKLIST = [
    "ExternalValidation", "ProspectiveValidation", "Calibration", "Fairness/Equity",
    "Reproducibility", "ReportingCompleteness", "Safety/ClinicalRisk", "DataShift/Generalizability",
    "ComparatorAdequacy", "Confounding/Bias", "Ethics/Governance", "Privacy"
]

st.title("🧠 Paper Gap Analyzer")
st.caption("Upload up to 50 papers (PDF/DOCX/TXT). Find shared gaps and download a DOCX report.")

uploads = st.file_uploader("Upload your papers", type=["pdf", "docx", "txt"], accept_multiple_files=True)
if uploads and len(uploads) > MAX_PAPERS:
    st.warning(f"Only the first {MAX_PAPERS} files will be analyzed.")
    uploads = uploads[:MAX_PAPERS]

c1, c2, c3, c4 = st.columns([1,1,1,2])
with c1:
    run_btn = st.button("Analyze", type="primary", use_container_width=True)
with c2:
    clear_btn = st.button("Clear", use_container_width=True)
with c3:
    brief_mode = st.toggle("Executive Brief", value=False, help="Very short 1-page DOCX")
with c4:
    st.write("")  # spacer

if clear_btn:
    st.session_state.clear()
    st.experimental_rerun()

# -------- Helpers --------
def build_client(key: str) -> OpenAI:
    if not key or not key.startswith("sk-"):
        raise RuntimeError("Please provide a valid OpenAI API key.")
    return OpenAI(api_key=key)

def cleanup_json_str(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(json)?\s*|```$", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", s)
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        s = m.group(0)
    s = re.sub(r",\s*(\])", r"\1", s)
    s = re.sub(r",\s*(\})", r"\1", s)
    return s

def parse_possible_json(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except Exception:
        cleaned = cleanup_json_str(content)
        return json.loads(cleaned)

def safe_chat(client: OpenAI, model: str, messages: List[Dict[str, str]], *,
              use_schema: bool, schema: Optional[Dict[str, Any]] = None, retries: int = 4) -> str:
    last_err = None
    allow_plain = False
    for attempt in range(1, retries + 1):
        try:
            msgs = [dict(m) for m in messages]
            if use_schema and not allow_plain and schema:
                resp = client.chat.completions.create(
                    model=model, messages=msgs,
                    response_format={"type": "json_schema", "json_schema": schema},
                )
            else:
                if msgs and "Return ONLY valid JSON" not in msgs[-1]["content"]:
                    msgs[-1]["content"] += "\n\nReturn ONLY valid, minified JSON that matches the expected fields."
                resp = client.chat.completions.create(model=model, messages=msgs)
            return resp.choices[0].message.content
        except Exception as e:
            err = str(e).lower()
            last_err = e
            if "unsupported" in err or "response_format" in err or "temperature" in err:
                allow_plain = True
            time.sleep(1.2 * attempt)
    raise RuntimeError(f"LLM request failed after retries: {last_err}")

# -------- Reading files (Streams) --------
def read_txt_stream(upload) -> str:
    return upload.read().decode("utf-8", errors="ignore")

def read_docx_stream(file_bytes: bytes) -> str:
    if Document is None:
        raise RuntimeError("python-docx not installed")
    d = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in d.paragraphs)

def read_pdf_pages(file_bytes: bytes) -> List[str]:
    """Return a list of page texts (empty string if extraction fails for a page)."""
    if PdfReader is None:
        raise RuntimeError("pypdf not installed")
    bio = io.BytesIO(file_bytes)
    try:
        try:
            reader = PdfReader(bio, strict=False)
        except TypeError:
            reader = PdfReader(bio)
        pages = []
        for page in getattr(reader, "pages", []) or []:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return pages
    except Exception as e:
        raise RuntimeError(f"PDF parse failed: {e}")

def extract_text_with_pages(upload) -> Tuple[str, List[str]]:
    """Return (full_text, page_texts). For non-PDF, page_texts is a single 'page'."""
    name = upload.name.lower()
    if name.endswith(".txt"):
        t = read_txt_stream(upload)
        return t, [t]
    data = upload.read()
    if name.endswith(".docx"):
        t = read_docx_stream(data)
        return t, [t]
    if name.endswith(".pdf"):
        pages = read_pdf_pages(data)
        # Insert markers so the LLM can point to page numbers.
        marked = []
        for i, p in enumerate(pages, 1):
            marked.append(f"[[PAGE {i}]]\n{p}")
        return "\n\n".join(marked), pages
    raise ValueError("Unsupported file type")

# -------- Sectioning (heuristic) --------
SECTION_HEADERS = [
    r"\babstract\b", r"\bbackground\b", r"\bintroduction\b", r"\bmethods?\b",
    r"\bmaterials and methods\b", r"\bresults\b", r"\bdiscussion\b",
    r"\blimitations?\b", r"\bconclusion(s)?\b"
]

def split_into_sections(text: str) -> Dict[str, str]:
    """Very simple regex-based splitter; robust to OCR noise but imperfect."""
    idxs = []
    for m in re.finditer("|".join(SECTION_HEADERS), text, flags=re.IGNORECASE):
        idxs.append((m.start(), m.group(0).lower()))
    if not idxs:
        return {"FullText": text}
    idxs.sort()
    sections: Dict[str, str] = {}
    for i, (pos, hdr) in enumerate(idxs):
        end = idxs[i+1][0] if i+1 < len(idxs) else len(text)
        key = hdr.strip().title()
        sections[key] = text[pos:end].strip()
    return sections

# -------- Schemas --------
def paper_schema_simple(name="paper_gap") -> Dict[str, Any]:
    return {
        "name": name, "strict": True,
        "schema": {
            "type": "object", "additionalProperties": True,
            "properties": {
                "paper_id": {"type": "string"},
                "title": {"type": "string"},
                "domain": {"type": "string"},
                "topic_keywords": {"type": "array", "items": {"type": "string"}},
                "methodology": {"type": "string"},
                "reported_limitations": {"type": "array", "items": {"type": ["string","object"]}},
                "reported_future_work": {"type": "array", "items": {"type": ["string","object"]}},
                "inferred_gaps": {"type": "array", "items": {"type": ["string","object"]}}
            },
            "required": ["paper_id","title","domain","topic_keywords","methodology",
                         "reported_limitations","reported_future_work","inferred_gaps"]
        }
    }

def paper_schema_rigorous(name="paper_gap_v2") -> Dict[str, Any]:
    return {
        "name": name, "strict": True,
        "schema": {
            "type": "object", "additionalProperties": True,
            "properties": {
                "paper_id": {"type": "string"},
                "title": {"type": "string"},
                "domain": {"type": "string"},
                "topic_keywords": {"type": "array", "items": {"type": "string"}},
                "methodology": {"type": "string"},
                "reported_limitations": {
                    "type": "array",
                    "items": {
                        "type": "object", "additionalProperties": True,
                        "properties": {
                            "description": {"type": "string"},
                            "section": {"type": "string"},
                            "evidence_quote": {"type": "string"},
                            "evidence_page": {"type": ["integer","array","string","null"]},
                            "confidence": {"type": ["number","integer"]},
                        },
                        "required": ["description","section","evidence_quote"]
                    }
                },
                "reported_future_work": {"type": "array", "items": {"type": ["string","object"]}},
                "inferred_gaps": {
                    "type": "array",
                    "items": {
                        "type": "object", "additionalProperties": True,
                        "properties": {
                            "description": {"type": "string"},
                            "category": {"type": "string"},
                            "severity": {"type": ["integer","number"]},
                            "confidence": {"type": ["number","integer"]},
                            "evidence_quote": {"type": ["string","null"]},
                            "evidence_page": {"type": ["integer","array","string","null"]},
                        },
                        "required": ["description","category","severity","confidence"]
                    }
                }
            },
            "required": ["paper_id","title","domain","topic_keywords","methodology",
                         "reported_limitations","reported_future_work","inferred_gaps"]
        }
    }

# -------- Normalization --------
def norm_list_str(x):
    if not isinstance(x, list): return []
    return [str(t) for t in x if isinstance(t,(str,int,float))]

def normalize_simple(data: Dict[str,Any], paper_id: str) -> Dict[str,Any]:
    out = {
        "paper_id": str(data.get("paper_id", paper_id)),
        "title": str(data.get("title", paper_id)),
        "domain": str(data.get("domain", "")),
        "topic_keywords": norm_list_str(data.get("topic_keywords") or []),
        "methodology": str(data.get("methodology", "")),
        "reported_limitations": [],
        "reported_future_work": [],
        "inferred_gaps": []
    }
    for k in ("reported_limitations","reported_future_work"):
        out[k] = norm_list_str(data.get(k) or [])
        if not out[k] and isinstance(data.get(k), list):
            # try dicts with text/description
            tmp=[]
            for d in data[k]:
                if isinstance(d, dict):
                    s = str(d.get("text") or d.get("description") or "").strip()
                    if s: tmp.append(s)
            out[k]=tmp
    gaps = data.get("inferred_gaps") or []
    if isinstance(gaps, list):
        for g in gaps:
            if isinstance(g, dict):
                desc = str(g.get("description") or g.get("text") or "").strip()
                if not desc: continue
                try: sev=int(round(float(g.get("severity",3))))
                except: sev=3
                out["inferred_gaps"].append({
                    "description": desc, "category": str(g.get("category") or "Other"),
                    "severity": min(max(sev,1),5),
                    "confidence": float(g.get("confidence",0.5)) if isinstance(g.get("confidence",0.5),(int,float,str)) else 0.5,
                    "evidence_quote": g.get("evidence_quote") if isinstance(g.get("evidence_quote"), str) else None,
                    "evidence_page": g.get("evidence_page"),
                })
            elif isinstance(g,(str,int,float)):
                s=str(g).strip()
                if s: out["inferred_gaps"].append({
                    "description": s, "category":"Other", "severity":3, "confidence":0.5,
                    "evidence_quote": None, "evidence_page": None
                })
    return out

def normalize_rigorous(data: Dict[str,Any], paper_id: str) -> Dict[str,Any]:
    out = normalize_simple(data, paper_id)
    # reported_limitations: force objects -> strings while keeping evidence separately
    rep = []
    for item in data.get("reported_limitations") or []:
        if isinstance(item, dict):
            desc = str(item.get("description") or item.get("text") or "").strip()
            if not desc: continue
            rep.append({
                "description": desc,
                "section": str(item.get("section") or ""),
                "evidence_quote": str(item.get("evidence_quote") or "")[:400],
                "evidence_page": item.get("evidence_page"),
                "confidence": float(item.get("confidence", 0.8)) if isinstance(item.get("confidence",0.8),(int,float,str)) else 0.8
            })
        elif isinstance(item,(str,int,float)):
            rep.append({
                "description": str(item).strip(), "section":"", "evidence_quote":"", "evidence_page": None, "confidence":0.6
            })
    out["reported_limitations"] = rep
    return out

# -------- LLM analysis --------
def analyze_paper_simple(client: OpenAI, model: str, paper_id: str, text: str) -> Dict[str,Any]:
    if len(text) > MAX_CHARS_PER_PAPER: text = text[:MAX_CHARS_PER_PAPER]
    system = ("You are a meticulous research-methods auditor.\n"
              "Extract concise structured metadata, reported limitations, and inferred gaps grounded in the paper text.")
    user = (f"PAPER_ID: {paper_id}\n\nReturn only JSON matching the schema (no prose).\n\n"
            f"--- PAPER FULL TEXT START ---\n{text}\n--- PAPER FULL TEXT END ---")
    content = safe_chat(client, model,
                        [{"role":"system","content":system},{"role":"user","content":user}],
                        use_schema=True, schema=paper_schema_simple())
    return normalize_simple(parse_possible_json(content), paper_id)

def analyze_paper_rigorous(client: OpenAI, model: str, paper_id: str, sectioned_text: Dict[str,str]) -> Dict[str,Any]:
    """
    sectioned_text: keys like 'Abstract','Methods','Results','Discussion','Limitations','FullText' etc.
    We also expect page markers like [[PAGE 3]] to appear inside values when available (from PDFs).
    """
    # Keep prompt compact: only include present sections
    parts = []
    for k, v in sectioned_text.items():
        if not v or not v.strip(): continue
        # truncate each section to control cost
        vv = v.strip()
        if len(vv) > int(MAX_CHARS_PER_PAPER/len(sectioned_text)):  # coarse budget per section
            vv = vv[: int(MAX_CHARS_PER_PAPER/len(sectioned_text))]
        parts.append(f"=== {k} ===\n{vv}\n")
    joined = "\n".join(parts)

    system = (
        "You are an evidence-first research auditor. Use quotes and page markers when available.\n"
        "If page markers like [[PAGE 5]] are present near the quoted text, include that page number in 'evidence_page'.\n"
        "For inferred gaps, keep confidence low unless supported by a quote."
    )
    user = (
        "Return ONLY JSON matching the schema. For each REPORTED limitation: include 'description', 'section', "
        "'evidence_quote' (<=60 words), and 'evidence_page' if you can infer it from [[PAGE N]].\n"
        "For each INFERRED gap: include 'description','category','severity (1-5)','confidence', and optional 'evidence_quote'/'evidence_page'.\n\n"
        + joined
    )

    content = safe_chat(client, model,
                        [{"role":"system","content":system},{"role":"user","content":user}],
                        use_schema=True, schema=paper_schema_rigorous())
    return normalize_rigorous(parse_possible_json(content), paper_id)

# -------- Embeddings / clustering --------
def embed(client: OpenAI, embedding_model: str, texts: List[str]) -> np.ndarray:
    out=[]
    if not texts: return np.zeros((0, 1536), dtype=np.float32)
    B=64
    for i in range(0,len(texts),B):
        batch=texts[i:i+B]
        emb=client.embeddings.create(model=embedding_model, input=batch)
        vecs=[d.embedding for d in emb.data]
        out.append(np.array(vecs, dtype=np.float32))
    return np.vstack(out)

def greedy_cluster(vectors: np.ndarray, threshold: float) -> List[List[int]]:
    n=len(vectors); unused=set(range(n)); clusters=[]
    while unused:
        seed=unused.pop(); members=[seed]
        centroid=vectors[seed].copy()
        changed=True
        while changed:
            changed=False; to_add=[]
            for j in list(unused):
                sim=float(np.dot(centroid, vectors[j])/(np.linalg.norm(centroid)*np.linalg.norm(vectors[j])+1e-9))
                if sim>=threshold: to_add.append(j)
            if to_add:
                for j in to_add: unused.remove(j); members.append(j)
                centroid=vectors[members].mean(axis=0); changed=True
        clusters.append(members)
    return clusters

def hdbscan_cluster(vectors: np.ndarray) -> List[List[int]]:
    if not HDBSCAN_AVAILABLE or sk_normalize is None:
        return greedy_cluster(vectors, threshold=0.82)
    X = sk_normalize(vectors)  # L2 normalize
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=1, metric='euclidean')
    labels = clusterer.fit_predict(X)
    clusters: Dict[int,List[int]] = {}
    for idx, lab in enumerate(labels):
        if lab == -1:  # noise -> its own singleton cluster
            clusters[f"noise-{idx}"] = [idx]
        else:
            clusters.setdefault(lab, []).append(idx)
    return list(clusters.values())

# -------- Cluster stats & re-ranking --------
def cluster_stats_from_members(members: List[int], items: List[Dict[str,Any]]) -> Dict[str,Any]:
    subset=[items[i] for i in members]
    uniq_papers=sorted({x["paper_id"] for x in subset})
    severities=[x.get("severity",3) for x in subset]
    srcs=[x.get("source","Reported") for x in subset]
    evidence=[1 if (x.get("evidence_quote")) else 0 for x in subset]
    return {
        "paper_coverage": len(uniq_papers),
        "unique_papers": uniq_papers,
        "count": len(subset),
        "avg_severity": float(np.mean(severities) if severities else 0),
        "median_severity": float(np.median(severities) if severities else 0),
        "reported_share": float(srcs.count("Reported")/len(srcs)) if srcs else 0.0,
        "inferred_share": float(srcs.count("Inferred")/len(srcs)) if srcs else 0.0,
        "evidence_strength": float(np.mean(evidence) if evidence else 0.0),
        "samples": [x["text"] for x in subset[:5]],
        "members_idx": members
    }

def rank_clusters(stats: List[Dict[str,Any]], total_papers: int, rigorous: bool) -> List[int]:
    scores=[]
    for i, c in enumerate(stats):
        coverage = c["paper_coverage"]/max(1,total_papers)
        sev = c["median_severity"]/5.0
        ev = c.get("evidence_strength",0.0)
        if rigorous:
            score = 0.5*coverage + 0.3*sev + 0.2*ev
        else:
            score = 0.7*coverage + 0.3*sev
        scores.append((score,i))
    scores.sort(reverse=True)
    return [i for _,i in scores]

# -------- Checklist tagging (LLM) --------
def llm_tag_gaps(client: OpenAI, model: str, gap_texts: List[str]) -> List[List[str]]:
    if not gap_texts: return [[] for _ in gap_texts]
    system = "You assign concise checklist tags to research gaps. Return ONLY JSON."
    user = ("For each gap, choose zero or more tags from this list:\n" +
            ", ".join(CHECKLIST) +
            "\nReturn JSON with {'tags': [['Tag1','Tag2',...], ...]} in the same order as inputs.\n\n" +
            json.dumps({"gaps": gap_texts}, ensure_ascii=False))
    content = safe_chat(
        client, model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        use_schema=False
    )
    try:
        data = parse_possible_json(content)
        return data.get("tags", [[] for _ in gap_texts])
    except Exception:
        return [[] for _ in gap_texts]

# -------- Narrative / Executive LLM --------
def llm_narrative(client: OpenAI, model: str, clusters_payload: Dict[str,Any]) -> str:
    system = ("You write clear, non-technical academic summaries.\n"
              "Return plain text only: paragraphs separated by a blank line. No headings or bullets.")
    user = (
        "Write 3–5 short paragraphs narrating the main shared gaps (one per paragraph), "
        "then ONE final paragraph recommending the single gap with the best potential, with a brief rationale and a one-sentence next step.\n"
        "Rules: NO lists, NO bullets, NO headings, NO tables, NO explicit numbers/percentages—just prose. <= 600 words.\n\n"
        "JSON:\n" + json.dumps(clusters_payload, ensure_ascii=False)
    )
    content = safe_chat(client, model, [{"role":"system","content":system},{"role":"user","content":user}],
                        use_schema=False)
    return re.sub(r"`{3,}.*?`{3,}", "", content, flags=re.DOTALL).strip()

def llm_executive_brief(client: OpenAI, model: str, clusters_payload: Dict[str,Any]) -> str:
    system = "You write ultra-concise executive briefs for researchers."
    user = (
        "In <= 220 words total, write:\n"
        "• Two short paragraphs that plainly describe the main gaps (no lists, no numbers).\n"
        "• One final 2–3 sentence recommendation naming the best gap to pursue and a concrete starting step.\n"
        "No headings, no bullets—just three compact paragraphs.\n\n"
        "JSON:\n" + json.dumps(clusters_payload, ensure_ascii=False)
    )
    content = safe_chat(client, model, [{"role":"system","content":system},{"role":"user","content":user}],
                        use_schema=False)
    return re.sub(r"`{3,}.*?`{3,}", "", content, flags=re.DOTALL).strip()

# -------- DOCX builders --------
def docx_from_paragraphs(title: str, paragraphs: List[str]) -> bytes:
    if Document is None: return b""
    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Inches(0.8); sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.8); sec.right_margin = Inches(0.8)
    h = doc.add_paragraph(title); h.style = "Title"; h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    from datetime import datetime
    sub = doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    for ptxt in [p.strip() for p in paragraphs if p.strip()]:
        p = doc.add_paragraph(ptxt)
        pf = p.paragraph_format; pf.space_after = Pt(6); pf.line_spacing = 1.2
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    bio = io.BytesIO(); doc.save(bio); return bio.getvalue()

# ================== MAIN ACTION ==================
if run_btn:
    if not api_key:
        st.error("Please enter your OpenAI API key (sidebar) or set it in Secrets.")
        st.stop()
    if not uploads:
        st.warning("Please upload at least one file.")
        st.stop()

    client = build_client(api_key)

    # Step 1 — per-paper analysis
    st.subheader("Step 1 — Analyzing papers")
    per_paper: List[Dict[str,Any]] = []
    progress = st.progress(0.0); status = st.empty()

    for i, up in enumerate(uploads, 1):
        status.info(f"Reading: {up.name}")
        try:
            full_text, pages = extract_text_with_pages(up)
        except Exception as e:
            st.warning(f"Skipping {up.name}: read error — {e}")
            progress.progress(i/len(uploads)); continue

        if len(full_text.strip()) < 500:
            st.warning(f"Skipping {up.name}: very little text (consider OCR).")
            progress.progress(i/len(uploads)); continue

        status.info(f"LLM analysis: {up.name}")
        try:
            if rigor:
                # Section-aware with page markers
                sections = split_into_sections(full_text)
                analysis = analyze_paper_rigorous(client, model, up.name, sections)
            else:
                analysis = analyze_paper_simple(client, model, up.name, full_text[:limit_chars])
        except Exception as e:
            st.warning(f"LLM analysis failed for {up.name}: {e}")
            progress.progress(i/len(uploads)); continue

        per_paper.append(analysis)
        progress.progress(i/len(uploads))

    status.empty()
    if not per_paper:
        st.error("No papers analyzed successfully."); st.stop()
    st.success(f"Analyzed {len(per_paper)} papers.")

    # Step 2 — build raw gap list
    st.subheader("Step 2 — Review candidate gaps (accept / edit)")
    raw_items=[]
    for a in per_paper:
        pid=a.get("paper_id") or "unknown"

        # Reported limitations → gaps with evidence
        rep = a.get("reported_limitations", [])
        for item in rep:
            if isinstance(item, dict):
                txt = (item.get("description") or "").strip()
                if not txt: continue
                raw_items.append({
                    "paper_id": pid,
                    "text": txt,
                    "source": "Reported",
                    "severity": 3,
                    "section": item.get("section",""),
                    "evidence_quote": item.get("evidence_quote",""),
                    "evidence_page": item.get("evidence_page", None),
                    "tags": []
                })
            elif isinstance(item, (str,int,float)):
                t=str(item).strip()
                if t:
                    raw_items.append({"paper_id": pid, "text": t, "source":"Reported",
                                      "severity":3, "section":"", "evidence_quote":"", "evidence_page":None, "tags":[]})

        # Inferred gaps
        for g in a.get("inferred_gaps", []):
            if isinstance(g, dict):
                desc = (g.get("description") or "").strip()
                if not desc: continue
                sev = g.get("severity",3)
                try: sev = int(round(float(sev)))
                except: sev=3
                raw_items.append({
                    "paper_id": pid,
                    "text": desc,
                    "source": "Inferred",
                    "severity": min(max(sev,1),5),
                    "section": "",
                    "evidence_quote": g.get("evidence_quote",""),
                    "evidence_page": g.get("evidence_page", None),
                    "tags": []
                })
            elif isinstance(g,(str,int,float)):
                t=str(g).strip()
                if t:
                    raw_items.append({"paper_id": pid, "text": t, "source":"Inferred",
                                      "severity":3, "section":"", "evidence_quote":"", "evidence_page":None, "tags":[]})

    if not raw_items:
        st.warning("No gap statements found."); st.stop()

    # Optional: tag with checklist (helps later re-ranking/filters)
    if rigor:
        with st.spinner("Tagging gaps with checklist categories…"):
            tag_lists = llm_tag_gaps(client, model, [x["text"] for x in raw_items])
            for x, tags in zip(raw_items, tag_lists):
                x["tags"] = [t for t in tags if t in CHECKLIST][:4]

    # Reviewer UI (data editor)
    import pandas as pd
    df = pd.DataFrame([{
        "Accept": True if (item["source"]=="Reported" or item["evidence_quote"]) else False,
        "Text": item["text"],
        "Source": item["source"],
        "Severity": item["severity"],
        "Section": item.get("section",""),
        "EvidenceQuote": item.get("evidence_quote",""),
        "EvidencePage": item.get("evidence_page",""),
        "Tags": ", ".join(item.get("tags",[])),
        "PaperID": item["paper_id"],
    } for item in raw_items])

    st.caption("Tip: uncheck items you don't want clustered; edit text freely before clustering.")
    edited = st.data_editor(df, height=360, use_container_width=True,
                            column_config={
                                "Accept": st.column_config.CheckboxColumn(),
                                "Text": st.column_config.TextColumn(width="large"),
                                "Severity": st.column_config.NumberColumn(min_value=1, max_value=5, step=1),
                            })

    proceed = st.button("Cluster accepted gaps", type="primary")
    if not proceed:
        st.stop()

    # Apply reviewer choices
    accepted_items=[]
    for row, base in zip(edited.to_dict("records"), raw_items):
        if not row["Accept"]: continue
        item = base.copy()
        item["text"] = row["Text"].strip()
        item["severity"] = int(row["Severity"])
        item["section"] = row.get("Section","")
        item["evidence_quote"] = row.get("EvidenceQuote","")
        item["evidence_page"] = row.get("EvidencePage","")
        item["tags"] = [t.strip() for t in (row.get("Tags","") or "").split(",") if t.strip()]
        accepted_items.append(item)

    if not accepted_items:
        st.error("No gaps selected."); st.stop()

    # Step 3 — embeddings + clustering
    st.subheader("Step 3 — Clustering shared gaps")
    texts = [g["text"] for g in accepted_items]
    vecs = embed(client, embedding_model, texts)

    if rigor:
        clusters_members = hdbscan_cluster(vecs)
    else:
        clusters_members = greedy_cluster(vecs, threshold=sim_threshold)

    # Compute stats and re-rank
    stats=[]
    for mem in clusters_members:
        stats.append(cluster_stats_from_members(mem, accepted_items))

    order = rank_clusters(stats, total_papers=len(per_paper), rigorous=rigor)
    top_idx = order[0] if order else 0

    # Build payload for reporting
    clusters_payload = {
        "total_papers": len(per_paper),
        "clusters": []
    }
    for rank_pos, ci in enumerate(order):
        c = stats[ci]
        clusters_payload["clusters"].append({
            "index": ci,
            "label": f"Cluster {ci}",
            "paper_coverage": c["paper_coverage"],
            "count": c["count"],
            "avg_severity": round(c["avg_severity"],2),
            "median_severity": round(c["median_severity"],2),
            "reported_share": round(c["reported_share"],3),
            "inferred_share": round(c["inferred_share"],3),
            "evidence_strength": round(c["evidence_strength"],3),
            "example_statements": c["samples"],
            "is_recommended": (ci == top_idx),
        })

    st.success(f"Formed {len(stats)} clusters. Recommended: **Cluster {top_idx}**")

    with st.expander("Preview: top cluster examples"):
        st.write(stats[order[0]]["samples"] if order else [])

    # Step 4 — report generation
    st.subheader("Step 4 — Generate report")
    try:
        if brief_mode:
            text = llm_executive_brief(client, model, clusters_payload)
            title = "Executive Brief — Research Gaps"
        else:
            text = llm_narrative(client, model, clusters_payload)
            title = "Research Gaps — Narrative Summary"
    except Exception as e:
        st.error(f"Report generation failed: {e}")
        st.stop()

    st.text_area("Report preview", value=text, height=260)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    docx_bytes = docx_from_paragraphs(title, paragraphs)

    if docx_bytes:
        st.download_button(
            "⬇️ Download DOCX",
            data=docx_bytes,
            file_name=("executive_brief.docx" if brief_mode else "final_report_narrative.docx"),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    else:
        st.warning("python-docx not installed; cannot build DOCX.")
