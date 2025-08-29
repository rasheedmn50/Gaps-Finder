#!/usr/bin/env python3
# Streamlit Paper Gap Analyzer — Narrative Word Report
# ----------------------------------------------------
# • Upload up to 50 PDFs/DOCX/TXT and analyze shared research gaps.
# • Robust PDF parsing (lenient pypdf, suppressed warnings).
# • LLM with retries + JSON-schema fallback, robust JSON cleanup.
# • Narrative Word (.docx) report with short paragraphs only.
# • No bullets/tables in the report; just readable prose.
#
# Setup (locally):
#   pip install -r requirements.txt
#   streamlit run app.py
#
# In Streamlit Cloud, set your secret:
#   OPENAI_API_KEY = "sk-..."
#
from __future__ import annotations

import io
import json
import re
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st

# ---------- Optional deps ----------
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

from openai import OpenAI

# ---------- App config ----------
st.set_page_config(page_title="Paper Gap Analyzer", page_icon="🧠", layout="wide")

DEFAULT_MODEL = "gpt-5"
DEFAULT_EMBEDDING = "text-embedding-3-large"
MAX_PAPERS = 50
MAX_CHARS_PER_PAPER = 80_000

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")

api_key_input = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    placeholder="sk-...",
    help="We recommend using Streamlit Secrets in the cloud. This input overrides secrets if set."
)
api_key = api_key_input or st.secrets.get("OPENAI_API_KEY", "")

model = st.sidebar.text_input("Chat Model", value=DEFAULT_MODEL, help="e.g., gpt-5")
embedding_model = st.sidebar.text_input("Embedding Model", value=DEFAULT_EMBEDDING)
sim_threshold = st.sidebar.slider("Gap similarity threshold", 0.70, 0.90, 0.82, 0.01)
limit_chars = st.sidebar.number_input("Max chars per paper", 20000, 180000, MAX_CHARS_PER_PAPER, 2000)
st.sidebar.caption("Tip: Reduce char limit if you hit model context limits.")

st.title("🧠 Paper Gap Analyzer (Streamlit)")
st.caption("Upload up to 50 papers (PDF/DOCX/TXT). The app finds shared gaps and generates a concise narrative Word report.")

# ---------- File uploader ----------
uploads = st.file_uploader(
    "Upload your papers (PDF, DOCX, or TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)
if uploads and len(uploads) > MAX_PAPERS:
    st.warning(f"Only the first {MAX_PAPERS} files will be analyzed.")
    uploads = uploads[:MAX_PAPERS]

col_a, col_b, col_c = st.columns([1,1,2])
with col_a:
    run_btn = st.button("Analyze", type="primary", use_container_width=True)
with col_b:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.session_state.clear()
    st.experimental_rerun()

# ---------- Utilities ----------
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
    s = re.sub(r",\s*(\})", r"\1", s)
    s = re.sub(r",\s*(\])", r"\1", s)
    return s

def parse_possible_json(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except Exception:
        cleaned = cleanup_json_str(content)
        return json.loads(cleaned)

def safe_chat(client: OpenAI, model: str, messages: List[Dict[str, str]], *, use_schema: bool, schema: Optional[Dict[str, Any]] = None) -> str:
    last_err = None
    allow_plain = False
    for attempt in range(1, 5):
        try:
            msgs = [dict(m) for m in messages]
            if use_schema and not allow_plain and schema:
                resp = client.chat.completions.create(
                    model=model,
                    messages=msgs,
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
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"LLM request failed after retries: {last_err}")

def paper_schema_json(name: str = "paper_gap") -> Dict[str, Any]:
    return {
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "paper_id": {"type": "string"},
                "title": {"type": "string"},
                "domain": {"type": "string"},
                "topic_keywords": {"type": "array", "items": {"type": "string"}},
                "methodology": {"type": "string"},
                "reported_limitations": {"type": "array", "items": {"type": ["string", "object"]}},
                "reported_future_work": {"type": "array", "items": {"type": ["string", "object"]}},
                "inferred_gaps": {
                    "type": "array",
                    "items": {
                        "type": ["string", "object"],
                        "additionalProperties": True,
                        "properties": {
                            "description": {"type": "string"},
                            "category": {"type": "string"},
                            "severity": {"type": ["integer", "number"]},
                            "confidence": {"type": ["number", "integer"]},
                            "evidence_quote": {"type": ["string", "null"]}
                        }
                    }
                }
            },
            "required": ["paper_id", "title", "domain", "topic_keywords", "methodology",
                         "reported_limitations", "reported_future_work", "inferred_gaps"]
        }
    }

def normalize_analysis(data: Dict[str, Any], paper_id: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "paper_id": str(data.get("paper_id", paper_id)),
        "title": str(data.get("title", paper_id)),
        "domain": str(data.get("domain", "")),
        "topic_keywords": data.get("topic_keywords") or [],
        "methodology": str(data.get("methodology", "")),
        "reported_limitations": [],
        "reported_future_work": [],
        "inferred_gaps": [],
    }
    if not isinstance(out["topic_keywords"], list):
        out["topic_keywords"] = []
    else:
        out["topic_keywords"] = [str(x) for x in out["topic_keywords"] if isinstance(x, (str, int, float))]

    for k in ("reported_limitations", "reported_future_work"):
        v = data.get(k) or []
        cleaned: List[str] = []
        if isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    s = item.strip()
                    if s:
                        cleaned.append(s)
                elif isinstance(item, dict):
                    s = str(item.get("text") or item.get("description") or "").strip()
                    if s:
                        cleaned.append(s)
        out[k] = cleaned

    gaps = data.get("inferred_gaps") or []
    if isinstance(gaps, list):
        for g in gaps:
            if isinstance(g, dict):
                desc = str(g.get("description") or g.get("text") or "").strip()
                if not desc:
                    continue
                try:
                    sev = int(round(float(g.get("severity", 3))))
                except Exception:
                    sev = 3
                sev = min(max(sev, 1), 5)
                out["inferred_gaps"].append({
                    "description": desc,
                    "category": str(g.get("category") or "Other"),
                    "severity": sev,
                    "confidence": float(g.get("confidence", 0.5)) if isinstance(g.get("confidence", 0.5), (int, float, str)) else 0.5,
                    "evidence_quote": g.get("evidence_quote") if isinstance(g.get("evidence_quote"), str) else None,
                })
            elif isinstance(g, (str, int, float)):
                s = str(g).strip()
                if s:
                    out["inferred_gaps"].append({
                        "description": s,
                        "category": "Other",
                        "severity": 3,
                        "confidence": 0.5,
                        "evidence_quote": None,
                    })
    return out

def read_txt(f) -> str:
    return f.read().decode("utf-8", errors="ignore")

def read_docx_stream(file_bytes: bytes) -> str:
    if Document is None:
        raise RuntimeError("python-docx not installed.")
    bio = io.BytesIO(file_bytes)
    d = Document(bio)
    return "\n".join(p.text for p in d.paragraphs)

def read_pdf_stream(file_bytes: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed.")
    bio = io.BytesIO(file_bytes)
    try:
        try:
            reader = PdfReader(bio, strict=False)
        except TypeError:
            reader = PdfReader(bio)
        texts = []
        for page in getattr(reader, "pages", []) or []:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")
        return "\n".join(texts)
    except Exception as e:
        raise RuntimeError(f"PDF parse failed: {e}")

def extract_text(upload) -> str:
    name = upload.name.lower()
    if name.endswith(".txt"):
        return read_txt(upload)
    data = upload.read()
    if name.endswith(".docx"):
        return read_docx_stream(data)
    if name.endswith(".pdf"):
        return read_pdf_stream(data)
    raise ValueError("Unsupported file type")

def analyze_single_paper(client: OpenAI, model: str, paper_id: str, text: str) -> Dict[str, Any]:
    if len(text) > MAX_CHARS_PER_PAPER:
        text = text[:MAX_CHARS_PER_PAPER]
    system_msg = (
        "You are a meticulous research-methods auditor.\n"
        "Extract concise structured metadata, reported limitations, and inferred gaps grounded in the provided paper text.\n"
        "Avoid duplication; prefer short, atomic items."
    )
    user_msg = (
        f"PAPER_ID: {paper_id}\n\n"
        "Return only JSON matching the schema (no prose).\n\n"
        "--- PAPER FULL TEXT START ---\n"
        f"{text}\n"
        "--- PAPER FULL TEXT END ---\n"
    )
    content = safe_chat(
        client, model,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        use_schema=True,
        schema=paper_schema_json(),
    )
    raw = parse_possible_json(content)
    return normalize_analysis(raw, paper_id)

def embed(client: OpenAI, embedding_model: str, texts: List[str]) -> np.ndarray:
    out = []
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    B = 64
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        emb = client.embeddings.create(model=embedding_model, input=batch)
        vecs = [d.embedding for d in emb.data]
        out.append(np.array(vecs, dtype=np.float32))
    return np.vstack(out)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def greedy_cluster(vectors: np.ndarray, items: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    n = len(items)
    unused = set(range(n))
    clusters = []
    while unused:
        seed = unused.pop()
        members = [seed]
        centroid = vectors[seed].copy()
        changed = True
        while changed:
            changed = False
            to_add = []
            for j in list(unused):
                if cosine_sim(centroid, vectors[j]) >= threshold:
                    to_add.append(j)
            if to_add:
                for j in to_add:
                    unused.remove(j)
                    members.append(j)
                centroid = vectors[members].mean(axis=0)
                changed = True
        clusters.append({"members": members})
    return clusters

def choose_top_gap(cluster_stats: List[Dict[str, Any]], total_papers: int) -> int:
    best_idx, best_score = 0, -1.0
    for i, c in enumerate(cluster_stats):
        coverage = c.get("paper_coverage", 0) / max(1, total_papers)
        sev = c.get("avg_severity", 0) / 5.0
        score = 0.7 * coverage + 0.3 * sev
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx

def llm_narrative_corpus(client: OpenAI, model: str, clusters_payload: Dict[str, Any]) -> str:
    system = (
        "You write clear, non-technical academic summaries.\n"
        "Return plain text only: paragraphs separated by a blank line. No headings or bullets."
    )
    user = (
        "Given the JSON about clustered gaps, write 3–5 short paragraphs narrating the main shared gaps "
        "across the corpus (one paragraph per gap). Then add ONE final paragraph recommending the single gap "
        "with the best potential, including a brief rationale and a single-sentence suggestion for how to start.\n\n"
        "Rules: NO lists, NO bullets, NO headings, NO tables, NO numbers or percentages—just prose paragraphs. "
        "Keep it under 600 words.\n\n"
        "JSON:\n" + json.dumps(clusters_payload, ensure_ascii=False)
    )
    content = safe_chat(
        client, model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        use_schema=False, schema=None,
    )
    return re.sub(r"`{3,}.*?`{3,}", "", content, flags=re.DOTALL).strip()

def build_docx_narrative(narrative_text: str) -> bytes:
    if Document is None:
        return b""
    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Inches(0.8)
        sec.bottom_margin = Inches(0.8)
        sec.left_margin = Inches(0.8)
        sec.right_margin = Inches(0.8)
    title = doc.add_paragraph("Research Gaps — Narrative Summary")
    title.style = "Title"
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    from datetime import datetime
    sub = doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    for ptxt in [p.strip() for p in narrative_text.split("\n\n") if p.strip()]:
        p = doc.add_paragraph(ptxt)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.2
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

# ---------- Main action ----------
if run_btn:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar (or set it in Streamlit Secrets).")
        st.stop()
    if not uploads:
        st.warning("Please upload at least one file.")
        st.stop()

    client = build_client(api_key)

    # Step 1: per-paper analysis
    st.subheader("Step 1 — Analyzing papers")
    per_paper: List[Dict[str, Any]] = []
    progress = st.progress(0.0)
    status = st.empty()

    for i, up in enumerate(uploads, 1):
        status.info(f"Reading: {up.name}")
        try:
            text = extract_text(up)
        except Exception as e:
            st.warning(f"Skipping {up.name}: read error — {e}")
            progress.progress(i/len(uploads))
            continue
        if len(text.strip()) < 500:
            st.warning(f"Skipping {up.name}: very little text extracted (consider OCR).")
            progress.progress(i/len(uploads))
            continue

        status.info(f"LLM analysis: {up.name}")
        try:
            analysis = analyze_single_paper(client, model, up.name, text[:limit_chars])
        except Exception as e:
            st.warning(f"LLM analysis failed for {up.name}: {e}")
            progress.progress(i/len(uploads))
            continue

        per_paper.append(analysis)
        progress.progress(i/len(uploads))

    status.empty()
    if not per_paper:
        st.error("No papers analyzed successfully.")
        st.stop()

    st.success(f"Analyzed {len(per_paper)} papers.")

    # Step 2: aggregate gaps
    st.subheader("Step 2 — Aggregating shared gaps")
    gap_items = []
    for a in per_paper:
        pid = a.get("paper_id") or "unknown"
        for txt in a.get("reported_limitations", []) or []:
            if isinstance(txt, (str, int, float)):
                t = str(txt).strip()
                if t:
                    gap_items.append({"paper_id": pid, "text": t, "source": "Reported", "severity": 3})
        for g in a.get("inferred_gaps", []) or []:
            if isinstance(g, dict):
                desc = str(g.get("description") or g.get("text") or "").strip()
                if not desc:
                    continue
                try:
                    sev = int(round(float(g.get("severity", 3))))
                except Exception:
                    sev = 3
                sev = min(max(sev, 1), 5)
                gap_items.append({"paper_id": pid, "text": desc, "source": "Inferred", "severity": sev})
            elif isinstance(g, (str, int, float)):
                t = str(g).strip()
                if t:
                    gap_items.append({"paper_id": pid, "text": t, "source": "Inferred", "severity": 3})

    if not gap_items:
        st.warning("No gap statements found; cannot compute clusters.")
        st.stop()

    texts = [g["text"] for g in gap_items]
    vecs = embed(client, embedding_model, texts)
    clusters = greedy_cluster(vecs, gap_items, threshold=sim_threshold)

    # Step 3: cluster stats + pick top gap
    cluster_stats: List[Dict[str, Any]] = []
    for ci, c in enumerate(clusters):
        members = [gap_items[i] for i in c["members"]]
        uniq_papers = sorted({m["paper_id"] for m in members})
        severities = [m.get("severity", 3) for m in members]
        srcs = [m.get("source", "Reported") for m in members]
        text_samples = [m["text"] for m in members[:5]]
        cluster_stats.append({
            "cluster_index": ci,
            "paper_coverage": len(uniq_papers),
            "unique_papers": uniq_papers,
            "count": len(members),
            "avg_severity": float(np.mean(severities) if severities else 0),
            "reported_share": float(srcs.count("Reported") / len(srcs)) if srcs else 0.0,
            "inferred_share": float(srcs.count("Inferred") / len(srcs)) if srcs else 0.0,
            "samples": text_samples,
        })

    top_idx = choose_top_gap(cluster_stats, total_papers=len(per_paper)) if cluster_stats else -1

    clusters_payload = {
        "total_papers": len(per_paper),
        "clusters": [
            {
                "index": c["cluster_index"],
                "label": f"Cluster {c['cluster_index']}",
                "paper_coverage": c["paper_coverage"],
                "count": c["count"],
                "avg_severity": round(c["avg_severity"], 2),
                "reported_share": round(c["reported_share"], 3),
                "inferred_share": round(c["inferred_share"], 3),
                "example_statements": c["samples"],
                "is_recommended": (c["cluster_index"] == top_idx)
            }
            for c in cluster_stats
        ]
    }

    # Quick visual summary
    st.info(f"Found {len(cluster_stats)} gap clusters. Recommended cluster: **#{top_idx}**")
    with st.expander("Preview example statements (first cluster only)", expanded=False):
        if cluster_stats:
            st.write(cluster_stats[0]["samples"])

    # Step 4: LLM narrative + Word download
    st.subheader("Step 3 — Narrative report")
    try:
        narrative = llm_narrative_corpus(client, model, clusters_payload)
    except Exception as e:
        st.error(f"Narrative generation failed: {e}")
        st.stop()

    st.text_area("Narrative preview (plain paragraphs, no bullets)", value=narrative, height=250)

    docx_bytes = build_docx_narrative(narrative)
    if docx_bytes:
        st.download_button(
            "⬇️ Download Word report (.docx)",
            data=docx_bytes,
            file_name="final_report_narrative.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )
    else:
        st.warning("python-docx not installed; cannot build Word file.")

    # Also allow JSON export (optional)
    st.download_button(
        "⬇️ Download aggregated_gaps.json",
        data=json.dumps(clusters_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="aggregated_gaps.json",
        mime="application/json",
        use_container_width=True,
    )
