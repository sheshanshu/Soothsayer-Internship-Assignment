"""
Retrieval + answer generation.
Requires:
 - an index + meta (built by pipeline.index)
 - Ollama running locally (optional; if missing, this returns a fallback answer)
This module exposes `answer_query(question, doc_meta, top_k=5)`.
"""

import os
import numpy as np
import json
import requests
from pipeline.index import build_index_and_embeddings, save_index_to_disk, load_index_from_disk
from sentence_transformers import SentenceTransformer

# Ollama config - local API
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")  # default Ollama port
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")  # change if different model pulled

# Cache models
_EMBED_MODEL = None

def _get_embed_model(model_name="all-mpnet-base-v2"):
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL

def _ensure_index_for_doc(doc_meta):
    # If doc_meta doesn't have index, build one and store path in doc_meta
    if doc_meta.get("index_built"):
        return doc_meta
    index, embeddings, meta = build_index_and_embeddings(doc_meta)
    # save to temp directory inside repo (.indices/<doc_id>)
    base_path = os.path.join(".indices", doc_meta.get("doc_id"))
    save_index_to_disk(index, embeddings, meta, base_path)
    doc_meta["index_built"] = True
    doc_meta["index_path"] = base_path
    return doc_meta

def _retrieve_top_k(doc_meta, question: str, top_k=5, embed_model_name="all-mpnet-base-v2"):
    # ensure index exists on disk
    doc_meta = _ensure_index_for_doc(doc_meta)
    base_path = doc_meta.get("index_path")
    loaded = load_index_from_disk(base_path)
    if loaded is None:
        return []
    index, embeddings, meta = loaded
    model = _get_embed_model(embed_model_name)
    q_emb = model.encode([question], convert_to_numpy=True)[0].astype(np.float32)
    D, I = index.search(np.array([q_emb]), top_k)
    hits = []
    for idx in I[0]:
        if idx < 0:
            continue
        chunk = meta["chunks"][idx]
        hits.append({"score": None, "chunk_index": int(idx), "chunk": chunk})
    return hits

def _call_ollama_with_prompt(prompt: str):
    """
    Call local Ollama API (assumes ollama daemon running).
    Endpoint: POST /api/generate (model in JSON)
    """
    try:
        url = f"{OLLAMA_URL}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.0
        }
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            # Ollama response structure may vary by version. Try common keys.
            if isinstance(data, dict) and "text" in data:
                return data["text"]
            # older/newer versions may return different shapes
            return json.dumps(data)
        else:
            return None
    except Exception as e:
        return None

def answer_query(question: str, doc_meta: dict, top_k=5):
    """
    High-level function to answer a question using doc_meta.
    Returns dict: { answer: str, sources: [str] }
    """
    hits = _retrieve_top_k(doc_meta, question, top_k=top_k)
    if not hits:
        return {"answer": "No indexed content available to answer the question.", "sources": []}

    # assemble context
    context_pieces = []
    sources = []
    for i, h in enumerate(hits, 1):
        txt = h["chunk"]["content"]
        context_pieces.append(f"[{i}] {txt}")
        meta = h["chunk"].get("meta", {})
        sources.append(f"Chunk[{i}] meta={meta}")

    context_text = "\n\n".join(context_pieces)
    prompt = (
        "You are an assistant that answers questions using ONLY the provided context pieces.\n\n"
        f"Context:\n{context_text}\n\n"
        f"User question: {question}\n\n"
        "Instructions:\n- Answer concisely and cite the context piece numbers you used (e.g., [1], [3]).\n"
        "- If the answer cannot be found in the context, reply: 'Insufficient information in the document.'\n"
        "- For numeric answers, show the numeric value and indicate which chunk it came from.\n"
    )

    # call Ollama
    resp = _call_ollama_with_prompt(prompt)
    if resp:
        answer_text = resp
    else:
        # fallback simple aggregator: return concatenated top hits
        snippet = "\n\n".join([p[:400] for p in context_pieces])
        answer_text = f"(Ollama unavailable) Top context snippets:\n\n{snippet}"

    return {"answer": answer_text, "sources": sources}

