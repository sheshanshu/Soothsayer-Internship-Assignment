Chunking + embedding + FAISS index management.
Functions:
 - build_index_and_embeddings(doc_meta, model_name)
 - save_index_to_disk(index, embeddings, meta, path)
 - load_index_from_disk(path)
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

DEFAULT_MODEL = "all-mpnet-base-v2"  # switch to a smaller model if you need speed

def build_index_and_embeddings(doc_meta: dict, model_name: str = DEFAULT_MODEL):
    """
    Build embeddings for doc_meta['chunks'] and return FAISS index and meta container.
    """
    chunks = doc_meta.get("chunks", [])
    texts = [c["content"] for c in chunks]
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    # meta contains mapping from idx -> chunk metadata
    meta = {"chunks": chunks, "embeddings_shape": embeddings.shape}
    return index, embeddings, meta

def save_index_to_disk(index, embeddings, meta, base_path: str):
    """
    Save FAISS index and metadata (pickles).
    base_path is directory path where files will be created.
    """
    os.makedirs(base_path, exist_ok=True)
    faiss.write_index(index, os.path.join(base_path, "index.faiss"))
    np.save(os.path.join(base_path, "embeddings.npy"), embeddings)
    with open(os.path.join(base_path, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    return base_path

def load_index_from_disk(base_path: str):
    idx_path = os.path.join(base_path, "index.faiss")
    emb_path = os.path.join(base_path, "embeddings.npy")
    meta_path = os.path.join(base_path, "meta.pkl")
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        return None
    index = faiss.read_index(idx_path)
    embeddings = np.load(emb_path) if os.path.exists(emb_path) else None
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, embeddings, meta
