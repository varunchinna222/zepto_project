from pathlib import Path
import os, chromadb
from sentence_transformers import SentenceTransformer
ROOT=Path(__file__).resolve().parent; DOCS=ROOT/'docs'; DB=ROOT/'chroma_db'; COLLECTION='zepto_policy'
MODEL='all-MiniLM-L6-v2'

def get_collection():
    client=chromadb.PersistentClient(path=str(DB)); return client.get_or_create_collection(COLLECTION,metadata={'hnsw:space':'cosine'})

def build_index():
    model=SentenceTransformer(MODEL); col=get_collection()
    existing=col.count()
    if existing: col.delete(where={})
    docs=[]; ids=[]; metas=[]
    for p in sorted(DOCS.glob('doc_*.txt')):
        text=p.read_text(encoding='utf8'); ids.append(p.stem); docs.append(text); metas.append({'document_id':p.stem})
    emb=model.encode(docs,normalize_embeddings=True).tolist(); col.add(ids=ids,documents=docs,metadatas=metas,embeddings=emb)
    return col.count()

def retrieve(query,k=3):
    model=SentenceTransformer(MODEL); col=get_collection(); q=model.encode([query],normalize_embeddings=True).tolist()[0]
    return col.query(query_embeddings=[q],n_results=k)
