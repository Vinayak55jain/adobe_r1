from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # < 100MB

def get_query_embedding(persona, job):
    return model.encode(f"{persona}. Task: {job}")

def get_chunk_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)
    return embeddings
