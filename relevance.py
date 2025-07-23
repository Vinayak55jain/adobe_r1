from sklearn.metrics.pairwise import cosine_similarity

def rank_chunks(chunks, chunk_embeddings, query_embedding, top_k=10):
    scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
    for i, chunk in enumerate(chunks):
        chunk["score"] = scores[i]
    sorted_chunks = sorted(chunks, key=lambda x: x["score"], reverse=True)
    return sorted_chunks[:top_k]
