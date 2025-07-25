from sentence_transformers import SentenceTransformer, util
import torch

class SemanticAnalyzer:
    def __init__(self, model_path: str):
        """Initializes the analyzer by loading the sentence-transformer model."""
        print(f"Loading model from {model_path}...")
        self.model = SentenceTransformer(model_path)
        print("Model loaded successfully.")

    def _generate_hypothetical_queries(self, persona: str, job_to_be_done: str) -> list[str]:
        """Generates specific, targeted queries based on the persona and JTBD."""
        base_query = f"{persona} planning for: {job_to_be_done}"
        
        # This is where the contextual intelligence lies. We create specific queries.
        queries = [
            base_query,
            f"Recommendations for {job_to_be_done}",
            f"Nightlife, bars, and entertainment suitable for {job_to_be_done}",
            f"Adventure activities, beaches, and water sports for {job_to_be_done}",
            f"Budget-friendly hotels, hostels, and group accommodations for {job_to_be_done}",
            f"Affordable local food, cuisine, and dining experiences for {job_to_be_done}",
            f"Group transportation options and travel tips for {job_to_be_done}"
        ]
        print(f"Generated {len(queries)} hypothetical queries for deeper understanding.")
        return queries

    def rank_chunks(self, chunks: list[dict], persona: str, job_to_be_done: str) -> list[dict]:
        """
        Ranks chunks using query expansion and contextual reranking.
        """
        if not chunks:
            return []

        # 1. Generate multiple, specific queries
        hypothetical_queries = self._generate_hypothetical_queries(persona, job_to_be_done)
        
        # 2. Embed the queries and all chunk contents
        print("Embedding content and all hypothetical queries...")
        query_embeddings = self.model.encode(hypothetical_queries, convert_to_tensor=True)
        
        chunk_texts_for_embedding = [
            f"Title: {chunk.get('section_title', '')}. Content: {chunk.get('content', '')}"
            for chunk in chunks
        ]
        chunk_embeddings = self.model.encode(chunk_texts_for_embedding, convert_to_tensor=True)
        
        # 3. Calculate similarity matrix: (num_chunks x num_queries)
        similarity_matrix = util.cos_sim(chunk_embeddings, query_embeddings)
        
        # 4. For each chunk, find its BEST score across all queries
        # This is the "Intelligent Reranking" step.
        top_scores_per_chunk, _ = torch.max(similarity_matrix, dim=1)
        
        # 5. Assign scores and rank the original chunks
        for i, chunk in enumerate(chunks):
            chunk['score'] = top_scores_per_chunk[i].item()
            
        chunks.sort(key=lambda x: x['score'], reverse=True)
        
        for i, chunk in enumerate(chunks):
            chunk['importance_rank'] = i + 1
            
        print("Ranking complete using advanced contextual analysis.")
        return chunks
