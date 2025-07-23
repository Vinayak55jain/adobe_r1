import json
import os
from extract_text import extract_all_pdfs
from embeddings import get_query_embedding, get_chunk_embeddings
from relevance import rank_chunks
from formatter import format_output, save_output_json

def load_config():
    with open("config/input.json") as f:
        return json.load(f)

def main():
    config = load_config()
    input_dir = "input/"
    
    print("[1] Extracting text...")
    chunks = extract_all_pdfs(input_dir, config["documents"])

    print("[2] Embedding persona and job...")
    query_embedding = get_query_embedding(config["persona"], config["job"])
    
    print("[3] Embedding chunks...")
    chunk_embeddings = get_chunk_embeddings(chunks)

    print("[4] Ranking relevant sections...")
    top_chunks = rank_chunks(chunks, chunk_embeddings, query_embedding, top_k=10)

    print("[5] Formatting output...")
    result = format_output(config["documents"], config["persona"], config["job"], top_chunks)
    save_output_json(result)

    print("✅ Done! Output saved to `output/result.json`")

if __name__ == "__main__":
    main()
