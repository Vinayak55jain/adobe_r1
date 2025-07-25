import os
import json
from datetime import datetime
import time

from src.document_parser import parse_documents
from src.semantic_analyzer import SemanticAnalyzer

MODEL_PATH = './models/all-MiniLM-L6-v2'
TOP_N_RESULTS = 5

def run_analysis(doc_paths: list[str], persona: str, jbtd: str, output_dir: str = "outputs"):
    start_time = time.time()
    
    chunks = parse_documents(doc_paths)
    if not chunks:
        print("No text chunks could be parsed. Exiting.")
        return

    analyzer = SemanticAnalyzer(model_path=MODEL_PATH)
    ranked_chunks = analyzer.rank_chunks(chunks, persona, jbtd)
    
    # --- Build `extracted_sections` list (Adjusted Logic) ---
    extracted_sections_list = []
    found_sections = set()
    for chunk in ranked_chunks:
        if len(extracted_sections_list) >= TOP_N_RESULTS:
            break
        
        section_title = chunk.get('section_title', 'Untitled Section')
        # Use a tuple of (document, title) to track uniqueness
        section_identifier = (chunk["doc_name"], section_title)

        if section_identifier not in found_sections:
            extracted_sections_list.append({
                "document": chunk["doc_name"],
                "section_title": section_title,
                "importance_rank": len(extracted_sections_list) + 1,
                "page_number": chunk["page_number"]
            })
            found_sections.add(section_identifier)

    # --- Build `subsection_analysis` list ---
    subsection_analysis_list = []
    top_chunks = ranked_chunks[:TOP_N_RESULTS]
    for chunk in top_chunks:
        subsection_analysis_list.append({
            "document": chunk["doc_name"],
            "refined_text": chunk.get("content", ""), 
            "page_number": chunk["page_number"]
        })

    output_data = {
        "metadata": {
            "input_documents": [os.path.basename(p) for p in doc_paths],
            "persona": persona,
            "job_to_be_done": jbtd,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections_list,
        "subsection_analysis": subsection_analysis_list
    }
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    
    processing_time = time.time() - start_time
    print(f"\n--- Analysis Complete ---\nProcessing time: {processing_time:.2f} seconds")
    print(f"Output saved to: {output_path}")

if __name__ == '__main__':
    doc_filenames = [
        "South of France - Cities.pdf", "South of France - Cuisine.pdf",
        "South of France - History.pdf", "South of France - Restaurants and Hotels.pdf",
        "South of France - Things to Do.pdf", "South of France - Tips and Tricks.pdf",
        "South of France - Traditions and Culture.pdf"
    ]
    SAMPLE_DOCS = [os.path.join('test_data', f) for f in doc_filenames]
    SAMPLE_PERSONA = "Travel Planner"
    SAMPLE_JBTD = "Plan a trip of 4 days for a group of 10 college friends."

    missing_files = [doc for doc in SAMPLE_DOCS if not os.path.exists(doc)]
    if missing_files:
        print("Error: The following PDF files are missing from 'test_data':")
        for f in missing_files: print(f"- {os.path.basename(f)}")
    else:
        print("Running Travel Planner test case with new robust parser...")
        run_analysis(SAMPLE_DOCS, SAMPLE_PERSONA, SAMPLE_JBTD)
