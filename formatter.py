import json
from datetime import datetime
from summarizer import refine_text

def format_output(documents, persona, job, top_chunks):
    output = {
        "metadata": {
            "documents": documents,
            "persona": persona,
            "job_to_be_done": job,
            "timestamp": datetime.utcnow().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for i, chunk in enumerate(top_chunks):
        output["extracted_sections"].append({
            "document": chunk["document"],
            "page_number": chunk["page_number"],
            "section_title": "Unknown",  # You can improve with header detection
            "importance_rank": i + 1
        })
        output["subsection_analysis"].append({
            "document": chunk["document"],
            "page_number": chunk["page_number"],
            "refined_text": refine_text(chunk["text"])
        })
    return output

def save_output_json(output, path="output/result.json"):
    with open(path, "w") as f:
        json.dump(output, f, indent=4)
