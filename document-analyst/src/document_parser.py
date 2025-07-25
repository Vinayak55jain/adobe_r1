import fitz
import os
import re
import collections

# --- Helper functions ---
def get_body_font_size(doc: fitz.Document, default_size: float = 10.0) -> float:
    if not doc or len(doc) == 0: return default_size
    font_sizes = collections.Counter()
    try:
        for block in doc[0].get_text("dict")["blocks"]:
            if "lines" not in block: continue
            for line in block["lines"]:
                for span in line["spans"]: font_sizes[round(span["size"])] += 1
    except IndexError: return default_size
    return font_sizes.most_common(1)[0][0] if font_sizes else default_size

def split_large_chunk(chunk: dict, max_len: int = 2500) -> list[dict]:
    """Splits a chunk's content if it's too long."""
    if len(chunk['content']) <= max_len:
        return [chunk]
    
    new_chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', chunk['content'])
    current_content = ""
    for sentence in sentences:
        if len(current_content) + len(sentence) > max_len:
            new_chunk = chunk.copy()
            new_chunk['content'] = current_content.strip()
            new_chunks.append(new_chunk)
            current_content = ""
        current_content += sentence + " "
    
    if current_content.strip():
        new_chunk = chunk.copy()
        new_chunk['content'] = current_content.strip()
        new_chunks.append(new_chunk)
        
    return new_chunks

# --- Main Parsing Function ---
def parse_documents(doc_paths: list[str]) -> list[dict]:
    """Parses PDFs into structured chunks with adaptive heuristics and size control."""
    raw_chunks = []
    for doc_path in doc_paths:
        doc_name = os.path.basename(doc_path)
        try: doc = fitz.open(doc_path)
        except Exception as e:
            print(f"Error opening {doc_name}: {e}"); continue

        body_font_size = get_body_font_size(doc)
        current_title, current_content, current_page = "Introduction", "", 1
        
        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("dict", sort=True)["blocks"]
            for block in blocks:
                if "lines" not in block or not block['lines']: continue
                block_text = page.get_text("text", clip=block['bbox']).strip()
                if not block_text: continue

                is_heading = False
                first_span = block['lines'][0]['spans'][0] if block['lines'][0]['spans'] else {}
                if first_span:
                    is_larger = first_span.get('size', 0) > body_font_size * 1.05
                    is_bold = any(s in first_span.get('font', '').lower() for s in ['bold', 'black'])
                    is_short = len(block['lines']) <= 2 and len(block_text) < 100
                    if is_short and (is_larger or is_bold): is_heading = True

                if is_heading:
                    if current_content.strip():
                        raw_chunks.append({"doc_name": doc_name, "page_number": current_page, "section_title": current_title, "content": re.sub(r'\s+', ' ', current_content).strip()})
                    current_title, current_content, current_page = block_text, "", page_num
                else:
                    current_content += block_text + "\n"

        if current_content.strip():
            raw_chunks.append({"doc_name": doc_name, "page_number": current_page, "section_title": current_title, "content": re.sub(r'\s+', ' ', current_content).strip()})
    
    # Final processing: split oversized chunks
    final_chunks = []
    for chunk in raw_chunks:
        final_chunks.extend(split_large_chunk(chunk))

    print(f"Parsed and refined into {len(final_chunks)} chunks from {len(doc_paths)} documents.")
    return final_chunks
