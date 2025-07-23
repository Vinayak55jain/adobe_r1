import pymupdf as fitz  # Correct import for PyMuPDF
import os

def extract_all_pdfs(documents_dir, filenames):
    all_chunks = []
    
    for filename in filenames:
        path = os.path.join(documents_dir, filename)

        if not os.path.exists(path):
            print(f"[!] File not found: {path}")
            continue

        try:
            doc = fitz.open(path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)  # Load page explicitly
                text = page.get_text()

                if len(text.strip()) > 50:  # Skip nearly empty pages
                    all_chunks.append({
                        "document": filename,
                        "page_number": page_num + 1,
                        "text": text
                    })
            doc.close()
        except Exception as e:
            print(f"[!] Error processing {filename}: {e}")

    return all_chunks
