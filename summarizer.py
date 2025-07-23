def refine_text(text, max_lines=5):
    lines = [line.strip() for line in text.split("\n") if len(line.split()) > 4]
    return ' '.join(lines[:max_lines])
