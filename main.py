import fitz  # PyMuPDF
import json
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class TextElement:
    """Enhanced text element with comprehensive metadata"""
    text: str
    page_num: int
    font_size: float
    font_name: str
    is_bold: bool
    x0: float
    y0: float
    x1: float
    y1: float
    
    # Container-specific attributes
    pre_context: List[str] = None
    post_context: List[str] = None
    context_score: float = 0.0

class TwoContainerPDFExtractor:
    def __init__(self, pre_container_size=3, post_container_size=3):
        """
        Initialize the two-container PDF extractor
        
        Args:
            pre_container_size: Number of text elements to store before potential headings
            post_container_size: Number of text elements to store after potential headings
        """
        self.pre_container_size = pre_container_size
        self.post_container_size = post_container_size
        
        # Containers for context analysis
        self.pre_heading_container = deque(maxlen=pre_container_size)
        self.post_heading_container = deque(maxlen=post_container_size)
        
        # Classification thresholds based on research findings[4][7]
        self.font_size_threshold_ratio = 1.2
        self.position_weight = 0.3
        self.context_weight = 0.2
        self.format_weight = 0.5

    def extract_pdf_elements(self, pdf_path: str) -> List[TextElement]:
        """
        Extract text elements with comprehensive metadata using PyMuPDF
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of TextElement objects with rich metadata
        """
        elements = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get text with detailed formatting information[6][8]
                text_dict = page.get_text("dict")
                
                for block in text_dict["blocks"]:
                    if "lines" not in block:
                        continue
                        
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if not text or len(text) < 2:
                                continue
                            
                            # Enhanced font analysis[3]
                            font_size = span["size"]
                            font_name = span["font"]
                            is_bold = self._detect_bold_font(font_name, span.get("flags", 0))
                            
                            # Precise positioning
                            bbox = span["bbox"]
                            
                            element = TextElement(
                                text=text,
                                page_num=page_num + 1,
                                font_size=font_size,
                                font_name=font_name,
                                is_bold=is_bold,
                                x0=bbox[0],
                                y0=bbox[1], 
                                x1=bbox[2],
                                y1=bbox[3],
                                pre_context=[],
                                post_context=[]
                            )
                            
                            elements.append(element)
            
            doc.close()
            
        except Exception as e:
            print(f"Error extracting from {pdf_path}: {e}")
            
        return elements

    def _detect_bold_font(self, font_name: str, flags: int) -> bool:
        """
        Detect if font is bold using multiple indicators
        
        Args:
            font_name: Font name string
            flags: PyMuPDF font flags
            
        Returns:
            Boolean indicating if font is bold
        """
        # Check font name for bold indicators
        bold_indicators = ["bold", "black", "heavy", "demi", "medium"]
        name_is_bold = any(indicator in font_name.lower() for indicator in bold_indicators)
        
        # Check PyMuPDF flags (flag 16 = bold)
        flags_is_bold = bool(flags & 2**4)
        
        return name_is_bold or flags_is_bold

    def populate_containers(self, elements: List[TextElement]) -> List[TextElement]:
        """
        Populate pre and post containers for context analysis
        
        Args:
            elements: List of text elements
            
        Returns:
            Elements with populated container contexts
        """
        enhanced_elements = []
        
        for i, element in enumerate(elements):
            # Populate pre-heading container (text before current element)
            pre_start = max(0, i - self.pre_container_size)
            element.pre_context = [
                elements[j].text for j in range(pre_start, i)
                if self._is_valid_context_text(elements[j])
            ]
            
            # Populate post-heading container (text after current element)
            post_end = min(len(elements), i + self.post_container_size + 1)
            element.post_context = [
                elements[j].text for j in range(i + 1, post_end)
                if self._is_valid_context_text(elements[j])
            ]
            
            # Calculate context score
            element.context_score = self._calculate_context_score(element)
            
            enhanced_elements.append(element)
            
        return enhanced_elements

    def _is_valid_context_text(self, element: TextElement) -> bool:
        """
        Validate if text element is suitable for context analysis
        
        Args:
            element: Text element to validate
            
        Returns:
            Boolean indicating if element is valid context
        """
        text = element.text.lower().strip()
        
        # Filter out noise
        if (len(text) < 3 or 
            text.isdigit() or
            'copyright' in text or
            'page' in text and text.replace('page', '').strip().isdigit()):
            return False
            
        return True

    def _calculate_context_score(self, element: TextElement) -> float:
        """
        Calculate context score based on surrounding text patterns
        
        Args:
            element: Text element to score
            
        Returns:
            Context score (0.0 to 1.0)
        """
        score = 0.0
        
        # Pre-context analysis
        if element.pre_context:
            # Check if previous text suggests a section end
            last_pre = element.pre_context[-1].lower()
            if any(indicator in last_pre for indicator in ['.', 'conclusion', 'summary']):
                score += 0.3
        
        # Post-context analysis  
        if element.post_context:
            # Check if following text suggests content continuation
            first_post = element.post_context[0].lower()
            if (len(first_post) > 20 and 
                not first_post[0].isupper() and
                not any(indicator in first_post for indicator in ['1.', '2.', 'a.', 'i.'])):
                score += 0.4
        
        # Context transition patterns
        if (element.pre_context and element.post_context and
            len(element.text.split()) <= 8):  # Heading-like length
            score += 0.3
            
        return min(score, 1.0)

    def classify_elements(self, elements: List[TextElement]) -> Dict[str, Any]:
        """
        Classify elements using two-container approach with advanced heuristics
        
        Args:
            elements: List of enhanced text elements
            
        Returns:
            Dictionary with title and outline structure
        """
        if not elements:
            return {"title": "", "outline": []}
        
        # Calculate document-wide statistics
        font_sizes = [el.font_size for el in elements]
        avg_font_size = sum(font_sizes) / len(font_sizes)
        max_font_size = max(font_sizes)
        font_size_std = (sum((fs - avg_font_size) ** 2 for fs in font_sizes) / len(font_sizes)) ** 0.5
        
        # Title detection using containers and position
        title = self._detect_title(elements, max_font_size)
        
        # Heading classification
        outline = self._classify_headings(elements, avg_font_size, font_size_std, title)
        
        return {
            "title": title,
            "outline": outline
        }

    def _detect_title(self, elements: List[TextElement], max_font_size: float) -> str:
        """
        Detect document title using container context and positioning
        
        Args:
            elements: Text elements
            max_font_size: Maximum font size in document
            
        Returns:
            Document title string
        """
        first_page_elements = [el for el in elements if el.page_num == 1]
        if not first_page_elements:
            return ""
        
        # Find title candidates
        candidates = []
        for element in first_page_elements[:10]:  # Check first 10 elements
            if (element.font_size >= max_font_size * 0.9 and
                len(element.text.split()) > 2 and
                element.y1 > max(el.y1 for el in first_page_elements) * 0.7):
                
                # Container-based validation
                container_score = 0.0
                if not element.pre_context:  # Likely at document start
                    container_score += 0.4
                if element.post_context and len(element.post_context[0]) > 15:
                    container_score += 0.3
                
                candidates.append((element, container_score))
        
        if candidates:
            # Choose best candidate based on font size and container score
            best_candidate = max(candidates, 
                               key=lambda x: (x[0].font_size, x[1], -x[0].y0))
            return best_candidate[0].text
        
        return ""

    def _classify_headings(self, elements: List[TextElement], 
                         avg_font_size: float, font_size_std: float,
                         title: str) -> List[Dict[str, Any]]:
        """
        Classify headings using two-container approach with heuristics
        
        Args:
            elements: Text elements
            avg_font_size: Average font size
            font_size_std: Font size standard deviation  
            title: Document title to exclude
            
        Returns:
            List of classified headings
        """
        outline = []
        title_id = id(next((el for el in elements if el.text == title), None))
        
        for element in elements:
            if id(element) == title_id or not self._is_heading_candidate(element, avg_font_size):
                continue
            
            # Multi-factor scoring based on research[4][7]
            font_score = self._calculate_font_score(element, avg_font_size, font_size_std)
            position_score = self._calculate_position_score(element, elements)
            format_score = self._calculate_format_score(element)
            container_score = element.context_score
            
            # Weighted total score
            total_score = (
                font_score * self.format_weight +
                position_score * self.position_weight +
                format_score * 0.3 +
                container_score * self.context_weight
            )
            
            # Classify based on score and patterns
            level = self._determine_heading_level(element, total_score)
            
            if level:
                outline.append({
                    "level": level,
                    "text": element.text,
                    "page": element.page_num,
                    "confidence": total_score
                })
        
        # Sort by page and position
        outline.sort(key=lambda x: (x["page"], -next(
            el.y1 for el in elements if el.text == x["text"]
        )))
        
        # Remove confidence scores for final output
        return [{"level": h["level"], "text": h["text"], "page": h["page"]} 
                for h in outline]

    def _is_heading_candidate(self, element: TextElement, avg_font_size: float) -> bool:
        """Check if element is a potential heading candidate"""
        return (
            3 <= len(element.text) <= 200 and
            len(element.text.split()) <= 15 and
            element.font_size >= avg_font_size * 0.9 and
            not self._is_noise_text(element.text)
        )

    def _is_noise_text(self, text: str) -> bool:
        """Filter out noise text (headers, footers, page numbers)"""
        text_lower = text.lower().strip()
        return (
            text.strip().isdigit() or
            'copyright' in text_lower or
            'page' in text_lower and any(c.isdigit() for c in text_lower) or
            re.match(r'^(figure|table|appendix)\s+\d+', text_lower)
        )

    def _calculate_font_score(self, element: TextElement, 
                            avg_font_size: float, font_size_std: float) -> float:
        """Calculate font-based score"""
        size_ratio = element.font_size / avg_font_size
        bold_bonus = 0.3 if element.is_bold else 0.0
        
        if size_ratio >= 1.5:
            return 1.0 + bold_bonus
        elif size_ratio >= 1.2:
            return 0.8 + bold_bonus
        elif size_ratio >= 1.0:
            return 0.6 + bold_bonus
        else:
            return 0.3 + bold_bonus

    def _calculate_position_score(self, element: TextElement, 
                                all_elements: List[TextElement]) -> float:
        """Calculate position-based score"""
        page_elements = [el for el in all_elements if el.page_num == element.page_num]
        if not page_elements:
            return 0.5
        
        max_y = max(el.y1 for el in page_elements)
        relative_y = element.y1 / max_y
        
        # Higher on page = better score
        return relative_y

    def _calculate_format_score(self, element: TextElement) -> float:
        """Calculate formatting-based score"""
        score = 0.0
        text = element.text
        
        # Check for numbering patterns
        if re.match(r'^\d+\.\s+', text):
            score += 0.9
        elif re.match(r'^\d+\.\d+\s+', text):
            score += 0.7
        elif re.match(r'^\d+\.\d+\.\d+\s+', text):
            score += 0.5
        
        # Check text formatting
        if text.isupper():
            score += 0.3
        elif text.istitle():
            score += 0.2
        
        return min(score, 1.0)

    def _determine_heading_level(self, element: TextElement, total_score: float) -> Optional[str]:
        """Determine heading level based on total score and patterns"""
        text = element.text.strip()
        
        # Pattern-based classification with high confidence
        if re.match(r'^\d+\.\s+', text):
            return "H1"
        elif re.match(r'^\d+\.\d+\s+', text):
            return "H2"
        elif re.match(r'^\d+\.\d+\.\d+\s+', text):
            return "H3"
        
        # Score-based classification
        if total_score >= 0.8:
            return "H1"
        elif total_score >= 0.6:
            return "H2"
        elif total_score >= 0.4:
            return "H3"
        
        return None

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main processing function combining all steps
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted outline
        """
        try:
            print(f"Processing {pdf_path}...")
            
            # Step 1: Extract text with metadata
            elements = self.extract_pdf_elements(pdf_path)
            if not elements:
                return {"title": "", "outline": []}
            
            print(f"Extracted {len(elements)} text elements")
            
            # Step 2: Populate containers for context analysis
            elements = self.populate_containers(elements)
            print("Container context populated")
            
            # Step 3: Classify using two-container approach
            result = self.classify_elements(elements)
            print(f"Found title: '{result['title']}'")
            print(f"Found {len(result['outline'])} headings")
            
            return result
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return {"title": "", "outline": []}

def main():
    """Example usage of the two-container PDF extractor"""
    
    # Initialize extractor with custom container sizes
    extractor = TwoContainerPDFExtractor(
        pre_container_size=3,   # Store 3 elements before potential headings
        post_container_size=3   # Store 3 elements after potential headings
    )
    
    # Process PDF
    pdf_path = r"C:\Users\vinay\OneDrive\Documents\adobehack\input\file01.pdf"  # Replace with your PDF path
    result = extractor.process_pdf(pdf_path)
    
    # Save results
    output_path = r"C:\Users\vinay\OneDrive\Documents\adobehack\output\file01.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_path}")
    print(f"Title: {result['title']}")
    print(f"Outline entries: {len(result['outline'])}")
    
    # Display outline structure
    for item in result['outline']:
        indent = "  " * (int(item['level'][1]) - 1)
        print(f"{indent}{item['level']}: {item['text']} (Page {item['page']})")

if __name__ == "__main__":
    main()
