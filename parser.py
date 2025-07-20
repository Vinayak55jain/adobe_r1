import sys
import os
import re
import json
import pickle
import numpy as np
from collections import Counter
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine, LTChar, LAParams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
import warnings
warnings.filterwarnings('ignore')

class PointBasedPDFOutlineExtractor:
    def __init__(self, model_type='logistic', max_features=5000, use_ml=True):
        """
        Initialize the enhanced point-based PDF outline extractor
        
        Args:
            model_type: 'logistic' or 'naive_bayes'
            max_features: Maximum number of features for vectorizer
            use_ml: Whether to use ML classifier or pure point-based heuristics
        """
        self.model_type = model_type
        self.max_features = max_features
        self.use_ml = use_ml
        self.vectorizer = None
        self.feature_selector = None
        self.classifier = None
        self.is_trained = False
        
        # Point-based scoring weights (based on research findings)
        self.scoring_weights = {
            'font_size_score': 30,      # Most important factor [3][8][11]
            'alignment_score': 25,       # Critical for title vs heading distinction [4][10]
            'spacing_score': 20,         # Whitespace analysis importance [23][37]
            'numbering_score': 15,       # Pattern recognition [Ground truth analysis]
            'position_score': 10         # Page position relevance [Research findings]
        }
        
    def extract_pdf_elements(self, pdf_path):
        """Enhanced PDF element extraction with detailed spacing analysis"""
        elements = []
        
        try:
            for page_num, page_layout in enumerate(extract_pages(pdf_path, laparams=LAParams())):
                page_elements = []
                for element in page_layout:
                    if isinstance(element, LTTextBox):
                        for text_line in element:
                            if isinstance(text_line, LTTextLine):
                                text = text_line.get_text().strip()
                                if not text or len(text) < 2:
                                    continue
                                    
                                # Enhanced font information extraction
                                font_sizes = []
                                font_names = []
                                font_weights = []
                                
                                for char in text_line:
                                    if isinstance(char, LTChar):
                                        font_sizes.append(char.size)
                                        font_names.append(char.fontname)
                                        # Check for bold indicators in font name
                                        font_weights.append(1 if any(bold_indicator in char.fontname.lower() 
                                                                   for bold_indicator in ['bold', 'black', 'heavy']) else 0)
                                
                                if font_sizes:
                                    avg_font_size = sum(font_sizes) / len(font_sizes)
                                    most_common_fontname = max(set(font_names), key=font_names.count) if font_names else "Unknown"
                                    is_bold = sum(font_weights) / len(font_weights) > 0.5 if font_weights else False
                                else:
                                    avg_font_size = 12.0
                                    most_common_fontname = "Unknown"
                                    is_bold = False
                                
                                element_data = {
                                    "text": text,
                                    "page_num": page_num + 1,
                                    "font_size": avg_font_size,
                                    "font_name": most_common_fontname,
                                    "is_bold": is_bold,
                                    "x0": text_line.bbox[0],
                                    "y0": text_line.bbox[1],
                                    "x1": text_line.bbox[2],
                                    "y1": text_line.bbox[3],
                                    "bbox": text_line.bbox
                                }
                                
                                page_elements.append(element_data)
                                elements.append(element_data)
                
                # Calculate spacing for current page elements
                self._calculate_page_spacing(page_elements, page_num + 1)
                
        except Exception as e:
            print(f"Error extracting from {pdf_path}: {e}")
            
        return elements

    def _calculate_page_spacing(self, page_elements, page_num):
        """Calculate spacing metrics for elements on a page"""
        if len(page_elements) < 2:
            return
            
        # Sort elements by y-coordinate (top to bottom)
        sorted_elements = sorted(page_elements, key=lambda x: -x['y1'])
        
        for i, element in enumerate(sorted_elements):
            # Calculate vertical spacing to next element
            if i < len(sorted_elements) - 1:
                next_element = sorted_elements[i + 1]
                vertical_gap = element['y0'] - next_element['y1']
                element['vertical_spacing'] = max(0, vertical_gap)
            else:
                element['vertical_spacing'] = 0
                
            # Calculate horizontal spacing patterns
            element['horizontal_spacing'] = self._analyze_horizontal_spacing(element['text'])

    def _analyze_horizontal_spacing(self, text):
        """Analyze horizontal spacing patterns in text"""
        # Check for excessive spacing patterns
        space_count = text.count('  ')  # Multiple spaces
        tab_count = text.count('\t')    # Tabs
        return space_count + (tab_count * 2)  # Weight tabs more heavily

    def calculate_point_scores(self, elements):
        """Calculate point-based scores for each element"""
        if not elements:
            return elements
            
        # Global statistics for normalization
        font_sizes = [el['font_size'] for el in elements if el['font_size']]
        if not font_sizes:
            return elements
            
        font_stats = {
            'min': min(font_sizes),
            'max': max(font_sizes),
            'mean': np.mean(font_sizes),
            'std': np.std(font_sizes),
            'percentiles': np.percentile(font_sizes, [25, 50, 75, 90, 95])
        }
        
        # Page dimensions for alignment calculations
        page_widths = {}
        for el in elements:
            page_num = el['page_num']
            if page_num not in page_widths:
                page_elements = [e for e in elements if e['page_num'] == page_num]
                page_widths[page_num] = max([e['x1'] for e in page_elements]) if page_elements else 612.0
        
        # Calculate scores for each element
        for el in elements:
            scores = {}
            
            # 1. Font Size Score (30 points max) [3][8][11]
            scores['font_size'] = self._calculate_font_size_score(el, font_stats)
            
            # 2. Alignment Score (25 points max) [4][10]
            scores['alignment'] = self._calculate_alignment_score(el, page_widths.get(el['page_num'], 612.0))
            
            # 3. Spacing Score (20 points max) [23][37]
            scores['spacing'] = self._calculate_spacing_score(el)
            
            # 4. Numbering Score (15 points max) [Ground truth analysis]
            scores['numbering'] = self._calculate_numbering_score(el)
            
            # 5. Position Score (10 points max)
            scores['position'] = self._calculate_position_score(el, elements)
            
            # Calculate weighted total score
            total_score = sum(scores[key] * self.scoring_weights[f"{key}_score"] / 100 
                             for key in scores.keys())
            
            # Add all scores to element
            el['scores'] = scores
            el['total_score'] = total_score
            el['classification_confidence'] = min(total_score / 100, 1.0)
            
        return elements

    def _calculate_font_size_score(self, element, font_stats):
        """Calculate font size-based score (0-100 scale)"""
        font_size = element['font_size']
        
        # Percentile-based scoring [3]
        if font_size >= font_stats['percentiles'][4]:  # 95th percentile
            base_score = 100
        elif font_size >= font_stats['percentiles'][3]:  # 90th percentile
            base_score = 85
        elif font_size >= font_stats['percentiles'][2]:  # 75th percentile
            base_score = 70
        elif font_size >= font_stats['percentiles'][1]:  # 50th percentile
            base_score = 50
        else:
            base_score = 20
        
        # Bold font bonus [Research findings]
        if element.get('is_bold', False):
            base_score = min(100, base_score + 15)
            
        return base_score

    def _calculate_alignment_score(self, element, page_width):
        """Calculate alignment-based score focusing on center vs left alignment"""
        x_center = (element['x0'] + element['x1']) / 2
        page_center = page_width / 2
        
        # Distance from center as percentage of page width
        center_distance = abs(x_center - page_center) / page_width
        
        # Scoring based on alignment [4][10]
        if center_distance < 0.1:  # Very centered (within 10% of page width)
            alignment_score = 100  # Strong title candidate
        elif center_distance < 0.25:  # Somewhat centered (within 25%)
            alignment_score = 70   # Possible title or major heading
        elif element['x0'] < page_width * 0.15:  # Left-aligned (within 15% of left margin)
            alignment_score = 60   # Typical heading alignment
        else:
            alignment_score = 20   # Body text or indented content
            
        return alignment_score

    def _calculate_spacing_score(self, element, max_spacing=50):
        """Calculate spacing-based score [23][37]"""
        vertical_spacing = element.get('vertical_spacing', 0)
        horizontal_spacing = element.get('horizontal_spacing', 0)
        
        # Normalize vertical spacing (headings typically have more space above/below)
        vertical_score = min(100, (vertical_spacing / max_spacing) * 100) if vertical_spacing > 0 else 30
        
        # Horizontal spacing (excessive spacing might indicate special formatting)
        horizontal_score = min(30, horizontal_spacing * 10) if horizontal_spacing > 0 else 0
        
        # Combine scores (vertical spacing is more important for headings)
        spacing_score = (vertical_score * 0.8) + (horizontal_score * 0.2)
        
        return min(100, spacing_score)

    def _calculate_numbering_score(self, element):
        """Calculate numbering pattern-based score based on ground truth analysis"""
        text = element['text'].strip()
        
        # H1 patterns from ground truth
        if re.match(r'^\d+\.\s+', text):  # "1. Introduction..."
            return 100
        elif text.lower() in ['revision history', 'table of contents', 'acknowledgements', 'references']:
            return 95
        
        # H2 patterns from ground truth  
        elif re.match(r'^\d+\.\d+\s+', text):  # "2.1 Intended Audience"
            return 85
        
        # H3 patterns (extrapolated)
        elif re.match(r'^\d+\.\d+\.\d+\s+', text):  # "2.1.1 Something"
            return 70
        
        # Other potential heading indicators
        elif text[0].isupper() and len(text.split()) <= 8:  # Capitalized, short
            return 40
        
        else:
            return 0

    def _calculate_position_score(self, element, all_elements):
        """Calculate position-based score"""
        page_num = element['page_num']
        y_pos = element['y1']
        
        # Get elements on same page for relative positioning
        page_elements = [el for el in all_elements if el['page_num'] == page_num]
        
        if not page_elements:
            return 50
        
        # Calculate relative position on page (0 = bottom, 1 = top)
        page_y_positions = [el['y1'] for el in page_elements]
        min_y, max_y = min(page_y_positions), max(page_y_positions)
        
        if max_y == min_y:
            relative_position = 0.5
        else:
            relative_position = (y_pos - min_y) / (max_y - min_y)
        
        # Score based on position (headings often appear near top of sections)
        if relative_position > 0.8:  # Top 20% of page
            return 100
        elif relative_position > 0.6:  # Top 40% of page
            return 80
        elif relative_position > 0.4:  # Middle section
            return 60
        else:  # Lower section
            return 40

    def classify_elements_by_points(self, elements):
        """Classify elements based on point scores"""
        if not elements:
            return {"title": "", "outline": []}
        
        # Calculate point scores
        elements = self.calculate_point_scores(elements)
        
        # Filter potential heading candidates based on minimum score threshold
        candidates = [el for el in elements if el['total_score'] >= 25]  # Minimum 25/100 points
        
        # Sort by score for better classification
        candidates.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Title detection (highest scoring, centered, first page preferred)
        title_candidates = [el for el in candidates 
                          if el['page_num'] == 1 and el['scores']['alignment'] >= 70]
        
        title_text = ""
        title_id = None
        
        if title_candidates:
            # Choose highest scoring title candidate
            title_element = max(title_candidates, key=lambda x: x['total_score'])
            title_text = title_element['text']
            title_id = id(title_element)
        
        # Heading classification
        outline = []
        used_ids = {title_id} if title_id else set()
        
        for element in candidates:
            if id(element) in used_ids:
                continue
                
            # Determine heading level based on score thresholds and patterns
            level = self._determine_heading_level(element)
            
            if level:
                outline.append({
                    "level": level,
                    "text": element['text'],
                    "page": element['page_num'],
                    "confidence": element['classification_confidence'],
                    "score_breakdown": element['scores']
                })
                used_ids.add(id(element))
        
        # Sort outline by page and position
        outline.sort(key=lambda x: (x['page'], -elements[[el['text'] for el in elements].index(x['text'])]['y1']))
        
        # Remove confidence and score_breakdown from final output (for compatibility)
        final_outline = [{"level": h["level"], "text": h["text"], "page": h["page"]} for h in outline]
        
        return {
            "title": title_text,
            "outline": final_outline
        }

    def _determine_heading_level(self, element):
        """Determine heading level based on point scores and patterns"""
        total_score = element['total_score']
        scores = element['scores']
        text = element['text'].strip()
        
        # High-confidence classifications based on patterns
        if scores['numbering'] >= 95:  # Strong pattern match (H1)
            return 'H1'
        elif scores['numbering'] >= 85:  # Strong H2 pattern
            return 'H2'
        elif scores['numbering'] >= 70:  # H3 pattern
            return 'H3'
        
        # Score-based classification
        elif total_score >= 80:
            return 'H1'
        elif total_score >= 65:
            return 'H2'
        elif total_score >= 50:
            return 'H3'
        
        # Additional heuristics for edge cases
        elif (scores['font_size'] >= 70 and scores['alignment'] >= 60 and 
              len(text.split()) <= 10):
            return 'H1' if scores['font_size'] >= 85 else 'H2'
        
        return None  # Not classified as heading

    def process_pdf(self, pdf_path):
        """Process a PDF using point-based classification"""
        try:
            elements = self.extract_pdf_elements(pdf_path)
            if not elements:
                return {"title": "", "outline": []}
            
            # Use point-based classification
            result = self.classify_elements_by_points(elements)
            
            return result
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return {"title": "", "outline": []}

    def save_model(self, model_path):
        """Save the scoring configuration"""
        model_data = {
            'scoring_weights': self.scoring_weights,
            'model_type': self.model_type,
            'max_features': self.max_features,
            'use_ml': self.use_ml
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Point-based model saved to {model_path} (Size: {size_mb:.2f} MB)")

    def load_model(self, model_path):
        """Load scoring configuration"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scoring_weights = model_data.get('scoring_weights', self.scoring_weights)
        self.model_type = model_data.get('model_type', self.model_type)
        self.max_features = model_data.get('max_features', self.max_features)
        self.use_ml = model_data.get('use_ml', self.use_ml)
        
        print(f"Point-based model loaded from {model_path}")


def process_directory_with_point_system(input_dir, output_dir, model_path=None):
    """Process directory with enhanced point-based approach"""
    
    # Initialize extractor with point-based system
    extractor = PointBasedPDFOutlineExtractor(
        model_type='point_based',
        max_features=0,  # Not used for point-based system
        use_ml=False     # Use pure point-based classification
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process specific file (as per your original code)
    input_pdf_path = r"C:\Users\vinay\OneDrive\Documents\adobehack\input\sample.pdf"
    output_json_path = r"C:\Users\vinay\OneDrive\Documents\adobehack\output\sample.json"
    
    if not os.path.isfile(input_pdf_path):
        print(f"File not found: {input_pdf_path}")
        return
    
    print(f"Processing {os.path.basename(input_pdf_path)} with Point-Based System...")
    
    result = extractor.process_pdf(input_pdf_path)
    
    # Save result
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"  Title: '{result['title']}'")
    print(f"  Found {len(result['outline'])} headings")
    
    # Save model configuration
    if model_path:
        extractor.save_model(model_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python point_based_pdf_extractor.py <input_dir> <output_dir>")
        sys.exit(1)
    
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    
    if not os.path.exists(input_directory):
        print(f"Input directory does not exist: {input_directory}")
        sys.exit(1)
    
    # Process with enhanced point-based system
    model_path = "point_based_model.pkl"
    process_directory_with_point_system(input_directory, output_directory, model_path)
    
    print("\nPoint-Based Processing complete!")
    print("\nScoring System Weights:")
    print("- Font Size: 30% (Most critical factor)")
    print("- Alignment: 25% (Title vs heading distinction)")  
    print("- Spacing: 20% (Whitespace analysis)")
    print("- Numbering: 15% (Pattern recognition)")
    print("- Position: 10% (Page location relevance)")
