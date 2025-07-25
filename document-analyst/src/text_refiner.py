from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

def refine_text_extractive(full_text: str, sentence_count: int = 3) -> str:
    """
    Generates a concise extractive summary for the given text.
    """
    if not full_text:
        return ""
    
    parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    
    return " ".join([str(sentence) for sentence in summary])
