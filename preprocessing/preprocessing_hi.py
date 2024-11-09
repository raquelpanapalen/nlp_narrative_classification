import re
from pathlib import Path
import nltk
from nltk.corpus import stopwords
import stanza
from stanza.utils.conll import CoNLL
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Preprocess text files")
    parser.add_argument(
        "-i", "--input_dir", type=str, required=True, help="Directory containing text files",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Output directory for processed files",
    )
    parser.add_argument(
        "--remove-stopwords", action="store_true", help="Remove stopwords from the text", default=False,
    )
    parser.add_argument(
        "--remove-punctuation", action="store_true", help="Remove punctuation from the text", default=False,
    )
    parser.add_argument(
        "--lower-text", action="store_true", help="Lowercase text", default=False
    )
    return parser.parse_args()

class TextPreprocessor:
    def __init__(self, remove_stopwords, remove_punctuation, lower_text):
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lower_text = lower_text

        self.boilerplate_patterns = {
            "source_notice": r"translated from [a-zA-Z]+|source",
            "unnecessary_clauses": r"for more details|to learn more|click here|visit our site",
            "urls": r"https?:\/\/[^\s]+",
        }

        self.compiled_patterns = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in self.boilerplate_patterns.items()}
        
        self.nlp = stanza.Pipeline("en", processors="tokenize,lemma,pos")

        self.nltk_stopwords = set(stopwords.words("english"))

        # Additional common words to filter
        self.additional_stopwords = {
            "said", "would", "could", "also", "one", "two", "three", "first", 
            "second", "third", "new", "time", "year", "years", "many", "much",
            "may", "might", "must", "like", "well", "therefore", "thus", "furthermore",
            "mr.", "sir", "madam", "miss", "kindly", "please", "respectfully", "dear", 
            "is", "are", "was", "were", "have", "has", "had", "do", "does", "did", 
            "doing", "been", "being", "and", "but", "or", "because", "although", "if", 
            "then", "that", "this", "I", "he", "she", "it", "we", "they", "you", "his", 
            "her", "their", "its", "very", "too", "so", "much", "many", "more", "some", 
            "only", "just", "still", "always", "never", "often", "about", "actually", 
            "basically", "in", "on", "at", "to", "from", "with", "by", "for", "of"
        }

        
        self.all_stopwords = self.nltk_stopwords.union(self.additional_stopwords)

    def clean_text(self, text):
        """Main preprocessing function to clean text."""
        if not text:
            return text

        # Remove patterns from translation artifacts
        for pattern_name, pattern in self.compiled_patterns.items():
            text = pattern.sub("", text)
        
        # Clean up extra whitespace and newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r"^\s+|\s+$", "", text)

        # Additional preprocessing steps based on flags
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", "", text)
        if self.remove_stopwords:
            text = self.clean_stopwords(text)
        if self.lower_text:
            text = text.lower()

        return text

    def clean_stopwords(self, text):
        """Remove stopwords from the text."""
        filtered_tokens = [word for word in nltk.word_tokenize(text) if word.lower() not in self.all_stopwords]
        return " ".join(filtered_tokens)

    def tokenize(self, text):
        """Tokenize text using Stanza."""
        return self.nlp(text)

    def save_file(self, doc, output_file):
        """Save to CoNLL format."""
        CoNLL.write_doc2conll(doc, output_file)

def main(data_dir, output_dir, remove_stopwords, remove_punctuation, lower_text):
    DATA_PATH = Path(data_dir)
    OUTPUT_PATH = Path(output_dir)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    preprocessor = TextPreprocessor(remove_stopwords, remove_punctuation, lower_text)

    for file_path in DATA_PATH.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        cleaned_text = preprocessor.clean_text(text)
        tokenized_text = preprocessor.tokenize(cleaned_text)
        output_file = OUTPUT_PATH / f"{file_path.stem}.conllu"
        preprocessor.save_file(tokenized_text, output_file)

if __name__ == "__main__":
    args = get_args()
    main(
        data_dir=args.input_dir,
        output_dir=args.output_dir,
        remove_stopwords=args.remove_stopwords,
        remove_punctuation=args.remove_punctuation,
        lower_text=args.lower_text,
    )
