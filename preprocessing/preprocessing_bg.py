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
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing text files",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory for processed files",
    )
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove stopwords from the text",
        default=False,
    )
    parser.add_argument(
        "--remove-punctuation",
        action="store_true",
        help="Remove punctuation from the text",
        default=False,
    )
    parser.add_argument(
        "--lower-text", action="store_true", help="Lowercase text", default=False
    )
    return parser.parse_args()

class TextPreprocessor:
    def __init__(self, remove_stopwords, remove_punctuation, lower_text, model_type='traditional'):
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lower_text = lower_text
        self.model_type = model_type

        # Boilerplate patterns
        self.boilerplate_patterns = {
            "headlines": r"^(.*\n)",
            "read_more": r"read more|most read|read all the|continue reading|source",
            "calls_to_action": r"subscribe to|join our list|unsubscribe",
            "content_sections": r"conclusion|summary|introduction|methods|results|references",
            "share": r"share this",
            "publish_date": r"published on",
            "copyright": r"copyright",
            "find_out_more": r"find out more|continue at [a-z]+\.[a-z]{1,3}",
            "terms_of_use": r"terms of use|comment policy",
            "about_the_writer": r"about the writer",
            "urls": r"https?:\/\/[^\s]+",
        }

        self.compiled_patterns = {
            name: (
                re.compile(pattern, re.IGNORECASE)
                if "[A-Z]" not in pattern
                else re.compile(pattern)
            )
            for name, pattern in self.boilerplate_patterns.items()
        }

        self.nlp = stanza.Pipeline("en", processors="tokenize,lemma,pos")
        self.nltk_stopwords = set(stopwords.words("english"))
        self.additional_stopwords = {
            "said", "would", "could", "also", "one", "two", "three", "first", 
            "second", "third", "new", "time", "year", "years", "many", 
            "much", "may", "might", "must", "like", "well",
        }
        self.all_stopwords = self.nltk_stopwords.union(self.additional_stopwords)

    def clean_text(self, text):
        if not text:
            return text

        # Remove patterns that should remove everything after them
        remove_after_patterns = [
            "about_the_writer", "share_your_thougths", "join_contribute", "ads",
        ]
        for pattern_name in remove_after_patterns:
            if pattern_name in self.compiled_patterns:
                pattern = self.compiled_patterns[pattern_name]
                text = self.remove_and_after(text, pattern)

        # Handle patterns that should remove the entire line
        remove_line_patterns = [
            "read_more", "un_subscribe", "sign_up", "follow_us", "click_here",
            "share", "publish_date", "copyright", "watch_video", "watch_more",
            "find out more", "latest_video", "terms_of_use", "urls",
            "journalist", "top_stories", "make_donation",
        ]
        for pattern_name in remove_line_patterns:
            if pattern_name in self.compiled_patterns:
                pattern = self.compiled_patterns[pattern_name]
                text = self.remove_line_if_matches(text, pattern)

        # Clean up extra whitespace and newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r"^\s+|\s+$", "", text)

        if self.model_type == 'traditional':
            if self.remove_punctuation:
                text = re.sub(r"[^\w\s]", "", text)

            if self.remove_stopwords:
                text = self.clean_stopwords(text)

            if self.lower_text:
                text = text.lower()

        return text

    def remove_line_if_matches(self, text, pattern):
        lines = text.split("\n")
        return "\n".join(line for line in lines if not pattern.search(line))

    def remove_and_after(self, text, pattern):
        match = pattern.search(text)
        if match:
            return text[: match.start()].strip()
        return text

    def clean_stopwords(self, text):
        filtered_tokens = [
            word for word in nltk.word_tokenize(text) if word.lower() not in self.all_stopwords
        ]
        return " ".join(filtered_tokens)

    def tokenize(self, text):
        return self.nlp(text)

    def save_file(self, doc, output_file):
        CoNLL.write_doc2conll(doc, output_file)

def main(data_dir, output_dir, remove_stopwords, remove_punctuation, lower_text):
    DATA_PATH = Path(data_dir)
    # Create separate output directories for traditional and deep learning
    traditional_output_dir = Path(output_dir) / "traditional-processed-documents"
    deep_learning_output_dir = Path(output_dir) / "deeplearning-processed-documents"

    traditional_output_dir.mkdir(parents=True, exist_ok=True)
    deep_learning_output_dir.mkdir(parents=True, exist_ok=True)

    # Process for traditional model
    preprocessor_traditional = TextPreprocessor(remove_stopwords, remove_punctuation, lower_text, model_type='traditional')
    for file_path in DATA_PATH.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"Processing {file_path.name} for traditional model...")
        cleaned_text = preprocessor_traditional.clean_text(text)
        tokenized_text = preprocessor_traditional.tokenize(cleaned_text)

        output_file = traditional_output_dir / f"{file_path.stem}.conllu"
        preprocessor_traditional.save_file(tokenized_text, output_file)

    # Process for deep learning model (keeping original text)
    preprocessor_deep_learning = TextPreprocessor(remove_stopwords=False, remove_punctuation=False, lower_text=False, model_type='deep_learning')
    for file_path in DATA_PATH.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"Processing {file_path.name} for deep learning model...")
        cleaned_text = preprocessor_deep_learning.clean_text(text)
        tokenized_text = preprocessor_deep_learning.tokenize(cleaned_text)

        output_file = deep_learning_output_dir / f"{file_path.stem}.conllu"
        preprocessor_deep_learning.save_file(tokenized_text, output_file)

if __name__ == "__main__":
    args = get_args()
    main(
        data_dir=args.input_dir,
        output_dir=args.output_dir,
        remove_stopwords=args.remove_stopwords,
        remove_punctuation=args.remove_punctuation,
        lower_text=args.lower_text,
    )
