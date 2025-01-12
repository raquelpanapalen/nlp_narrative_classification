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
        help="Output directory for processed files",
    )
    return parser.parse_args()

class TextPreprocessor:
    def __init__(self):
        # Initialize Stanza for Russian
        self.nlp = stanza.Pipeline("ru", processors="tokenize,lemma,pos")

    def clean_text(self, text):
        if not text:
            return text

        # Remove URLs
        text = re.sub(r"https?:\/\/[^\s]+", "", text)
        
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text):
        return self.nlp(text)

    def save_file(self, doc, output_file):
        CoNLL.write_doc2conll(doc, output_file)

def main(input_dir, output_dir):
    DATA_PATH = Path(input_dir)
    OUTPUT_PATH = Path(output_dir)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    preprocessor = TextPreprocessor()

    for file_path in DATA_PATH.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"Processing {file_path.name}...")
        cleaned_text = preprocessor.clean_text(text)
        tokenized_text = preprocessor.tokenize(cleaned_text)

        output_file = OUTPUT_PATH / f"{file_path.stem}.conllu"
        preprocessor.save_file(tokenized_text, output_file)

if __name__ == "__main__":
    args = get_args()
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
