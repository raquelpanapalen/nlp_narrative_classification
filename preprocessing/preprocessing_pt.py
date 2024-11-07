import re
from pathlib import Path
import nltk
from nltk.corpus import stopwords
import argparse
import string
import spacy

"""
This script preprocesses text files for two types of models: traditional and deep learning.
The traditional model applies cleaning techniques like removing stopwords and punctuation,
whereas the deep learning model retains the original text to leverage complex representations
without extensive preprocessing.
It can also output files in CoNLL-U-like format with detailed annotations.
"""

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
    parser.add_argument(
        "--lemmatize", action="store_true", help="Lemmatize the text", default=False
    )
    parser.add_argument(
        "--conllu-format", action="store_true", help="Output in CoNLL-U-like format", default=False
    )
    return parser.parse_args()

class TextPreprocessor:
    def __init__(self, remove_stopwords, remove_punctuation, lower_text, lemmatize, conllu_format, model_type='traditional'):
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lower_text = lower_text
        self.lemmatize = lemmatize
        self.conllu_format = conllu_format
        self.model_type = model_type

        # Load stopwords if required
        if self.remove_stopwords:
            nltk.download('stopwords')
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = set()

        # Load spaCy model for tokenization and annotation
        self.nlp = spacy.load('en_core_web_sm')

    def preprocess(self, text):
        # Lowercase the text if needed
        if self.lower_text:
            text = text.lower()

        # Remove punctuation if needed
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove stopwords if needed
        if self.remove_stopwords:
            words = text.split()
            words = [word for word in words if word not in self.stopwords]
            text = ' '.join(words)

        # Lemmatize the text if needed
        if self.lemmatize:
            doc = self.nlp(text)
            text = ' '.join([token.lemma_ for token in doc])

        return text

    def process_file(self, input_file, output_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        processed_text = self.preprocess(text)

        if self.conllu_format:
            # Process text with spaCy to get annotations
            doc = self.nlp(processed_text)
            conllu_output = []
            sent_id = 0
            for sent in doc.sents:
                conllu_output.append(f"# sent_id = {sent_id}")
                for token in sent:
                    conllu_output.append(
                        f"{token.i + 1}\t{token.text}\t{token.lemma_}\t{token.pos_}\t{token.tag_}\t{token.morph}\t{token.head.i + 1 if token.head != token else 0}\t_\t_\tstart_char={token.idx}|end_char={token.idx + len(token.text)}"
                    )
                sent_id += 1
                conllu_output.append("")

            processed_text = "\n".join(conllu_output)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(processed_text)

    def process_files(self, input_dir, output_dir):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for text_file in input_path.glob("*.txt"):
            output_file = output_path / text_file.name
            self.process_file(text_file, output_file)

if __name__ == "__main__":
    args = get_args()

    # Instantiate the TextPreprocessor class with the given arguments
    preprocessor = TextPreprocessor(
        remove_stopwords=args.remove_stopwords,
        remove_punctuation=args.remove_punctuation,
        lower_text=args.lower_text,
        lemmatize=args.lemmatize,
        conllu_format=args.conllu_format
    )

    # Process the text files in the input directory and save to output directory
    preprocessor.process_files(args.input_dir, args.output_dir)



# Preprocessing options summary:
#--remove-stopwords: Removes common words like "and," "the," etc., that don't contribute much to the meaning.
#--remove-punctuation: Removes punctuation marks (e.g., periods, commas).
#--lower-text: Converts all text to lowercase.
#--lemmatize: Converts words to their base forms (e.g., "running" → "run").
#--stem: Reduces words to their stem forms (e.g., "running" → "run").
#--remove-numbers: Removes all numeric characters from the text.
#--replace-urls: Replaces URLs with a specific token like <URL>.


# Example usage:
# python preprocessing_2.py -i /Users/denisbugaenco/nlp_narrative_classification/data/PT/translated-documents -o /Users/denisbugaenco/Desktop/NLP/pt_output_CoNLL --remove-stopwords --lower-text --lemmatize --conllu-format
