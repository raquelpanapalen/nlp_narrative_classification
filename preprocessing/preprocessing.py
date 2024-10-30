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
    def __init__(self, remove_stopwords, remove_punctuation, lower_text):
        """
        Traditional models: Remove stopwords, punctuation before tokenization.
        Deep learning models (e.g., transformers): Retain both stopwords and punctuation.
        """
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lower_text = lower_text

        self.boilerplate_patterns = {
            "read_more": r"read more|most read|read all the|continue reading",
            "un_subscribe": r"subscribe to|join our list|unsubscribe",
            "sign_up": r"sign up",
            "follow_us": r"follow us",
            "click_here": r"click here",
            "share": r"share this",
            "publish_date": r"published on",
            "copyright": r"copyright",
            "watch_video": r"watch this video|watch the interview|this video is from",
            "watch_more": r"watch more|watch the full|must watch",
            "find out more": r"find out more|continue at [a-z].[a-z]{1,3}",
            "latest_video": r"latest video",
            "terms_of_use": r"terms of use|comment policy",
            "about_the_writer": r"about the writer",
            "share_your_thougths": r"share your",
            "urls": r"https?:\/\/[^\s]+",
            "join_contribute": r"Anyone can join.\n\nAnyone can contribute.\n\n",
            "ads": r"We pay for your stories!",
            "twitter_x_timestamps": r"[—-] .+? \(@[\w]+\) (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}",
            "journalist": r"[Bb]y (?:[A-Z][a-zA-Z\-']+ [A-Z][a-zA-Z\-']+, [A-Z][a-zA-Z\s]+ News:|[A-Z][a-zA-Z\-']+ [A-Z][a-zA-Z\-']+ of [A-Z][a-zA-Z\s]+ News)",
            "top_stories": r"top stories curated by",
            "twitter_pics": r"pic.twitter.com/[a-zA-Z0-9]+",
            "make_donation": r"make a donation|donation buttons",
            "fact_check": r"fact check",
        }

        # Compile patterns for efficiency
        self.compiled_patterns = {
            name: (
                re.compile(pattern, re.IGNORECASE)
                if "[A-Z]" not in pattern
                else re.compile(pattern)
            )
            for name, pattern in self.boilerplate_patterns.items()
        }

        # Tokenizer and normalizer pipeline
        self.nlp = stanza.Pipeline("en", processors="tokenize,lemma,pos")

        # Load English stopwords from NLTK
        self.nltk_stopwords = set(stopwords.words("english"))

        # Additional common words to filter
        self.additional_stopwords = {
            "said",
            "would",
            "could",
            "also",
            "one",
            "two",
            "three",
            "first",
            "second",
            "third",
            "new",
            "time",
            "year",
            "years",
            "many",
            "much",
            "may",
            "might",
            "must",
            "like",
            "well",
        }

        self.all_stopwords = self.nltk_stopwords.union(self.additional_stopwords)

    def remove_line_if_matches(self, text, pattern):
        """Remove entire line if it contains the pattern."""
        lines = text.split("\n")
        return "\n".join(line for line in lines if not pattern.search(line))

    def remove_and_after(self, text, pattern):
        """Remove everything from the pattern match to the end of text."""
        match = pattern.search(text)
        if match:
            return text[: match.start()].strip()
        return text

    def clean_text(self, text):
        """Main preprocessing function to clean text."""
        if not text:
            return text

        # First handle patterns that should remove everything after them
        remove_after_patterns = [
            "about_the_writer",
            "share_your_thougths",
            "join_contribute",
            "ads",
        ]
        for pattern_name in remove_after_patterns:
            pattern = self.compiled_patterns[pattern_name]
            text = self.remove_and_after(text, pattern)

        # Handle patterns that should remove the entire line
        remove_line_patterns = [
            "read_more",
            "un_subscribe",
            "sign_up",
            "follow_us",
            "click_here",
            "share",
            "publish_date",
            "copyright",
            "watch_video",
            "watch_more",
            "find out more",
            "latest_video",
            "terms_of_use",
            "urls",
            "journalist",
            "top_stories",
            "make_donation",
        ]
        for pattern_name in remove_line_patterns:
            pattern = self.compiled_patterns[pattern_name]
            text = self.remove_line_if_matches(text, pattern)

        # Handle patterns that should replace with nothing
        remove_space_patterns = ["twitter_pics", "twitter_x_timestamps", "fact_check"]
        for pattern_name in remove_space_patterns:
            pattern = self.compiled_patterns[pattern_name]
            text = pattern.sub("", text)

        # Clean up extra whitespace and newlines
        text = re.sub(
            r"\n\s*\n", "\n\n", text
        )  # Replace multiple newlines with double newline
        text = re.sub(r"^\s+|\s+$", "", text)  # Remove leading/trailing whitespace

        if self.remove_punctuation:
            # Remove punctuation
            text = re.sub(r"[^\w\s]", "", text)

        if self.remove_stopwords:
            # Remove stopwords
            text = self.clean_stopwords(text)

        if self.lower_text:
            # Lowercase text
            text = text.lower()

        return text

    def tokenize(self, text):
        """Tokenize text using Stanza."""
        return self.nlp(text)

    def clean_stopwords(self, text):
        """Remove stopwords from the text."""
        filtered_tokens = []

        # tokenize with nltk word tokenizer and remove stopwords
        filtered_tokens = [
            word
            for word in nltk.word_tokenize(text)
            if word.lower() not in self.all_stopwords
        ]

        # Recombine filtered tokens into a string
        return " ".join(filtered_tokens)

    def save_file(self, doc, output_file):
        """Save to CoNLL format."""
        CoNLL.write_doc2conll(doc, output_file)


def main(
    data_dir,
    output_dir,
    remove_stopwords,
    remove_punctuation,
    lower_text,
):
    DATA_PATH = Path(data_dir)  # Directory containing text files
    OUTPUT_PATH = Path(output_dir)  # Output directory for processed files

    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    preprocessor = TextPreprocessor(remove_stopwords, remove_punctuation, lower_text)

    # Preprocess each text file
    for file_path in DATA_PATH.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"Processing {file_path.name}...")
        cleaned_text = preprocessor.clean_text(text)
        tokenized_text = preprocessor.tokenize(cleaned_text)

        # Tokenize and save to CoNLL format
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
