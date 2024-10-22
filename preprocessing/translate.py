import deepl
import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv


class MultiLanguageTranslator:
    """Handles translation of files from multiple folders with different source languages."""

    def __init__(self, auth_key: str, target_lang: str = "EN-US"):
        """
        Initialize the translator.

        Args:
            auth_key (str): DeepL API authentication key
            target_lang (str): Target language code (default: EN-US)
        """
        self.translator = deepl.Translator(auth_key)
        self.target_lang = target_lang
        self.translation_stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
        }

    def translate_file(
        self, input_path: str, output_path: str, source_lang: str
    ) -> bool:
        """Translate a single file."""
        try:
            with open(input_path, "r", encoding="utf-8") as file:
                text = file.read()

            if not text.strip():
                print(f"Skipping empty file: {input_path}")
                self.translation_stats["skipped"] += 1
                return False

            result = self.translator.translate_text(text, target_lang=self.target_lang)

            with open(output_path, "w", encoding="utf-8") as file:
                file.write(result.text)

            self.translation_stats["successful"] += 1
            return True

        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
            self.translation_stats["failed"] += 1
            return False

    def translate_folders(
        self,
        base_input_dir: Path,
        folder_language_map: Dict[str, str],
    ):
        """
        Translate files from multiple folders with different source languages.

        Args:
            base_input_dir (Path): Base directory containing language-specific folders
            folder_language_map (Dict[str, str]): Manual mapping of folder names to language codes
        """

        # Process each folder in the base directory
        for folder_name, source_lang in folder_language_map.items():
            input_folder = base_input_dir / folder_name / "raw-documents"

            # Skip if not a directory
            if not input_folder.is_dir():
                print(f"Skipping {folder_name}. Directory not found.")
                continue

            # Create output folder
            output_folder = base_input_dir / folder_name / "translated-documents"
            Path(output_folder).mkdir(parents=True, exist_ok=True)

            print(f"\nProcessing {folder_name} (Language: {source_lang})")
            print("-" * 50)

            # Translate files
            for file_path in input_folder.glob("*.txt"):
                output_path = output_folder / file_path.name
                self.translation_stats["total_files"] += 1
                print(f"Translating: {file_path.name}")
                self.translate_file(file_path, output_path, source_lang)

        # Print final statistics
        print("\nTranslation Summary")
        print("-" * 50)
        print(f"Total files processed: {self.translation_stats['total_files']}")
        print(f"Successfully translated: {self.translation_stats['successful']}")
        print(f"Failed: {self.translation_stats['failed']}")
        print(f"Skipped: {self.translation_stats['skipped']}")


if __name__ == "__main__":
    # Configuration
    load_dotenv()
    AUTH_KEY = os.getenv("DEEPL_AUTH_KEY", None)  # DeepL API authentication key
    if not AUTH_KEY:
        raise ValueError(
            "DeepL API authentication key not found. Set DEEPL_AUTH_KEY environment variable."
        )
    DATA_PATH = (
        Path(__file__).resolve().parents[1] / "data"
    )  # Data directory containing language folders

    FOLDER_LANGUAGE_MAP = {
        "PT": "PT",  # Portuguese
        "BG": "BG",  # Bulgarian
    }

    # Initialize and run translator
    translator = MultiLanguageTranslator(AUTH_KEY)
    translator.translate_folders(
        base_input_dir=DATA_PATH,
        folder_language_map=FOLDER_LANGUAGE_MAP,
    )
