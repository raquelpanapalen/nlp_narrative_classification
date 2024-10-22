# Narrative Classification Project

This project is about narrative classification for the NLP course at TU Wien.

## Getting Started

To work with the code, follow these steps:

1. **Clone the repository locally:**
    ```bash
    git clone git@github.com:raquelpanapalen/nlp_narrative_classification.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd narrative_classification
    ```

3. **Create a virtual environment:**
    ```bash
    python -m venv myenv
    ```

4. **Activate the virtual environment:**
    - On Windows:
      ```bash
      .\myenv\Scripts\activate
      ```
    - On macOS and Linux:
      ```bash
      source myenv/bin/activate
      ```

5. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

You are now ready to start working on the project!


## Data Translation

We translated the data from Bulgarian and Portuguese to American English using DeepL. The translated documents can be found under the respective folders ([BG](data/BG) / [PT](data/PT)) in the `translated-documents` directory. The original documents are located in the `raw-documents` directory.

To translate more data, follow these steps:

1. **Create an `.env` file in the project root directory.**
2. **Add your DeepL authentication key to the `.env` file:**
    ```
    DEEPL_AUTH_KEY=your_deepl_auth_key
    ```

You are now ready to translate additional data using DeepL!