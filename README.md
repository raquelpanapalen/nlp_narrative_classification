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
    cd nlp_narrative_classification
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

We translated the data from Bulgarian and Portuguese to American English using DeepL, and from Hindi to American English using Google Cloud Translation, because Hindi is not aviable in DeepL. The translated documents can be found under the respective folders ([BG](data/BG) / [PT](data/PT)) in the `translated-documents` directory. The original documents are located in the `raw-documents` directory.

To translate  data, follow these steps:

1. **Create an `.env` file in the project root directory.**
2. **Add DeepL authentication key or Google Cloud Translation API key to the `.env` file:**
    ```
    DEEPL_AUTH_KEY=your_deepl_auth_key 
    ```
    or
    ```
    GOOGLE_CLOUD_API_KEY=your_google_cloud_api_key
    ```
3. **After this, use the scripts `translate.py` for translating Bulgarian and Portuguese texts, and `translatehindi.py` for translating Hindi texts from ([preprocessing](preprocessing).**
