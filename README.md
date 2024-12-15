# Narrative Classification Project, group 8 "Word Wizards"

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


## Project Structure

The project follows the following structure in order to facilitate data preprocessing, modeling, and analysis:

- **`data/`**: Contains raw documents, processed documents, annotations for each language and JSON files for narratives and subnarratives labels.
  - `raw-documents/`: Original documents.
  - `traditional or deeplearning processed-documents/`: Preprocessed documents for traditional and deep learning models.
  - `subtask-2-annotations/`: Annotations for each txt file.
  - `labels/`JSON files for narratives and subnarratives used in modeling.

- **`datasets/`**: contains the dataset classes.
  - `NarrativeDataset.py`: Dataset class for traditional models.
  - `DeepLNarrativeDataset.py`: Dataset class for deep learning models.

- **`model performance analysis/`**: Includes notebooks for analyzing model results.
  - `Quantitative analysis.ipynb`: Notebook for quantitative evaluation of model performance.
  - `Qualitative analysis.ipynb`: Notebook for qualitative analysis of model predictions.

- **`models/`**: Contains the implementation of baseline models.
  - `lstm.py`: Implementation of the LSTM model.
  - `svm.py`: Implementation of the SVM model.
  - `transformer.py`: Implementation of the Transformer model.

- **`notebooks/`**: Exploratory notebooks for data analysis and visualization.

- **`preprocessing/`**: Scripts for cleaning and processing data.

- **`trainer/`**: Includes helper files for training and evaluation.
  - `loss.py`: Custom loss functions.
  - `scheduler.py`: Learning rate schedulers.
  - `trainer.py`: Main trainer class.

- **`base_config.py`**: Configuration file for setting up and running baseline experiments.

- **`baselines.py`**: Script for training and evaluating baseline models.

- **`baselines_bert.py`**: Incorporates predictions from BERT-based model, using only inference. However, the script is still a work in progress.

- **`predictions/`**: Stores predictions generated by baseline models for each topic.

---


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
3. **After this, use the scripts `translate.py` for translating Bulgarian and Portuguese texts, and `translatehindi.py` for translating Hindi texts from [preprocessing](preprocessing).**

## Data Preprocessing

Scripts `preprocessing.py`, `preprocessing_bg.py`, etc. from [preprocessing](preprocessing) prepares data for further analysis or modeling by applying transformations such as text normalization, tokenization, and filtering irrelevant information.

For preprocessing the Hindi-translated texts, the script `preprocessing_hindi.py` is used to clean and process the content before further analysis. This, again, includes tasks like removing unwanted stopwords (including custom stopwords for HI translated text), punctuation, and performing tokenization using Stanza's NLP pipeline.

To run the scripts:
```bash
    python preprocessing/script_name.py -i <input_directory> -o <output_directory> --remove-stopwords --remove-punctuation --lower-text
```

#### Parameters
- **`-i <input_directory>`**: Specifies the path to the input directory containing the text files.  
  **Example:** `data/BG/translated-documents`
- **`-o <output_directory>`**: Specifies the path to the output directory where the processed files will be saved.  
  **Example:** `data/BG`

#### Options
- **`--remove-stopwords`**: Removes common stopwords from the text to enhance the quality of text data.
- **`--remove-punctuation`**: Removes punctuation, which is often unnecessary for text processing.
- **`--lower-text`**: Converts all text to lowercase to ensure uniform formatting, beneficial for case-insensitive analysis.
  
The traditional-processed-documents folder contains text files preprocessed for traditional models with stopwords,vpunctuation removed, etc. , while the deeplearning-processed-documents folder contains the original text files retained for deep learning model input


## Baseline Results

Our baseline results for English - Subtask 2 from the SemEval 2025 leaderboard are as follows:

| Rank | Model                   | F1 Macro Coarse | F1 Std. Dev. Coarse | F1 Macro Fine | F1 Std. Dev. Fine |
|------|-------------------------|-----------------|---------------------|---------------|-------------------|
| 1    | LSTM                    | 0.19400         | 0.21300             | 0.26800       | 0.44300           |
| 2    | LSTM EN+BG (translated) | 0.20900         | 0.23100             | 0.19400       | 0.36100           |
| 3    | LSTM EN+HI (translated) | 0.22600	       | 0.27600	         | 0.06700	     | 0.18100           |
| 4    | SVM                     | 0.22400         | 0.25000             | 0.19900       | 0.32000           |
| 5    | Basic Transformer       | 0.22300         | 0.21100             | 0.16000       | 0.20700           |

These results provide a benchmark for evaluating the performance of narrative classification models on the given dataset.

### Estimation of results:

The **LSTM** model performed the best, with the highest F1 macro coarse (0.194) and fine (0.268) scores, though its fine-grained performance showed variability. 

The additional data from the EN+HI (translated) and EN+BG (translated) models did not significantly improve performance due to the unbalanced nature of the data across languages (comparison in [data_comparison.ipynb](notebooks/data_comparison.ipynb)).This imbalance hinders the generalization of the model, affecting its ability to perform consistently.

The **SVM** and **Basic Transformer** models ranked lower, with the Transformer showing the weakest fine-grained performance.

Quantitative and qualitative analysis can be found [here](https://github.com/raquelpanapalen/nlp_narrative_classification/tree/master/model%20performance%20analysis).
