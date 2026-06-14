# news-topic-classifier
# Task 1: News Topic Classifier Using BERT

## Objective

Fine-tune a transformer model to classify news text into topic categories using the AG News dataset.

The project uses `bert-base-uncased` from Hugging Face Transformers and evaluates the trained model using accuracy and F1-score. It also includes a Gradio interface for live topic prediction.

## Dataset

Dataset used:

- AG News Dataset from Hugging Face Datasets
- Notebook path used: `fancyzhx/ag_news`

Classes:

- World
- Sports
- Business
- Sci/Tech

The dataset contains news text samples and their corresponding topic labels.

## Notebook File

Use this notebook in Google Colab:

`bert_news_topic_classifier_colab.ipynb`

Recommended runtime:

- Python 3
- GPU enabled

In Colab, go to:

`Runtime > Change runtime type > Hardware accelerator > GPU`

## Main Steps

1. Install required libraries:
   - `transformers`
   - `datasets`
   - `accelerate`
   - `gradio`
   - `scikit-learn`
   - `seaborn`

2. Load the AG News dataset using Hugging Face Datasets.

   The notebook uses:

   ```python
   load_dataset("fancyzhx/ag_news")
   ```

   This avoids a Colab/Hugging Face Hub URI error that can happen with the older shortcut:

   ```python
   load_dataset("ag_news")
   ```

3. Tokenize the news text using the BERT tokenizer.

4. Fine-tune `bert-base-uncased` for sequence classification.

5. Evaluate the model using:
   - Accuracy
   - Macro F1-score
   - Weighted F1-score
   - Classification report
   - Confusion matrix

6. Save the trained model.

7. Deploy the model with a Gradio interface for live prediction.

## Model

Base model:

`bert-base-uncased`

Task type:

Multi-class text classification

Number of output labels:

`4`

Label mapping:

| Label ID | Topic |
|---|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

## How to Run

1. Open `bert_news_topic_classifier_colab.ipynb` in Google Colab.

2. Run the installation cell.

3. Run all cells from top to bottom.

4. Keep `FAST_RUN = True` if you want faster training for testing.

5. Set `FAST_RUN = False` if you want to train on the full AG News dataset.

6. After training, run the Gradio deployment cell.

7. Enter any news headline or short news article text to get a topic prediction.

## Important Settings

Inside the notebook:

```python
FAST_RUN = True
MAX_TRAIN_SAMPLES = 8000
MAX_TEST_SAMPLES = 2000
MAX_LENGTH = 128
NUM_EPOCHS = 2
BATCH_SIZE = 16
```

For better final results, you can use:

```python
FAST_RUN = False
NUM_EPOCHS = 3
BATCH_SIZE = 16
```

Training on the full dataset may take longer, especially without GPU.

## Evaluation Metrics

The notebook reports:

### Accuracy

Measures the percentage of correctly classified news samples.

### Macro F1-score

Computes F1-score independently for each class and then averages them equally.

### Weighted F1-score

Computes F1-score for each class and averages them according to class frequency.

## Gradio Deployment

The notebook includes a Gradio interface:

```python
demo.launch(share=True)
```

This creates a public temporary Gradio link in Colab. You can enter a news headline and get predicted topic probabilities.

Example input:

```text
The stock market rose after strong quarterly earnings from major technology companies.
```

Expected topic:

```text
Business
```

## Output Files

After running the notebook, the following outputs are created inside Colab:

- `/content/bert_ag_news_classifier`
- `/content/app.py`

The model folder contains the fine-tuned BERT model and tokenizer files.

The `app.py` file is a simple standalone Gradio app that can be used with the saved model.

## Skills Demonstrated

- Natural Language Processing using Transformers
- Transfer learning with BERT
- Fine-tuning a pretrained language model
- Text tokenization and preprocessing
- Multi-class classification
- Accuracy and F1-score evaluation
- Lightweight deployment using Gradio

## Project Summary

This task builds a complete BERT-based news classifier. The AG News dataset is loaded from Hugging Face, tokenized with the BERT tokenizer, and used to fine-tune `bert-base-uncased`. The trained model is evaluated using accuracy and F1-score, then deployed through a Gradio interface for live text classification.
