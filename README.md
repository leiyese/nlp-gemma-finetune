# NLP Sentiment Classification with Financial PhraseBank

This project explores sentiment classification on the Financial PhraseBank dataset using multiple small language models (SLMs), including Gemma, BERT, and GPT. The goal is to fine-tune and compare these models for financial sentiment analysis, leveraging free-tier resources such as Google Colab.

## Dataset

We use the [Financial PhraseBank](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis) dataset, a widely used benchmark for financial sentiment analysis. It contains financial news sentences labeled as **positive**, **neutral**, or **negative**.

## Project Structure

- `data/` — Contains the Financial PhraseBank CSV file and data splits.
- `notebooks/` — Jupyter notebooks for exploratory data analysis (EDA), data preparation, and model training.
- `scripts/` — Utility scripts for preprocessing, training, and evaluation.
- `models/` — Saved model checkpoints and results.

## Workflow

1. **Data Preparation & EDA**
   - Clean and explore the dataset.
   - Visualize class distributions and sentence lengths.
   - Split data into training, validation, and test sets (stratified by label).

2. **Model Training (in Google Colab)**
   - For each model (Gemma, BERT, GPT):
     - Load the preprocessed data from Google Drive.
     - Fine-tune the model on the training set.
     - Evaluate on validation and test sets.
     - Save results and model checkpoints.

3. **Comparison & Analysis**
   - Compare model performance using accuracy, F1-score, and other metrics.
   - Summarize findings in a results notebook or markdown file.

## Models Used

- **gemma-2-2b**: A small language model suitable for free-tier Colab usage.
- **BERT**: Pretrained transformer model (e.g., DistilBERT or FinBERT).
- **GPT**: Small GPT variant (e.g., GPT-2 or similar SLM).

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- torch
- transformers (for BERT, GPT)
- Gemma library (huggingface transformers)

Install requirements with:

```bash
pip install -r requirements.txt
```

## How to Run

1. Run EDA and data preparation in the provided notebook.
2. Upload data splits to Google Drive.
3. Open the model training notebooks in Google Colab and follow the instructions for each model.

## References
- [Financial PhraseBank Dataset](https://www.research.tuni.fi/finance/datasets/financial-phrasebank/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Gemma Model Documentation](https://ai.google.dev/gemma)
