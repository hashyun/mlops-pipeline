import numpy as np

from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

## DistilBERT
model_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
distil_bert_model = pipeline(task="sentiment-analysis", model=model_checkpoint)

## FinBERT
model_checkpoint = "yiyanghkust/finbert-tone"
finbert = BertForSequenceClassification.from_pretrained(model_checkpoint,num_labels=3)
tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
finbert_model = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)


def get_sentiment(df, text_col, sentiment_model):
    df[f'sentiment_{text_col}'] = df[text_col].apply(lambda x: sentiment_model(x) if len(x) > 0 else [{}])

    df[f'sentiment_{text_col}_label'] = df[f'sentiment_{text_col}'].apply(lambda x: x[0].get('label', np.nan))
    df[f'sentiment_{text_col}_score'] = df[f'sentiment_{text_col}'].apply(lambda x: x[0].get('score', np.nan))
    df.drop(columns = f'sentiment_{text_col}', inplace = True)

    return df 