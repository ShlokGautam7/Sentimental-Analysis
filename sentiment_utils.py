# sentiment_utils.py
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

# download once
nltk.download("vader_lexicon", quiet=True)

sia = SentimentIntensityAnalyzer()

def add_vader_sentiment(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Add columns:
      - sentiment_compound: float in [-1,1]
      - sentiment_label: POSITIVE / NEUTRAL / NEGATIVE
    Returns a new DataFrame copy.
    """
    df = df.copy()
    if text_col not in df.columns:
        df["text"] = ""
    else:
        df["text"] = df[text_col].fillna("").astype(str)

    def safe_score(t):
        if not t or str(t).strip() == "":
            return 0.0
        return sia.polarity_scores(str(t))["compound"]

    df["sentiment_compound"] = df["text"].apply(safe_score)
    def label(v):
        if v >= 0.05:
            return "POSITIVE"
        if v <= -0.05:
            return "NEGATIVE"
        return "NEUTRAL"
    df["sentiment_label"] = df["sentiment_compound"].apply(label)
    return df
