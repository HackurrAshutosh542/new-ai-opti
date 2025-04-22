import joblib
import pandas as pd

# Load sentiment model and TF-IDF vectorizer
model = joblib.load("sentiment_logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def analyze_sentiment(df):
    if 'feedback' not in df.columns:
        return pd.DataFrame(columns=['feedback', 'sentiment'])

    feedbacks = df['feedback'].fillna("").astype(str)
    tfidf = vectorizer.transform(feedbacks)
    preds = model.predict(tfidf)

    results = pd.DataFrame({
        'feedback': feedbacks,
        'sentiment': ['Positive' if p == 1 else 'Negative' for p in preds]
    })
    return results
