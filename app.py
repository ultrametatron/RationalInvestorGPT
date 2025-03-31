# RationalInvestorGPT: Behavioral Forecasting API with News Sentiment

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify
import requests
import openai

# API Keys (replace with your actual keys)
NEWS_API_KEY = "your_newsapi_key_here"
OPENAI_API_KEY = "your_openai_key_here"
openai.api_key = OPENAI_API_KEY

# Sample historical market data (reference class)
data = {
    'drawdown_pct': [-12, -8, -15, -10, -20, -5],
    'volatility': [0.18, 0.12, 0.25, 0.15, 0.3, 0.1],
    'recovery_months': [3, 2, 5, 3, 8, 1],
    'rebounded': [1, 1, 1, 1, 0, 1],
    'momentum_1w': [-0.03, 0.02, -0.05, -0.02, -0.06, 0.01],
    'volatility_7d': [0.22, 0.14, 0.28, 0.17, 0.35, 0.12],
    'volatility_30d': [0.18, 0.12, 0.25, 0.15, 0.3, 0.1]
}
df = pd.DataFrame(data)

# Fit Nearest Neighbors model using drawdown and 30-day volatility
X = df[['drawdown_pct', 'volatility']]
nn_model = NearestNeighbors(n_neighbors=3)
nn_model.fit(X)

# Flask API setup
app = Flask(__name__)

@app.route('/')
def home():
    return "RationalInvestorGPT API is live. Use POST /forecast with drawdown_pct, volatility, and optional asset_symbol."


def fetch_recent_headlines(asset_symbol, num_articles=5):
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': asset_symbol,
        'sortBy': 'publishedAt',
        'language': 'en',
        'pageSize': num_articles,
        'apiKey': NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    articles = response.json().get("articles", [])
    headlines = [article["title"] for article in articles if article.get("title")]
    return headlines


def classify_news_sentiment_with_gpt(headlines):
    prompt = (
        "Classify the following financial news headlines as Positive, Negative, or Neutral. "
        "Then return an overall sentiment score from -1 (very negative) to +1 (very positive).\n\n"
    )
    for h in headlines:
        prompt += f"- {h}\n"

    messages = [
        {"role": "system", "content": "You are a financial sentiment analyst."},
        {"role": "user", "content": prompt}
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )

    return completion["choices"][0]["message"]["content"]


@app.route('/forecast', methods=['POST'])
def forecast():
    user_input = request.get_json()
    drawdown = user_input.get('drawdown_pct')
    volatility = user_input.get('volatility')
    volatility_7d = user_input.get('volatility_7d')
    momentum_1w = user_input.get('momentum_1w')
    asset_symbol = user_input.get('asset_symbol')

    # Nearest neighbor matching
    distances, indices = nn_model.kneighbors([[drawdown, volatility]])
    similar_cases = df.iloc[indices[0]]

    # Forecast stats
    avg_recovery = similar_cases['recovery_months'].mean()
    rebound_prob = similar_cases['rebounded'].mean()

    # Derived signals
    vol_spike_ratio = volatility_7d / volatility if volatility else None
    momentum_signal = 'positive' if momentum_1w > 0 else 'negative' if momentum_1w < 0 else 'neutral'

    # News sentiment
    sentiment_summary = None
    if asset_symbol:
        headlines = fetch_recent_headlines(asset_symbol)
        sentiment_summary = classify_news_sentiment_with_gpt(headlines)

    response = {
        'matched_cases': similar_cases.to_dict(orient='records'),
        'average_recovery_time_months': round(avg_recovery, 2),
        'historical_rebound_probability': round(rebound_prob, 2),
        'momentum_signal': momentum_signal,
        'volatility_spike_ratio': round(vol_spike_ratio, 2) if vol_spike_ratio is not None else None,
        'news_sentiment_summary': sentiment_summary
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
