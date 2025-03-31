# RationalInvestorGPT: Behavioral Forecasting API with News Sentiment and Live Price Data

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify
import requests
import openai
import yfinance as yf
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Load reference class from historical data

def load_reference_class():
    df = yf.download('SPY', period='5y', interval='1d')
    if 'Adj Close' not in df.columns:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)
    df['returns'] = df['Adj Close'].pct_change()
    df['rolling_max'] = df['Adj Close'].rolling(window=252, min_periods=1).max()
    df['drawdown_pct'] = (df['Adj Close'] - df['rolling_max']) / df['rolling_max']
    df['volatility_30d'] = df['returns'].rolling(30).std() * np.sqrt(252)
    df['volatility_7d'] = df['returns'].rolling(7).std() * np.sqrt(252)
    df['momentum_1w'] = df['Adj Close'].pct_change(periods=5)
    df['forward_return_3mo'] = df['Adj Close'].pct_change(periods=63).shift(-63)

    df['time_to_recovery'] = np.nan
    for i in range(len(df)):
        current_price = df['Adj Close'].iloc[i]
        for j in range(i + 1, len(df)):
            if df['Adj Close'].iloc[j] >= current_price:
                df.at[df.index[i], 'time_to_recovery'] = (df.index[j] - df.index[i]).days / 30.0
                break

    df = df.dropna(subset=['forward_return_3mo', 'time_to_recovery'])
    df['rebounded'] = (df['forward_return_3mo'] > 0).astype(int)
    df['recovery_months'] = np.where(df['rebounded'] == 1, 3, 6)

    return df[['drawdown_pct', 'volatility_30d', 'volatility_7d', 'momentum_1w', 'rebounded', 'recovery_months', 'time_to_recovery']].dropna()

# Load expanded historical dataset
df = load_reference_class()

# Fit Nearest Neighbors model using drawdown and 30-day volatility
X = df[['drawdown_pct', 'volatility_30d']]
nn_model = NearestNeighbors(n_neighbors=3)
nn_model.fit(X)

# Flask API setup
app = Flask(__name__)

@app.route('/')
def home():
    return "RationalInvestorGPT API is live. Use POST /forecast with an asset_symbol."


def fetch_recent_headlines(asset_symbol, num_articles=5):
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': asset_symbol,
        'sortBy': 'publishedAt',
        'language': 'en',
        'pageSize': num_articles,
        'apiKey': NEWS_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        headlines = [article["title"] for article in articles if article.get("title")]
        return headlines
    except Exception as e:
        return [f"Error fetching headlines: {str(e)}"]


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

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating sentiment summary: {str(e)}"


def get_price_metrics(ticker):
    try:
        df = yf.download(ticker, period="2mo", interval="1d")
        if df.empty:
            raise ValueError("No data returned from yfinance.")
        df["returns"] = df["Adj Close"].pct_change()

        current_price = df["Adj Close"].iloc[-1]
        peak_price = df["Adj Close"].max()
        drawdown_pct = (current_price - peak_price) / peak_price

        volatility_30d = df["returns"].rolling(30).std().iloc[-1] * np.sqrt(252)
        volatility_7d = df["returns"].rolling(7).std().iloc[-1] * np.sqrt(252)
        momentum_1w = (df["Adj Close"].iloc[-1] - df["Adj Close"].iloc[-6]) / df["Adj Close"].iloc[-6]

        return drawdown_pct, volatility_30d, volatility_7d, momentum_1w
    except Exception as e:
        raise RuntimeError(f"Failed to compute price metrics: {str(e)}")


@app.route('/forecast', methods=['POST'])
def forecast():
    user_input = request.get_json()
    asset_symbol = user_input.get('asset_symbol')

    if not asset_symbol:
        return jsonify({"error": "asset_symbol is required"}), 400

    drawdown, volatility, volatility_7d, momentum_1w = get_price_metrics(asset_symbol)

    distances, indices = nn_model.kneighbors([[drawdown, volatility]])
    similar_cases = df.iloc[indices[0]]

    avg_recovery = similar_cases['recovery_months'].mean()
    rebound_prob = similar_cases['rebounded'].mean()
    avg_time_to_recovery = similar_cases['time_to_recovery'].mean()

    vol_spike_ratio = volatility_7d / volatility if volatility else None
    momentum_signal = 'positive' if momentum_1w > 0 else 'negative' if momentum_1w < 0 else 'neutral'

    headlines = fetch_recent_headlines(asset_symbol)
    sentiment_summary = classify_news_sentiment_with_gpt(headlines)

    response = {
        'matched_cases': similar_cases.to_dict(orient='records'),
        'average_recovery_time_months': round(avg_recovery, 2),
        'historical_rebound_probability': round(rebound_prob, 2),
        'drawdown_pct': round(drawdown, 4),
        'volatility_30d': round(volatility, 4),
        'volatility_7d': round(volatility_7d, 4),
        'momentum_1w': round(momentum_1w, 4),
        'momentum_signal': momentum_signal,
        'volatility_spike_ratio': round(vol_spike_ratio, 2) if vol_spike_ratio is not None else None,
        'news_sentiment_summary': sentiment_summary,
        'average_time_to_recovery_months': round(avg_time_to_recovery, 2)
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
