# RationalInvestorGPT: Behavioral Forecasting API with News Sentiment and Live Price Data
# Combines Alpha Vantage for short-term, yfinance for 5-year reference class.

# Library Imports
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify
import requests
import openai
from alpha_vantage.timeseries import TimeSeries
import yfinance as yf
import time
import os
import datetime
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

# Caching Configuration
cache_storage = {}
CACHE_EXPIRY_SECONDS = 900

def get_cache_key(ticker, period):
    return f"{ticker}:{period}"

# Global DataFrame for Reference Class
reference_df = None

# Rate Limiting Decorator
def rate_limit_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if 'API call frequency' in str(e):
                time.sleep(15)
                return func(*args, **kwargs)
            else:
                raise e
    return wrapper

# Alpha Vantage Client Initialization
ts_unadj = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')

# Fetch Alpha Vantage Data (Short-Term)
@rate_limit_handler
def fetch_alpha_data(ticker: str) -> pd.DataFrame:
    cache_key = get_cache_key(ticker, "2mo")
    cached_entry = cache_storage.get(cache_key)
    now = datetime.datetime.utcnow()

    if cached_entry:
        df_cached, timestamp = cached_entry
        if (now - timestamp).total_seconds() < CACHE_EXPIRY_SECONDS:
            return df_cached

    data, _ = ts_unadj.get_daily(symbol=ticker, outputsize='compact')
    df = data.copy()
    df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume',
    }, inplace=True)

    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    cache_storage[cache_key] = (df, now)
    return df

# Load Reference Class Data (Long-Term)
def load_reference_class() -> pd.DataFrame:
    global reference_df
    if reference_df is not None:
        return reference_df

    df_raw = yf.download("SPY", period="5y", interval="1d")

    if 'Close' not in df_raw.columns or df_raw.empty:
        reference_df = pd.DataFrame()
        return reference_df

    close_data = df_raw['Close']

    if close_data.ndim == 2:
        close_data = close_data.iloc[:, 0]

    df = pd.DataFrame({'Close': close_data})

    df['returns'] = df['Close'].pct_change()
    df['rolling_max'] = df['Close'].rolling(window=252, min_periods=1).max()
    df['drawdown_pct'] = (df['Close'] - df['rolling_max']) / df['rolling_max']
    df['volatility_30d'] = df['returns'].rolling(30).std() * np.sqrt(252)
    df['volatility_7d'] = df['returns'].rolling(7).std() * np.sqrt(252)
    df['momentum_1w'] = df['Close'].pct_change(periods=5)
    df['forward_return_3mo'] = df['Close'].pct_change(periods=63).shift(-63)

    df['time_to_recovery'] = np.nan
    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        for j in range(i + 1, len(df)):
            if df['Close'].iloc[j] >= current_price:
                df.at[df.index[i], 'time_to_recovery'] = (df.index[j] - df.index[i]).days / 30.0
                break

    df = df.dropna(subset=['forward_return_3mo', 'time_to_recovery'])
    df['rebounded'] = (df['forward_return_3mo'] > 0).astype(int)
    df['recovery_months'] = np.where(df['rebounded'] == 1, 3, 6)

    reference_df = df[['drawdown_pct', 'volatility_30d', 'volatility_7d',
                       'momentum_1w', 'rebounded', 'recovery_months',
                       'time_to_recovery']].dropna()

    return reference_df

# Calculate Price Metrics
@rate_limit_handler
def get_price_metrics(ticker):
    df = fetch_alpha_data(ticker)
    df["returns"] = df["Close"].pct_change()
    current_price = df["Close"].iloc[-1]
    peak_price = df["Close"].max()
    drawdown_pct = (current_price - peak_price) / peak_price if peak_price != 0 else 0
    volatility_30d = df["returns"].rolling(30).std().iloc[-1] * np.sqrt(252)
    volatility_7d = df["returns"].rolling(7).std().iloc[-1] * np.sqrt(252)
    momentum_1w = df["Close"].pct_change(periods=5).iloc[-1]
    return drawdown_pct, volatility_30d, volatility_7d, momentum_1w

# Flask Application Initialization
app = Flask(__name__)

@app.route('/')
def home():
    return "RationalInvestorGPT API: AV for short-term, yfinance for 5-year reference class."

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    asset_symbol = data.get('asset_symbol')

    if not asset_symbol:
        return jsonify({"error": "Asset symbol is required"}), 400

    drawdown_pct, volatility_30d, volatility_7d, momentum_1w = get_price_metrics(asset_symbol)

    headlines_info = fetch_recent_headlines(asset_symbol, languages=['en'], num_articles=10)
    news_sentiment_summary = classify_news_sentiment_with_gpt(headlines_info)

    response = {
        "asset_symbol": asset_symbol,
        "drawdown_pct": drawdown_pct,
        "volatility_30d": volatility_30d,
        "volatility_7d": volatility_7d,
        "momentum_1w": momentum_1w,
        "news_sentiment_summary": news_sentiment_summary
    }

    return jsonify(response)

# Fetch Recent Headlines (English Only)
def fetch_recent_headlines(asset_symbol, languages=['en'], num_articles=10):
    all_headlines = []

    for lang in languages:
        url = "https://newsapi.org/v2/everything"
        params = {'q': asset_symbol, 'sortBy': 'publishedAt', 'language': lang, 'pageSize': num_articles, 'apiKey': NEWS_API_KEY}
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        for article in articles:
            title = article.get("title")
            published_at_str = article.get("publishedAt")
            if title and published_at_str:
                published_dt = datetime.datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
                days_ago = (datetime.datetime.now(datetime.timezone.utc) - published_dt).days
                all_headlines.append((title, published_dt, days_ago, lang))

    all_headlines.sort(key=lambda x: x[2])
    return all_headlines

# Run Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
