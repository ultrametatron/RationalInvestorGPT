# RationalInvestorGPT: Behavioral Forecasting API with News Sentiment and Live Price Data
# Combines Alpha Vantage for short-term, yfinance for 5-year reference class.

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify
import requests
import openai
from alpha_vantage.timeseries import TimeSeries
import yfinance as yf  # New: we add yfinance
import time
import os
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

# Simple caching dictionary & config
cache_storage = {}
CACHE_EXPIRY_SECONDS = 900  # 15 minutes

def get_cache_key(ticker, period):
    return f"{ticker}:{period}"

# We'll maintain a global reference_df for the reference class analysis
reference_df = None

########################################################
# Rate-limit handling decorator for Alpha Vantage calls
########################################################
def rate_limit_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if 'API call frequency' in str(e):
                time.sleep(15)  # Wait 15 seconds and retry
                return func(*args, **kwargs)
            else:
                raise e
    return wrapper

########################################################
# Create a single TimeSeries client for short-term data
########################################################
ts_unadj = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')

########################################################
# Unified function to fetch short-term (2mo) data from Alpha Vantage
########################################################
@rate_limit_handler
def fetch_alpha_data(ticker: str) -> pd.DataFrame:
    """
    Fetch daily UNADJUSTED data from Alpha Vantage for about 2 months.
    Avoids premium endpoints by only using 'compact' outputsize.

    Returns a DataFrame with columns: ['Open','High','Low','Close','Volume'].
    Caches results for CACHE_EXPIRY_SECONDS.
    """
    cache_key = get_cache_key(ticker, "2mo")
    cached_entry = cache_storage.get(cache_key)
    now = datetime.datetime.utcnow()

    # Check cache
    if cached_entry:
        df_cached, timestamp = cached_entry
        if (now - timestamp).total_seconds() < CACHE_EXPIRY_SECONDS:
            return df_cached

    # If not in cache or expired, fetch from Alpha Vantage
    data, meta_data = ts_unadj.get_daily(symbol=ticker, outputsize='compact')
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

    # Store in cache
    cache_storage[cache_key] = (df, now)
    return df

########################################################
# Reference Class Loading using yfinance for 5-year data
########################################################
def load_reference_class() -> pd.DataFrame:
    global reference_df
    if reference_df is not None:
        return reference_df

    # Download 5-year daily data from yfinance
    df_raw = yf.download("SPY", period="5y", interval="1d")

    # If 'Close' missing or empty, bail out
    if 'Close' not in df_raw.columns or df_raw.empty:
        reference_df = pd.DataFrame()
        return reference_df

    # Extract the 'Close' part
    close_data = df_raw['Close']

    # If close_data is 2D, pick just the first column
    if close_data.ndim == 2:
        close_data = close_data.iloc[:, 0]

    # Build a single-column DataFrame named 'Close'
    df = pd.DataFrame({'Close': close_data})

    # Rolling calculations
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

    reference_df = df[[
        'drawdown_pct', 'volatility_30d', 'volatility_7d',
        'momentum_1w', 'rebounded', 'recovery_months',
        'time_to_recovery'
    ]].dropna()

    return reference_df


########################################################
# Flask Application
########################################################
app = Flask(__name__)

@app.route('/')
def home():
    return "RationalInvestorGPT API: AV for short-term, yfinance for 5-year reference class."

########################################################
# News Headline Fetching (Multi-Language + Recency)
########################################################
def fetch_recent_headlines(asset_symbol, languages=['en'], num_articles=10):
    """
    Fetch recent headlines in multiple languages from NewsAPI,
    capturing publishedAt for recency weighting.
    """
    all_headlines = []

    for lang in languages:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': asset_symbol,
            'sortBy': 'publishedAt',
            'language': lang,
            'pageSize': 10,
            'apiKey': NEWS_API_KEY
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            for article in articles:
                title = article.get("title")
                published_at_str = article.get("publishedAt")
                if title and published_at_str:
                    published_dt = datetime.datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
                    days_ago = (datetime.datetime.utcnow() - published_dt).days
                    all_headlines.append((title, published_dt, days_ago, lang))
        except Exception as e:
            continue  # skip appending errors as headlines

    # Sort by recency (ascending days_ago => more recent first)
    all_headlines.sort(key=lambda x: x[2])

    return all_headlines

########################################################
# Sentiment classification with GPT (recency weighting)
########################################################

def fetch_recent_headlines(asset_symbol, languages=['en'], num_articles=15):
    """
    Fetch recent English headlines from NewsAPI, capturing publishedAt for recency weighting.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': asset_symbol,
        'sortBy': 'publishedAt',
        'language': 'en',
        'pageSize': num_articles,
        'apiKey': NEWS_API_KEY
    }
    all_headlines = []

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])

        for article in articles:
            title = article.get("title")
            published_at_str = article.get("publishedAt")
            if title and published_at_str:
                published_dt = datetime.datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
                days_ago = (datetime.datetime.now(datetime.timezone.utc) - published_dt).days
                all_headlines.append((title, published_dt, days_ago))

        all_headlines.sort(key=lambda x: x[2])

    except Exception as e:
        print(f"Error fetching headlines: {e}")

    return all_headlines




########################################################
# get_price_metrics using short-term Alpha Vantage
########################################################
@rate_limit_handler
def classify_news_sentiment_with_gpt(headlines_info):
    if not headlines_info:
        return "No headlines found."

    prompt = (
        "Classify each financial news headline as Positive, Neutral, or Negative. "
        "Weigh recent headlines (fewer days ago) more heavily when determining overall sentiment. "
        "Provide a concise summary at the end with an overall sentiment score from -1 (very negative) to +1 (very positive).\n\n"
    )

    for title, _, days_ago in headlines_info:
        recency_label = "RECENT" if days_ago <= 2 else "OLDER"
        prompt += f"- [{recency_label}, {days_ago} days ago]: {title}\n"

    messages = [
        {"role": "system", "content": "You are a financial sentiment analyst."},
        {"role": "user", "content": prompt}
    ]

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages
        )

        return completion.choices[0].message.content

    except Exception as e:
        print(f"Error with GPT sentiment analysis: {e}")
        return "Sentiment analysis could not be performed."

########################################################
# Main Forecast Endpoint
########################################################
@app.route('/forecast', methods=['POST'])
def forecast():
    user_input = request.get_json()
    asset_symbol = user_input.get('asset_symbol')
    if not asset_symbol:
        return jsonify({"error": "asset_symbol is required"}), 400

    try:
        drawdown, volatility_30d, volatility_7d, momentum_1w = get_price_metrics(asset_symbol)
    except Exception as e:
        print(f"Error in get_price_metrics: {e}")
        return jsonify({"error": f"Failed to compute price metrics: {str(e)}"}), 500

    # Load reference class (5 years) from yfinance + do nearest neighbors
    df = load_reference_class()
    if df.empty or 'drawdown_pct' not in df.columns:
        return jsonify({"error": "Reference class data could not be loaded. Please try again later."}), 503

    X = df[["drawdown_pct", "volatility_30d"]]
    nn_model = NearestNeighbors(n_neighbors=3)
    nn_model.fit(X)
    distances, indices = nn_model.kneighbors([[drawdown, volatility_30d]])
    similar_cases = df.iloc[indices[0]]

    avg_recovery = similar_cases['recovery_months'].mean()
    rebound_prob = similar_cases['rebounded'].mean()
    avg_time_to_recovery = similar_cases['time_to_recovery'].mean()

    vol_spike_ratio = (volatility_7d / volatility_30d) if (volatility_30d and not np.isnan(volatility_30d)) else None
    momentum_signal = 'positive' if momentum_1w > 0 else 'negative' if momentum_1w < 0 else 'neutral'

    # Summarize recent news
    headlines_info = fetch_recent_headlines(asset_symbol, languages=['en'], num_articles=15)
    sentiment_summary = classify_news_sentiment_with_gpt(headlines_info)

    response = {
        'matched_cases': similar_cases.to_dict(orient='records'),
        'average_recovery_time_months': round(avg_recovery, 2) if not np.isnan(avg_recovery) else None,
        'historical_rebound_probability': round(rebound_prob, 2) if not np.isnan(rebound_prob) else None,
        'drawdown_pct': round(drawdown, 4) if not np.isnan(drawdown) else None,
        'volatility_30d': round(volatility_30d, 4) if volatility_30d is not None else None,
        'volatility_7d': round(volatility_7d, 4) if volatility_7d is not None else None,
        'momentum_1w': round(momentum_1w, 4),
        'momentum_signal': momentum_signal,
        'volatility_spike_ratio': round(vol_spike_ratio, 2) if vol_spike_ratio else None,
        'news_sentiment_summary': sentiment_summary,
        'average_time_to_recovery_months': round(avg_time_to_recovery, 2) if not np.isnan(avg_time_to_recovery) else None
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
