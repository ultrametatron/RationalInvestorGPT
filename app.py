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
    """
    Downloads 5-year daily data for SPY using yfinance.
    Then applies the reference class computations:
    rolling drawdown, volatility, momentum, etc.
    """
    global reference_df
    if reference_df is not None:
        return reference_df

    # Download 5-year daily data from yfinance
    df = yf.download("SPY", period="5y", interval="1d")  # 1d daily

    # If 'Close' missing, return empty DataFrame
    if 'Close' not in df.columns or df.empty:
        reference_df = pd.DataFrame()
        return reference_df

    # Compute rolling metrics
    df['returns'] = df['Close'].pct_change()
    df['rolling_max'] = df['Close'].rolling(window=252, min_periods=1).max()
    df['drawdown_pct'] = (df['Close'] - df['rolling_max']) / df['rolling_max']
    df['volatility_30d'] = df['returns'].rolling(30).std() * np.sqrt(252)
    df['volatility_7d'] = df['returns'].rolling(7).std() * np.sqrt(252)
    df['momentum_1w'] = df['Close'].pct_change(periods=5)
    df['forward_return_3mo'] = df['Close'].pct_change(periods=63).shift(-63)

    # Compute time to recovery
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

    # Keep only needed columns
    reference_df = df[['drawdown_pct', 'volatility_30d', 'volatility_7d', 'momentum_1w', 'rebounded',
                       'recovery_months', 'time_to_recovery']].dropna()

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
def fetch_recent_headlines(asset_symbol, languages=['en','fr', 'de', 'ja'], num_articles=5):
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
            'pageSize': num_articles,
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
            all_headlines.append((f"Error fetching {lang} headlines: {str(e)}", None, 9999, lang))

    # Sort by recency (ascending days_ago => more recent first)
    all_headlines.sort(key=lambda x: x[2])

    return all_headlines

########################################################
# Sentiment classification with GPT (recency weighting)
########################################################
def classify_news_sentiment_with_gpt(headlines_info):
    """
    Takes a list of tuples (title, published_dt, days_ago, language).
    Asks GPT to weigh recent articles more.
    """
    if not headlines_info:
        return "No headlines found."

    prompt = (
        "We have financial news headlines in multiple languages. "
        "Please translate non-English headlines into English if needed, then classify each headline as Positive, Neutral, or Negative. "
        "Weigh more recent headlines (lower 'days_ago') more heavily in forming an overall sentiment. "
        "At the end, provide a short summary line with an overall sentiment score from -1 to +1 (e.g., 'Overall score: +0.3').\n\n"
    )

    for (title, published_dt, days_ago, lang) in headlines_info:
        recency_label = "RECENT" if days_ago <= 2 else "OLDER"
        prompt += f"- [{lang.upper()}, {recency_label}, {days_ago} days ago]: {title}\n"

    messages = [
        {"role": "system", "content": "You are a multilingual financial sentiment analyst."},
        {"role": "user", "content": prompt}
    ]

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating sentiment summary: {str(e)}"

########################################################
# get_price_metrics using short-term Alpha Vantage
########################################################
@rate_limit_handler
def get_price_metrics(ticker):
    """
    Uses Alpha Vantage unadjusted daily (compact) to fetch ~2mo of data.
    Then calculates basic drawdown, volatility, momentum for the forecast endpoint.
    """
    df = fetch_alpha_data(ticker)
    if 'Close' not in df.columns or df.empty:
        raise ValueError("'Close' column missing or no data returned from Alpha Vantage.")

    df["returns"] = df["Close"].pct_change()
    current_price = df["Close"].iloc[-1]
    peak_price = df["Close"].max()
    drawdown_pct = (current_price - peak_price) / peak_price if peak_price != 0 else 0

    # 30-day volatility
    volatility_30d = (
        df["returns"].rolling(30).std().iloc[-1] * np.sqrt(252)
        if len(df) >= 30 else np.nan
    )
    # 7-day volatility
    volatility_7d = (
        df["returns"].rolling(7).std().iloc[-1] * np.sqrt(252)
        if len(df) >= 7 else np.nan
    )

    # 1-week momentum
    if len(df) >= 6:
        momentum_1w = (df["Close"].iloc[-1] - df["Close"].iloc[-6]) / df["Close"].iloc[-6]
    else:
        momentum_1w = 0.0

    return drawdown_pct, volatility_30d, volatility_7d, momentum_1w

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
    headlines_info = fetch_recent_headlines(asset_symbol, languages=['en','fr','de','ja'], num_articles=5)
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
