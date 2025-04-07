# RationalInvestorGPT: Behavioral Forecasting API integrating Alpha Vantage (short-term) and yfinance (reference class)

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, jsonify
import requests
import openai
from alpha_vantage.timeseries import TimeSeries
import time
import os
import datetime
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables
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

# Initialize Alpha Vantage
ts = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')

# Reference class DataFrame
reference_df = None

# Rate-limit handler decorator
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

@rate_limit_handler
def fetch_alpha_data(ticker: str, period: str = '2mo') -> pd.DataFrame:
    outputsize = 'compact' if period == '2mo' else 'full'
    cache_key = get_cache_key(ticker, period)
    cached_entry = cache_storage.get(cache_key)
    now = datetime.datetime.utcnow()

    if cached_entry:
        df_cached, timestamp = cached_entry
        if (now - timestamp).total_seconds() < CACHE_EXPIRY_SECONDS:
            return df_cached

    data, meta_data = ts.get_daily(symbol=ticker, outputsize=outputsize)
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

def load_reference_class(asset_symbol: str = 'SPY') -> pd.DataFrame:
    global reference_df
    if reference_df is not None and asset_symbol == 'SPY':
        return reference_df

    df = yf.download(asset_symbol, period='5y', interval='1d')

    if df.empty or 'Close' not in df.columns:
        return pd.DataFrame()

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

    df.dropna(subset=['forward_return_3mo', 'time_to_recovery'], inplace=True)
    df['rebounded'] = (df['forward_return_3mo'] > 0).astype(int)
    df['recovery_months'] = np.where(df['rebounded'] == 1, 3, 6)

    final_df = df[['drawdown_pct', 'volatility_30d', 'volatility_7d', 'momentum_1w',
                   'rebounded', 'recovery_months', 'time_to_recovery']].dropna()

    if asset_symbol == 'SPY':
        reference_df = final_df

    return final_df

# Flask Application
app = Flask(__name__)

@app.route('/')
def home():
    return "RationalInvestorGPT API is live. Use POST /forecast with an asset_symbol."


########################################################
# Flask Application
########################################################
app = Flask(__name__)

@app.route('/')
def home():
    return "RationalInvestorGPT API (Alpha Vantage) is live. Use POST /forecast with an asset_symbol."

########################################################
# News Headline Fetching
########################################################
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

########################################################
# Sentiment classification with GPT
########################################################
def classify_news_sentiment_with_gpt(headlines):
    prompt = (
        "Classify the following financial news headlines as Positive, Neutral, or Negative. "
        "Then return a brief summary of sentiment by labeling each headline. "
        "At the end, include an overall sentiment score from -1 to +1 on its own line (e.g. Overall score: +0.3). "
        "Format the response clearly with bullet points and consistent spacing.\n\n"
    )
    for h in headlines:
        prompt += f"- {h}\n"

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
        return f"Error generating sentiment summary: {str(e)}"

########################################################
# Replace yfinance-based get_price_metrics with Alpha Vantage
########################################################
@rate_limit_handler
def get_price_metrics(ticker):
    # Fetch ~2 months of daily data from Alpha Vantage
    df = fetch_alpha_data(ticker, period="2mo")
    if 'Close' not in df.columns or df.empty:
        raise ValueError("'Close' column missing or no data returned from Alpha Vantage.")

    df["returns"] = df["Close"].pct_change()

    current_price = df["Close"].iloc[-1]
    peak_price = df["Close"].max()
    drawdown_pct = (current_price - peak_price) / peak_price if peak_price != 0 else 0

    # 30-day volatility
    volatility_30d = df["returns"].rolling(30).std().iloc[-1] * np.sqrt(252) if len(df) >= 30 else np.nan
    # 7-day volatility
    volatility_7d = df["returns"].rolling(7).std().iloc[-1] * np.sqrt(252) if len(df) >= 7 else np.nan

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

    # Load reference class and fit nearest neighbors model
    df = load_reference_class(asset_symbol)
    if df.empty or 'drawdown_pct' not in df.columns:
        return jsonify({"error": "Reference class data could not be loaded. Please try again later."}), 503

    X = df[['drawdown_pct', 'volatility_30d']]
    nn_model = NearestNeighbors(n_neighbors=3)
    nn_model.fit(X)
    distances, indices = nn_model.kneighbors([[drawdown, volatility_30d]])
    similar_cases = df.iloc[indices[0]]

    avg_recovery = similar_cases['recovery_months'].mean()
    rebound_prob = similar_cases['rebounded'].mean()
    avg_time_to_recovery = similar_cases['time_to_recovery'].mean()

    vol_spike_ratio = (volatility_7d / volatility_30d) if (volatility_30d and not np.isnan(volatility_30d)) else None
    momentum_signal = 'positive' if momentum_1w > 0 else 'negative' if momentum_1w < 0 else 'neutral'

    headlines = fetch_recent_headlines(asset_symbol)
    sentiment_summary = classify_news_sentiment_with_gpt(headlines)

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
