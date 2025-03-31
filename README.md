# 🧠 RationalInvestorGPT

**RationalInvestorGPT** is a behavioral forecasting API that combines historical market data, live asset performance, financial news sentiment, and GPT-driven nudges to support better personal investment decisions.

---

## 📈 What It Does

- Uses real-time data from Yahoo Finance to assess:
  - Recent drawdown
  - Volatility (7-day and 30-day)
  - Momentum (1-week return)
- Matches your market scenario against historical patterns from the S&P 500 (via `SPY`)
- Estimates:
  - Rebound probability
  - Recovery time (to break even)
- Fetches news headlines and summarizes market sentiment using GPT
- Returns all of this via a single `/forecast` API endpoint

---

## 🔧 How to Use

### 1. Make a POST request to:
```
https://<your-render-url>.onrender.com/forecast
```

### 2. Request Body:
```json
{
  "asset_symbol": "AAPL"
}
```

### 3. Response Example:
```json
{
  "drawdown_pct": -0.1125,
  "volatility_30d": 0.1821,
  "momentum_signal": "negative",
  "average_time_to_recovery_months": 3.25,
  "historical_rebound_probability": 0.67,
  "news_sentiment_summary": "Overall sentiment is moderately positive...",
  ...
}
```

---

## 🔐 Environment Variables

Create a `.env` file with:
```bash
OPENAI_API_KEY=your_openai_key
NEWS_API_KEY=your_newsapi_key
```

> Add these to Render's environment variables tab when deploying.

---

## 🚀 Tech Stack

- Python + Flask
- Pandas, NumPy, Scikit-learn
- yFinance (for real-time financial data)
- OpenAI GPT-4 (for news sentiment)
- NewsAPI (for headline aggregation)
- Render (for deployment)

---

## 🤝 Contributing

Have ideas for new nudges, visualizations, or behavioral metrics? Open an issue or PR!

---

## 🧠 Inspiration

This project is built around the principles of:
- Reference class forecasting
- Behavioral strategy and nudges
- Bounded rationality and decision hygiene

