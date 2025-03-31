# 🧠 RationalInvestorGPT (Rewritten)

**RationalInvestorGPT** is a behavioral forecasting API that leverages reference class forecasting, real-time market data, and subtle nudges drawn from behavioral economics. It’s designed to help users overcome biases (like loss aversion or recency bias) and make more reflective, data-driven investment decisions.

---

## 📈 What It Does

- **Collects Live Data** from sources like Yahoo Finance to gauge:
  - Recent drawdown (how far the price has dropped from a previous peak)
  - Volatility (7-day and 30-day windows) to measure market risk
  - 1-week momentum (short-term price trajectory)
- **Matches** the user’s scenario against historical data from multiple market indexes (e.g., S&P 500) to:
  - Estimate rebound probability
  - Project potential recovery time (time to break even)
- **Fetches News Headlines** using NewsAPI, then summarizes sentiment via GPT
- **Returns** a consolidated JSON response through a single `/forecast` endpoint, providing both quantitative forecasts and qualitative nudges
- **Integrates** seamlessly with the custom GPT "RationalIntegrationGPT," enabling real-time, behaviorally informed financial advice backed by forecasting and nudging principles.

---

## 🔧 How To Use

1. **Endpoint**

   ```
   POST https://<your-render-url>.onrender.com/forecast
   ```

2. **Request Body** (JSON):

   ```json
   {
     "asset_symbol": "AAPL"
   }
   ```

   Optionally include `intent`, `reason_or_horizon`, and `anxiety_level` if you want a fuller behavioral perspective.

3. **Sample Response**:

   ```json
   {
     "drawdown_pct": -0.1125,
     "volatility_30d": 0.1821,
     "momentum_signal": "negative",
     "average_time_to_recovery_months": 3.25,
     "historical_rebound_probability": 0.67,
     "news_sentiment_summary": "Overall sentiment is moderately positive...",
     "current_price": 150.32,
     "volatility_spike_ratio": 1.1
     ...
   }
   ```

   This includes key metrics, sentiment insights, and soft “nudges” to inform the user.

---

## 🔐 Environment Variables

Create a `.env` file with:

```bash
OPENAI_API_KEY=your_openai_key
NEWS_API_KEY=your_newsapi_key
```

> Add these as environment variables in your hosting platform (e.g., Render) if you’re deploying there.

---

## 🚀 Tech Stack

- **Python + Flask** for API construction
- **Pandas, NumPy, scikit-learn** for data manipulation and reference class logic
- **yFinance** for real-time financial data retrieval
- **OpenAI GPT-4** to interpret and summarize news sentiment
- **NewsAPI** to gather relevant headlines
- **Render** (or Replit) for cloud deployment
-
-
## 🧠 Inspiration

RationalInvestorGPT draws upon:

- Reference class forecasting (inspired by Lovallo) to reduce over-optimism or panic
- **Behavioral strategy** and **nudges** (inspired by Kahneman, Tversky, Thaler, and Sunstein) to gently influence rational behavior
- **Bounded rationality** concepts, acknowledging real-world cognitive limitations
- **Decision hygiene** principles, emphasizing slow, reflective investment choices

With these foundations, the tool aims to help investors make calmer, more data-driven decisions in today’s volatile markets.

