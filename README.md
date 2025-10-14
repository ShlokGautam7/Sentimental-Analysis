# Strategic Intelligence Dashboard — Sentimental Analysis

A unified dashboard that fetches, analyzes, and forecasts sentiment trends from Google News and Twitter using Python, Streamlit, and NLP techniques. Designed for the Celebrare Internship, this project enables strategic monitoring of AI/ML-related topics by aggregating events, visualizing sentiment, and sending real-time alerts via Slack.

## Features

- **Multi-source Sentiment Analysis:** Retrieves articles from Google News and tweets from Twitter, then applies VADER sentiment scoring for effective event monitoring.
- **Unified Data Handling:** Normalizes, concatenates, and processes time-series data from both sources for seamless analytics.
- **Aggregated Visualization:** Uses Streamlit and Plotly to plot sentiment, keyword trends, and message volume over user-defined time periods.
- **Alerts & Notifications:** Detects sentiment spikes, keyword surges, and volume bursts, sends instant alerts to Slack for actionable intelligence.
- **Forecasting:** Implements Facebook's Prophet library to predict future sentiment trends based on historical data.
- **Flexible Configuration:** Customizable through an intuitive sidebar for topics, keyword tracking, data limits, frequency, smoothing windows, and thresholds.

## Tech Stack

- Python (`pandas`, `numpy`, `nltk`, `scipy`, `prophet`, `requests`)
- Streamlit (real-time dashboard UI)
- Plotly (interactive charts)
- GNews (news API client)
- Tweepy (Twitter API client)
- Slack Webhook (for notifications)
- dotenv (secure key and token management)

## Setup Instructions

1. **Clone the Repository**
    ```
    git clone https://github.com/ShlokGautam7/Sentimental-Analysis.git
    cd Sentimental-Analysis
    ```

2. **Install Dependencies**
    ```
    pip install -r requirements.txt
    ```

3. **Configure Environment Variables**
    Create a `.env` file with your API keys:

    ```
    TWITTER_BEARER_TOKEN=your-twitter-api-key
    SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
    DEFAULT_KEYWORDS=AI,ML
    ```

4. **Launch the Dashboard**
    ```
    streamlit run app.py
    ```

## Major Components

| File Name             | Purpose                                                                                 |
|-----------------------|-----------------------------------------------------------------------------------------|
| `app.py`              | Streamlit dashboard, config sidebar, data visualization, alert & forecast integration.  |
| `analysis.py`         | Data normalization, timeseries aggregation, sentiment forecast using Prophet.           |
| `data_fetch.py`       | Fetches and structures data from GNews and Twitter using APIs.                          |
| `sentiment_utils.py`  | Adds VADER sentiment scores and labels to news & tweets.                                |
| `utils.py`            | Utility functions for sending Slack messages and zipping files.                         |

## Example Workflow

1. Set your target topic or keywords in the dashboard sidebar.
2. Fetch new data from Google News and Twitter.
3. Analyze sentiment, visualize events, and monitor for spikes.
4. Send alerts for critical intelligence directly to Slack.
5. Forecast trends for strategic planning and insight generation.

## License

Distributed under the MIT License.

---




