# Celebrare — Strategic Intelligence Dashboard (Internship Task)

## Overview
This project fetches Google News + Twitter, assigns sentiment (VADER), aggregates into a timeseries, monitors keywords & spikes, forecasts sentiment with Prophet, and presents a Streamlit dashboard.

## Setup (recommended)
1. Clone or copy the project into a folder.
2. Create a Python venv:
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows

3. Install requirements:
   pip install -r requirements.txt

   If `prophet` fails to install, try:
   conda install -c conda-forge prophet
   or pip install prophet --use-feature=2020-resolver

4. Create a `.env` file (copy `.env.example`) and add:
   TWITTER_BEARER_TOKEN=AAAAAAAAAAAAAAAAAAAAAHqj4gEAAAAAqM0x50BPfMosUVtnReCpoVTZ%2BCQ%3DSxg7UCwdZqGVy9nUTN0rgFZbpjWTx0qdBmlZpYuBGIT6cdKMD5
   SLACK_WEBHOOK_URL=optional
   DEFAULT_KEYWORDS=AI,ML

## Run
1. In the activated venv, run:
   streamlit run app.py

2. The Streamlit app will open in the browser (http://localhost:8501).

## Notes
- Twitter API: you need a valid Twitter developer bearer token with Elevated access for recent search.
- gnews: no keys required.
- Prophet: may require additional system packages on some systems.

## Files to include in your ZIP submission
- All .py files, requirements.txt, README.md, .env (exclude secrets when sharing), sample outputs (CSV), screenshots / demo video.

