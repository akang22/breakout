import numpy as np
import streamlit as st 
import yfinance as yf

import andrewbreakout
import sp500
import tsx

ss = st.session_state

@st.cache_data
def is_valid_tsx(ticker):
    info = yf.Ticker(ticker).info
    return info['dayHigh'] > 3 and info['typeDisp'] == 'equity' and info['averageDailyVolume10Day'] > 50000

def main():
    options = ["SP500", "Filtered TSX", "Random 20"]

    selected_choice = st.selectbox("Select view option", options)

    if "rerun_rand" not in ss:
        ss.rerun_rand = True

    lookback_years = 5
    tolerance = 0.03
    window = 2
    breakout_window = st.slider(
        "Breakout Window (months)", min_value=1, max_value=24, value=6
    )
    margin = 0.08
    consecutive_days = 8


    tickers = []
    if selected_choice == options[0]:
        tickers = sp500.SP500_tickers
    if selected_choice == options[1]:
        tickers = [ticker for ticker in tsx.TSX_tickers if is_valid_tsx(ticker)]
    if selected_choice == options[2]:
        if ss.rerun_rand:
            ss.tickers = np.random.choice(sp500.SP500_tickers, 20, replace=False)
            ss.rerun_rand = False
        tickers = ss.tickers
    else:
        ss.rerun_rand = True

    if "page" not in ss:
        ss.page = 0

    prev_button, next_button = st.columns(2)
    if ss.page > 0 and prev_button.button("Prev"):
        ss.page = ss.page - 1
    if ss.page < len(tickers) / 5 - 1 and next_button.button("Next"):
        ss.page = ss.page + 1

    if "results" not in ss:
        ss.results = []

    filtered = [res for res in ss.results if res and res["first_breakout"]]
    if len(filtered) >= 5 * ss.page:
        for info in filtered[5 * ss.page:5*ss.page + 5]:
            with st.expander(f"Ticker: {info['ticker']}"):
                andrewbreakout.show_modal(info)

    for ticker in tickers[len(ss.results):]:
        with st.spinner(f"Checking: {ticker}"):
            info = andrewbreakout.find_clustered_resistance_breakout(
                ticker=ticker,
                lookback_years=lookback_years,
                tolerance=tolerance,
                breakout_window=breakout_window,
                window=window,
                margin=margin,
                consecutive_days=consecutive_days,
            )
        ss.results.append(info)

        filtered = [res for res in ss.results if res and res["first_breakout"]]

        if 5 * ss.page < len(filtered) <= 5 * ss.page + 5 and info and info["first_breakout"]:
            with st.expander(f"Ticker: {ticker}"):
                andrewbreakout.show_modal(info)

    st.header("All tickers (download):")
    st.dataframe([res["ticker"] for res in ss.results if res and res["first_breakout"]])

main()
