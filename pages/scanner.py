import numpy as np
import pandas as pd
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

    if "results" not in ss:
        ss.results = {}

    @st.experimental_fragment(run_every=1)
    def show_tickers():
        st.text(f"Percentage of stocks that broke out: {len([info for info in ss.results.values() if info and info['first_breakout']]) / len(tickers):.2%}")
        st.header("All tickers (download):")
        df= pd.DataFrame([key for key, res in ss.results.items() if res and res["first_breakout"]])
        st.dataframe(df.transpose())

    show_tickers()

    for ticker in tickers:
        if ticker not in ss.results:
            ss.results[ticker] = andrewbreakout.find_clustered_resistance_breakout(
                ticker=ticker,
                lookback_years=lookback_years,
                tolerance=tolerance,
                breakout_window=breakout_window,
                window=window,
                margin=margin,
                consecutive_days=consecutive_days,
            )

        info = ss.results[ticker]

        if info and info["first_breakout"]:
            with st.expander(f"Ticker: {ticker}", expanded=True):
                andrewbreakout.show_modal(info)

main()
