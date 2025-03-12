import numpy as np
import pandas as pd
import streamlit as st 
import yfinance as yf

import andrewbreakout
import sp500
import tsx

ss = st.session_state

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
        with st.spinner("Filtering TSX tickers..."):
            tickers = [ticker for ticker in tsx.TSX_tickers]
    if selected_choice == options[2]:
        if ss.rerun_rand:
            ss.tickers = np.random.choice(sp500.SP500_tickers, 20, replace=False)
            ss.rerun_rand = False
        tickers = ss.tickers
    else:
        ss.rerun_rand = True


    ss.results = {}

    @st.experimental_fragment(run_every=1)
    def show_tickers():
        if len(ss.results.values()) < len(tickers):
            st.text(f"{len(tickers)} tickers found, scanning...")
            return
        st.text(f"Percentage of stocks that broke out: {len([info for info in ss.results.values() if info and info['first_breakout']]) / len(tickers):.2%}")
        st.header("All tickers (download):")
        df= pd.DataFrame([key for key, res in ss.results.items() if res and res["first_breakout"]])
        st.dataframe(df.transpose())

    show_tickers()

    for ticker in tickers:
        try: 
            info = andrewbreakout.find_clustered_resistance_breakout(
            ticker=ticker,
            lookback_years=lookback_years,
            tolerance=tolerance,
            breakout_window=breakout_window,
            window=window,
            margin=margin,
            consecutive_days=consecutive_days,
        )

            ss.results[ticker] = info

            if info and info["first_breakout"]:
                st.text(info["first_breakout"])
                with st.expander(f"Ticker: {ticker}", expanded=True):
                    andrewbreakout.show_modal(info)
        except Exception as e:
            st.text(f"Received the following error with ticker {ticker}")
            st.exception(e)


main()
