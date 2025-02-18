import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sp500 import SP500_tickers


# ================================
# FUNCTIONS
# ================================


def flatten_and_sanitize(df, ticker=None):
    """
    Flatten multi-index columns if needed, extract the single ticker columns,
    rename them if necessary, ensure Open/High/Low/Close/Volume are numeric,
    drop rows with invalid O/H/L/C/V.
    Returns a single-level DataFrame with numeric columns.
    """
    if df.empty:
        return df

    # If df has multi-index columns, we need to flatten or extract
    if isinstance(df.columns, pd.MultiIndex):
        # If yfinance returned columns like ('High','<ticker>'), etc.
        if ticker and ticker in df.columns.levels[1]:
            try:
                df = df.xs(key=ticker, axis=1, level=1)
            except KeyError:
                pass
        else:
            df.columns = df.columns.droplevel(0)

    # Convert to standard column names
    col_map = {}
    for c in df.columns:
        c_lower = c.lower()
        if "open" in c_lower:
            col_map[c] = "Open"
        elif "high" in c_lower:
            col_map[c] = "High"
        elif "low" in c_lower:
            col_map[c] = "Low"
        elif "close" in c_lower and "adj" not in c_lower:
            col_map[c] = "Close"
        elif "volume" in c_lower:
            col_map[c] = "Volume"
    df.rename(columns=col_map, inplace=True)

    # Keep only the essential columns if they exist
    needed_cols = ["Open", "High", "Low", "Close", "Volume"]
    existing = [c for c in needed_cols if c in df.columns]

    # Convert them to numeric
    for col in existing:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN in any essential column
    df.dropna(subset=existing, inplace=True)

    return df


def isolate_single_high_column(df, ticker=None):
    """
    Ensures 'df' has exactly one 'High' column.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if "High" not in df.columns:
        raise ValueError("'High' column not found.")
    return df["High"].dropna()


def isolate_single_close_column(df, ticker=None):
    """
    Ensures 'df' has exactly one 'Close' column.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if "Close" not in df.columns:
        raise ValueError("'Close' column not found.")
    return df["Close"].dropna()


@st.cache_data
def get_score(prices, resistance, target_decay=None, window=2):
    if target_decay is None:
        # linear decay
        target_decay = lambda x: x

    time_weights = np.array([target_decay(i / len(prices)) for i in range(len(prices))])

    maxima_indices = []
    minima_indices = []

    for i in range(window, len(prices) - window):
        current_val = prices.iloc[i]
        neighborhood = prices.iloc[i - window : i + window + 1]
        if current_val == neighborhood.max():
            maxima_indices.append(i)
        if current_val == neighborhood.min():
            minima_indices.append(i)

    score_above = 0
    score_reward = 0
    score_maxima = 0

    for i in range(len(prices) - 1):
        price, next_price = prices.iloc[i], prices.iloc[i + 1]
        weight = time_weights[i]
        dist_adjust = 1 + 100 * abs(price - resistance) / resistance

        if price > resistance:
            penalty = dist_adjust
            if next_price > price:
                penalty *= 1.5
            score_above -= penalty

        if i in maxima_indices:
            if dist_adjust < 4:  # If within 3% of resistance
                score_reward += (1 / dist_adjust) * 600 * window * weight

        if i in minima_indices and price > resistance:
            score_maxima -= dist_adjust * 5 * window

    print(
        resistance,
        score_above,
        score_maxima,
        score_reward,
        score_above + score_maxima + score_reward,
    )

    return score_above + score_maxima + score_reward


@st.cache_data
def find_local_maxima(series_or_df, window=2):
    """
    Finds local maxima by comparing each day's High with its neighbors in a ± 'window'.
    Returns a DataFrame with ['Date','High'] for those maxima.
    """
    if isinstance(series_or_df, pd.DataFrame):
        if "High" not in series_or_df.columns:
            return pd.DataFrame(columns=["Date", "High"])
        data = series_or_df["High"].dropna()
    else:
        data = series_or_df.dropna()

    if len(data) < 2 * window + 1:
        return pd.DataFrame(columns=["Date", "High"])

    local_max_dates = []
    local_max_vals = []

    for i in range(window, len(data) - window):
        current_val = data.iloc[i]
        neighborhood = data.iloc[i - window : i + window + 1]
        if current_val == neighborhood.max():
            local_max_dates.append(data.index[i])
            local_max_vals.append(current_val)

    return pd.DataFrame({"Date": local_max_dates, "High": local_max_vals})


@st.cache_data
def determine_5_year_resistance(local_max_df, df, window, breakout_window, tolerance=0.03):
    """
    Clusters local maxima if they are within ± 'tolerance' in relative terms.
    Picks the cluster with the most touches (ties => cluster with the highest average).
    Returns the cluster's average price or None.
    """
    if local_max_df.empty:
        return None

    highs = local_max_df["High"].values
    if len(highs) == 0:
        return None

    # cluster_info is a list of tuples: (average_price_of_cluster, cluster_size)
    cluster_info = [
        (
            c,
            get_score(
                df[df.index < datetime.now() - timedelta(days=breakout_window * 30)]["Close"], c, window=window
            ),
        )
        for c in sorted(highs)
    ]
    # Choose the cluster with the most touches; break ties by highest average
    clusters = sorted(cluster_info, key=lambda x: (x[1], x[0]))
    best_cluster = clusters[-1]

    return best_cluster[0]


@st.cache_data
def find_clustered_resistance_breakout(
    ticker,
    lookback_years=5,
    tolerance=0.03,
    breakout_window=15,  # months: ticker is only valid if first breakout is within these months
    window=2,  # local maxima neighbor window
    margin=0.08,  # breakout threshold above resistance
    consecutive_days=5,
):
    """
    1) Download ~5 yrs daily data for 'ticker'.
    2) Flatten & sanitize columns.
    3) Find local maxima -> cluster -> 5-yr 'resistance'.
    4) Always check for breakouts in the last 15 months:
       - Close > resistance*(1+margin)
       - Must hold for 'consecutive_days' in a row.
    5) Then, validate the breakout by ensuring that the first breakout occurred
       within the last 'breakout_window' months.
    6) Return dict of relevant info.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(lookback_years * 365.25))

    raw_df = yf.download(ticker, interval="1wk", start=start_date, end=end_date, progress=False)
    df = flatten_and_sanitize(raw_df.copy(), ticker=ticker)

    try:
        df_high = isolate_single_high_column(df)
        df_close = isolate_single_close_column(df)
    except ValueError:
        return None

    local_max_df = find_local_maxima(df_close, window=window)
    if local_max_df.empty:
        return None

    cluster_res = determine_5_year_resistance(
        local_max_df, df, window, breakout_window, tolerance=tolerance
    )
    if cluster_res is None:
        return None

    # Always use a fixed 15-month lookback period for breakout detection
    breakout_lookback_months = 15
    breakout_start = end_date - timedelta(days=30 * breakout_lookback_months)
    df_close_recent = df_close[df_close.index >= breakout_start]

    breakout_threshold = cluster_res * (1 + margin)
    above_line = df_close_recent > breakout_threshold
    rolling_sum = above_line[::-1].rolling(window=consecutive_days).sum()[::-1]
    breakout_mask = rolling_sum == consecutive_days

    df_breakouts = df_close_recent[breakout_mask]
    breakout_dates = df_breakouts.index.strftime("%Y-%m-%d").tolist()

    # Validate breakout: only consider valid if the first breakout is within
    # the last breakout_window months
    valid_breakout = None
    valid_breakout_price = None
    if not df_breakouts.empty:
        first_breakout_date = df_breakouts.index[0]
        if first_breakout_date >= end_date - timedelta(days=30 * breakout_window):
            valid_breakout = first_breakout_date.strftime("%Y-%m-%d")
            valid_breakout_price = float(df_breakouts.iloc[0])

    info_dict = {
        "ticker": ticker,
        "df_high": df_high,
        "df_close": df_close,
        "df_raw": df,
        "resistance": cluster_res,
        "margin": margin,
        "consecutive_days": consecutive_days,
        "first_breakout": valid_breakout,
        "first_breakout_price": valid_breakout_price,
        "all_breakouts": breakout_dates,
    }

    return info_dict


def display_stock_with_resistance_return(info_dict):
    """
    Creates a candlestick chart with additional technical overlays and a horizontal
    resistance line. Returns the matplotlib figure.
    """
    ticker = info_dict["ticker"]
    df_raw = info_dict["df_raw"].copy()
    first_breakout_date = info_dict["first_breakout"]
    first_breakout_price = info_dict["first_breakout_price"]
    cluster_res = info_dict["resistance"]

    # Ensure the index name is 'Date'
    df_raw.index.name = "Date"

    # Calculate indicators: 20-day SMA and Bollinger Bands
    df_raw["SMA20"] = df_raw["Close"].rolling(20).mean()
    df_raw["STD20"] = df_raw["Close"].rolling(20).std()
    df_raw["UpperBB"] = df_raw["SMA20"] + 2 * df_raw["STD20"]
    df_raw["LowerBB"] = df_raw["SMA20"] - 2 * df_raw["STD20"]

    # Use breakout price if available; otherwise, use resistance
    line_price = cluster_res

    apds = [
        mpf.make_addplot(df_raw["Close"], color="blue", width=1),
        mpf.make_addplot(df_raw["UpperBB"], color="grey", width=0.75),
        mpf.make_addplot(df_raw["LowerBB"], color="grey", width=0.75),
    ]

    # Add a marker for the breakout day if applicable
    if first_breakout_date and first_breakout_price:
        ts = pd.to_datetime(first_breakout_date)
        if ts in df_raw.index:
            idx = df_raw.index.get_loc(ts)
            scatter_data = [np.nan] * len(df_raw)
            scatter_data[idx] = first_breakout_price
            apds.append(
                mpf.make_addplot(
                    scatter_data, type="scatter", marker="^", markersize=100, color="g"
                )
            )

    hline_dict = dict(
        hlines=[line_price],
        colors="red",
        linestyle="--",
    )

    if first_breakout_date:
        title_str = (
            f"{ticker}: Breakout on {first_breakout_date} @ {first_breakout_price:.2f}"
        )
    else:
        title_str = f"{ticker}: No breakout (Resistance={cluster_res:.2f})"

    fig, ax = mpf.plot(
        df_raw,
        type="candle",
        style="yahoo",
        volume=True,
        addplot=apds,
        hlines=hline_dict,
        title=title_str,
        figratio=(12, 6),
        figscale=1.1,
        warn_too_much_data=9999999,
        returnfig=True,
    )
    return fig


# ================================
# MAIN APP
# ================================
def main():
    st.title("Clustered Resistance Breakout Analysis")
    st.markdown(
        "This app analyzes selected stocks for resistance breakouts using yfinance data."
    )

    lookback_years = 5
    tolerance = 0.03
    window = 2
    breakout_window = st.slider(
        "Breakout Window (months)", min_value=1, max_value=24, value=6
    )
    margin = 0.08
    consecutive_days = 8

    selected_ticker = st.text_input("Input Ticker")

    options = ["Example Stocks", "Random Selection"]

    selected_choice = st.selectbox("Select view option", options)

    ss = st.session_state
    tickers = []

    if "rerun_rand" not in ss:
        ss.rerun_rand = True

    if selected_choice == options[0]:
        tickers = ["RJF", "ADP", "HPE", "MS", "UHS"]
    if selected_choice == options[1]:
        if ss.rerun_rand:
            ss.tickers = np.random.choice(SP500_tickers, 20, replace=False)
            ss.rerun_rand = False
        tickers = ss.tickers
    else:
        ss.rerun_rand = True
    if selected_ticker:
        tickers = [selected_ticker]

    with st.expander("List of Tickers"):
        st.text(tickers)

    if "page" not in ss:
        ss.page = 0

    prev_button, next_button = st.columns(2)
    if ss.page > 0 and prev_button.button("Prev"):
        ss.page = ss.page - 1
    if ss.page < len(tickers) / 5 - 1 and next_button.button("Next"):
        ss.page = ss.page + 1

    for ticker in tickers[5 * ss.page : 5 * ss.page + 5]:
        with st.spinner("Downloading data and performing analysis..."):
            info = find_clustered_resistance_breakout(
                ticker=ticker,
                lookback_years=lookback_years,
                tolerance=tolerance,
                breakout_window=breakout_window,
                window=window,
                margin=margin,
                consecutive_days=consecutive_days,
            )

        with st.expander(f"Ticker: {ticker}", expanded=not (not info or not info["first_breakout"])):
            if info is None:
                st.error(
                    "Analysis failed. Please check the parameters or try a different ticker."
                )
                return

            res = info["resistance"]
            thr = res * (1 + info["margin"])
            bdate = info["first_breakout"]
            bprice = info["first_breakout_price"]

            st.write(f"**Resistance:** {res:.2f}")
            st.write(f"**Threshold:** {thr:.2f}")
            if bdate:
                st.write(f"**Breakout Date:** {bdate} at **Price:** {bprice:.2f}")
            else:
                st.write(
                    "**No valid breakout found** in the specified breakout window."
                )

            st.write("### Chart")
            fig = display_stock_with_resistance_return(info)
            st.pyplot(fig)

    for ticker in tickers:
        info = find_clustered_resistance_breakout(
            ticker=ticker,
            lookback_years=lookback_years,
            tolerance=tolerance,
            breakout_window=breakout_window,
            window=window,
            margin=margin,
            consecutive_days=consecutive_days,
        )


if __name__ == "__main__":
    main()
