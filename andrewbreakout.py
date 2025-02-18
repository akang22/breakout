import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from support_resistance.pricelevels.cluster import RawPriceClusterLevels
from support_resistance.pricelevels.visualization.levels_on_candlestick import (
    plot_levels_on_candlestick,
)
from support_resistance.pricelevels.scoring.touch_scorer import TouchScorer


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

def get_score(df, resistance, target_decay = None, window = 20):
    if target_decay is None:
        # linear decay
        target_decay = lambda x: x

    prices = df["Close"]

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
        price, next_price = prices[i], prices[i + 1]
        weight = time_weights[i]
        dist_adjust = (1 + 100 * abs(price - resistance) / resistance)

        # (1) **Penalty for being above resistance, especially when increasing**
        if price > resistance:
            penalty = dist_adjust
            if next_price > price:
                penalty *= 1.5  
            score_above -= penalty

        # (2) **Reward for local maxima near resistance**
        if i in maxima_indices:
            if dist_adjust < 3:  # If within 2% of resistance
                score_reward += (1 / dist_adjust) * 600 * window * weight

        # (3) **Penalty for local minima above resistance or maxima below resistance**
        if i in minima_indices and price > resistance:
            score_maxima -= dist_adjust * 5 * window
        if i in maxima_indices and price < resistance:
            score_maxima -= dist_adjust * 5 * window

    print(resistance, score_above, score_reward, score_maxima)
    return score_above + score_maxima + score_reward

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

def determine_5_year_resistance(local_max_df, df, tolerance=0.03):
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
    cluster_info = [(c, get_score(df, c)) for c in highs]
    # Choose the cluster with the most touches; break ties by highest average
    best_cluster = max(cluster_info, key=lambda x: (x[1], x[0]))

    return best_cluster[0]

def find_clustered_resistance_breakout(
    ticker,
    lookback_years=5,
    tolerance=0.03,
    breakout_window=15,  # months: ticker is only valid if first breakout is within these months
    window=20,  # local maxima neighbor window
    margin=0.08,  # breakout threshold above resistance
    consecutive_days=20,
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

    raw_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df = flatten_and_sanitize(raw_df.copy(), ticker=ticker)

    try:
        df_high = isolate_single_high_column(df)
        df_close = isolate_single_close_column(df)
    except ValueError:
        return None

    local_max_df = find_local_maxima(df_high, window=window)
    if local_max_df.empty:
        return None

    cluster_res = determine_5_year_resistance(local_max_df, df, tolerance=tolerance)
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
        mpf.make_addplot(df_raw["SMA20"], color="blue", width=1),
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
# TICKERS LIST
# ================================
tickers = [
    "MMM",
    "AOS",
    "ABT",
    "ABBV",
    "ACN",
    "ADM",
    "ADBE",
    "ADP",
    "AAP",
    "AES",
    "AFL",
    "A",
    "APD",
    "AKAM",
    "ALK",
    "ALB",
    "ARE",
    "ALGN",
    "ALLE",
    "LNT",
    "ALL",
    "GOOGL",
    "GOOG",
    "MO",
    "AMZN",
    "AMCR",
    "AMD",
    "AEE",
    "AAL",
    "AEP",
    "AXP",
    "AIG",
    "AMT",
    "AWK",
    "AMP",
    "AME",
    "AMGN",
    "APH",
    "ADI",
    "ANSS",
    "AON",
    "APA",
    "AAPL",
    "AMAT",
    "APTV",
    "ACGL",
    "ANET",
    "AJG",
    "AIZ",
    "T",
    "ATO",
    "ADSK",
    "AZO",
    "AVB",
    "AVY",
    "AXON",
    "BKR",
    "BALL",
    "BAC",
    "BBWI",
    "BAX",
    "BDX",
    "WRB",
    "BBY",
    "BIO",
    "TECH",
    "BIIB",
    "BLK",
    "BK",
    "BA",
    "BKNG",
    "BWA",
    "BXP",
    "BSX",
    "BMY",
    "AVGO",
    "BR",
    "BRO",
    "CHRW",
    "CDNS",
    "CZR",
    "CPT",
    "CPB",
    "COF",
    "CAH",
    "KMX",
    "CCL",
    "CARR",
    "CTLT",
    "CAT",
    "CBOE",
    "CBRE",
    "CDW",
    "CE",
    "CNC",
    "CNP",
    "CF",
    "CRL",
    "SCHW",
    "CHTR",
    "CVX",
    "CMG",
    "CB",
    "CHD",
    "CI",
    "CINF",
    "CTAS",
    "CSCO",
    "C",
    "CFG",
    "CLX",
    "CME",
    "CMS",
    "KO",
    "CTSH",
    "CL",
    "CMCSA",
    "CMA",
    "CAG",
    "COP",
    "ED",
    "STZ",
    "CEG",
    "COO",
    "CPRT",
    "GLW",
    "CTVA",
    "CSGP",
    "COST",
    "CTRA",
    "CCI",
    "CSX",
    "CMI",
    "CVS",
    "DHI",
    "DHR",
    "DRI",
    "DVA",
    "DE",
    "DAL",
    "XRAY",
    "DVN",
    "DXCM",
    "FANG",
    "DLR",
    "DFS",
    "DIS",
    "DG",
    "DLTR",
    "D",
    "DPZ",
    "DOV",
    "DOW",
    "DTE",
    "DUK",
    "DD",
    "DXC",
    "EMN",
    "ETN",
    "EBAY",
    "ECL",
    "EIX",
    "EW",
    "EA",
    "ELV",
    "LLY",
    "EMR",
    "ENPH",
    "ETR",
    "EOG",
    "EFX",
    "EQIX",
    "EQR",
    "ESS",
    "EL",
    "ETSY",
    "EG",
    "EVRG",
    "ES",
    "EXC",
    "EXPE",
    "EXPD",
    "EXR",
    "XOM",
    "FFIV",
    "FDS",
    "FAST",
    "FRT",
    "FDX",
    "FITB",
    "FE",
    "FIS",
    "FMC",
    "F",
    "FTNT",
    "FTV",
    "FOXA",
    "FOX",
    "BEN",
    "FCX",
    "GEN",
    "GRMN",
    "IT",
    "GEHC",
    "GE",
    "GIS",
    "GM",
    "GPC",
    "GILD",
    "GL",
    "GPN",
    "GS",
    "HAL",
    "HBI",
    "HAS",
    "HCA",
    "HSIC",
    "HES",
    "HPE",
    "HLT",
    "HOLX",
    "HD",
    "HON",
    "HRL",
    "HST",
    "HWM",
    "HPQ",
    "HUM",
    "HBAN",
    "HII",
    "IBM",
    "IEX",
    "IDXX",
    "INFO",
    "ITW",
    "ILMN",
    "INCY",
    "IR",
    "INTC",
    "ICE",
    "IFF",
    "IP",
    "IPG",
    "INTU",
    "ISRG",
    "IVZ",
    "INVH",
    "IQV",
    "IRM",
    "JBHT",
    "JKHY",
    "J",
    "JNJ",
    "JCI",
    "JPM",
    "JNPR",
    "K",
    "KDP",
    "KEY",
    "KEYS",
    "KMB",
    "KIM",
    "KMI",
    "KLAC",
    "KHC",
    "KR",
    "LHX",
    "LH",
    "LRCX",
    "LW",
    "LVS",
    "LDOS",
    "LEN",
    "LNC",
    "LIN",
    "LYV",
    "LKQ",
    "LMT",
    "L",
    "LOW",
    "LYB",
    "MTB",
    "MPC",
    "MKTX",
    "MAR",
    "MMC",
    "MLM",
    "MAS",
    "MA",
    "MTCH",
    "MKC",
    "MCD",
    "MCK",
    "MDT",
    "MRK",
    "META",
    "MET",
    "MTD",
    "MGM",
    "MCHP",
    "MU",
    "MSFT",
    "MAA",
    "MRNA",
    "MHK",
    "MOH",
    "TAP",
    "MDLZ",
    "MPWR",
    "MNST",
    "MCO",
    "MS",
    "MOS",
    "MSI",
    "MSCI",
    "NDAQ",
    "NTAP",
    "NFLX",
    "NWL",
    "NEM",
    "NWSA",
    "NWS",
    "NEE",
    "NKE",
    "NI",
    "NDSN",
    "NSC",
    "NTRS",
    "NOC",
    "NCLH",
    "NRG",
    "NUE",
    "NVDA",
    "NVR",
    "NXPI",
    "ORLY",
    "OXY",
    "ODFL",
    "OMC",
    "ON",
    "OKE",
    "ORCL",
    "OGN",
    "OTIS",
    "PCAR",
    "PKG",
    "PARA",
    "PH",
    "PAYX",
    "PAYC",
    "PYPL",
    "PNR",
    "PEP",
    "PFE",
    "PCG",
    "PM",
    "PSX",
    "PNW",
    "PNC",
    "POOL",
    "PPG",
    "PPL",
    "PFG",
    "PG",
    "PGR",
    "PLD",
    "PRU",
    "PTC",
    "PEG",
    "PSA",
    "PHM",
    "QRVO",
    "PWR",
    "QCOM",
    "DGX",
    "RL",
    "RJF",
    "RTX",
    "O",
    "REG",
    "REGN",
    "RF",
    "RSG",
    "RMD",
    "RHI",
    "ROK",
    "ROL",
    "ROP",
    "ROST",
    "RCL",
    "SPGI",
    "CRM",
    "SBAC",
    "SLB",
    "STX",
    "SEE",
    "SRE",
    "NOW",
    "SHW",
    "SBNY",
    "SPG",
    "SWKS",
    "SJM",
    "SNA",
    "SEDG",
    "SO",
    "LUV",
    "SWK",
    "SBUX",
    "STT",
    "STE",
    "SYK",
    "SYF",
    "SNPS",
    "SYY",
    "TMUS",
    "TROW",
    "TTWO",
    "TPR",
    "TGT",
    "TEL",
    "TDY",
    "TFX",
    "TER",
    "TSLA",
    "TXN",
    "TXT",
    "TMO",
    "TJX",
    "TSCO",
    "TT",
    "TDG",
    "TRV",
    "TRMB",
    "TFC",
    "TYL",
    "TSN",
    "USB",
    "UDR",
    "ULTA",
    "UAA",
    "UA",
    "UNP",
    "UAL",
    "UNH",
    "UPS",
    "URI",
    "UHS",
    "VLO",
    "VTR",
    "VRSN",
    "VRSK",
    "VZ",
    "VRTX",
    "VFC",
    "VTRS",
    "V",
    "VNO",
    "VMC",
    "WAB",
    "WMT",
    "WBA",
    "WBD",
    "WM",
    "WAT",
    "WEC",
    "WFC",
    "WELL",
    "WST",
    "WDC",
    "WY",
    "WHR",
    "WMB",
    "WTW",
    "GWW",
    "WYNN",
    "XEL",
    "XYL",
    "YUM",
    "ZBRA",
    "ZBH",
    "ZION",
    "ZTS",
]


# ================================
# MAIN APP
# ================================
def main():
    st.title("Clustered Resistance Breakout Analysis")
    st.markdown(
        "This app analyzes selected stocks for resistance breakouts using yfinance data."
    )

    # 2) Sidebar for analysis parameters
    st.sidebar.header("Analysis Parameters")
    selected_ticker = st.sidebar.selectbox(
        "Select a ticker (or ALL)", ["ALL"] + sorted(tickers)
    )
    lookback_years = st.sidebar.slider(
        "Lookback Years", min_value=1, max_value=10, value=5
    )
    tolerance = (
        st.sidebar.slider("Cluster Tolerance (%)", min_value=1, max_value=10, value=3)
        / 100.0
    )
    breakout_window = st.sidebar.slider(
        "Breakout Window (months)", min_value=1, max_value=24, value=6
    )
    window = st.sidebar.slider(
        "Local Maxima Neighbor Window", min_value=1, max_value=100, value=20
    )
    margin = (
        st.sidebar.slider("Breakout Margin (%)", min_value=1, max_value=20, value=8)
        / 100.0
    )
    consecutive_days = st.sidebar.slider(
        "Consecutive Days", min_value=1, max_value=30, value=8
    )

    # 3) Analysis Button
    if st.button("Analyze"):
        if selected_ticker == "ALL":
            st.info("Analyzing ALL tickers. This might take a while...")
            for ticker in sorted(tickers):
                info = find_clustered_resistance_breakout(
                    ticker=ticker,
                    lookback_years=lookback_years,
                    tolerance=tolerance,
                    breakout_window=breakout_window,
                    window=window,
                    margin=margin,
                    consecutive_days=consecutive_days,
                )
                # Only display tickers that have a valid breakout
                if info is None or info["first_breakout"] is None:
                    continue

                with st.expander(f"Results for {ticker}"):
                    res = info["resistance"]
                    thr = res * (1 + info["margin"])
                    bdate = info["first_breakout"]
                    bprice = info["first_breakout_price"]

                    st.subheader(f"Ticker: {ticker}")
                    st.write(f"**Resistance:** {res:.2f}")
                    st.write(f"**Threshold:** {thr:.2f}")
                    if bdate:
                        st.write(
                            f"**Breakout Date:** {bdate} at **Price:** {bprice:.2f}"
                        )
                    else:
                        st.write(
                            "**No valid breakout found** in the specified breakout window."
                        )

                    st.write("#### Chart")
                    fig = display_stock_with_resistance_return(info)
                    st.pyplot(fig)

        else:
            with st.spinner("Downloading data and performing analysis..."):
                info = find_clustered_resistance_breakout(
                    ticker=selected_ticker,
                    lookback_years=lookback_years,
                    tolerance=tolerance,
                    breakout_window=breakout_window,
                    window=window,
                    margin=margin,
                    consecutive_days=consecutive_days,
                )

            if info is None:
                st.error(
                    "Analysis failed. Please check the parameters or try a different ticker."
                )
                return

            res = info["resistance"]
            thr = res * (1 + info["margin"])
            bdate = info["first_breakout"]
            bprice = info["first_breakout_price"]

            st.subheader(f"Ticker: {selected_ticker}")
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

            if st.checkbox("Show Raw Data"):
                st.dataframe(info["df_raw"])


if __name__ == "__main__":
    main()

