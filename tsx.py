import requests
import pandas as pd
import streamlit as st

# URL of the TSX interlisted companies list
def get_tsx():
    API_KEY = st.secrets["FMP"]

    url = f"https://financialmodelingprep.com/api/v3/stock-screener?exchange=TSX&apikey={API_KEY}&isEtf=false&isFund=false&priceMoreThan=3&volumeMoreThan=50000"
    response = requests.get(url)
    data = response.json()

    return [val['symbol'] for val in data]


TSX_tickers = get_tsx()
