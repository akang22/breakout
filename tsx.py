import requests
import pandas as pd

# URL of the TSX interlisted companies list
def get_tsx():
    url = "https://www.tsx.com/files/trading/interlisted-companies.txt"
    response = requests.get(url)
    data = response.text
    lines = [line for line in data.split("\n") if line.strip()]
    return [line.split("\t")[0].split(":")[0] for line in lines[3:]]

TSX_tickers = get_tsx()
