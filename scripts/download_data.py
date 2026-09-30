"""
Script to download historical stock market data (Amazon - AMZN)
from GitHub mirror of Kaggle DJIA 30 Stock Time Series (Rodolfo Saldanha's benchmark)
and save it to data/ directory.
"""
import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

URL = "https://raw.githubusercontent.com/soham2707/Stock-Market-Analysis-And-Forecasting-Using-Deep-Learning/master/AMZN_2006-01-01_to_2018-01-01.csv"
DEST_FILE_1 = os.path.join(DATA_DIR, "AMZN_2006-01-01_to_2018-01-01.csv")
DEST_FILE_2 = os.path.join(DATA_DIR, "AMZN.csv")

def download_dataset():
    print(f"Downloading AMZN dataset from {URL}...")
    urllib.request.urlretrieve(URL, DEST_FILE_1)
    # Also save as AMZN.csv for convenience
    with open(DEST_FILE_1, "rb") as src, open(DEST_FILE_2, "wb") as dst:
        dst.write(src.read())
    size = os.path.getsize(DEST_FILE_1)
    print(f"Downloaded successfully: {DEST_FILE_1} ({size / 1024:.2f} KB)")
    print(f"Copied to: {DEST_FILE_2}")

if __name__ == "__main__":
    download_dataset()
