import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt

st.title('📊 Aandelen Voorspeller 2.0')

ticker = st.text_input('Aandeelsticker (bijv. AMD, HPQ, ^GSPC)', 'AMD')

horizon_map = {
    '1 uur': 1,
    '2 uur': 2,
    '1 dag': 1,
    '2 dagen': 2,
    '1 week': 5,
    '2 weken': 10,
    '1 maand': 20,
    '2 maanden': 40
}

horizon_label = st.selectbox('Voorspel vooruit over:', list(horizon_map.keys()))
horizon = int(horizon_map[horizon_label])

if ticker:
    # Data ophalen en index naar datetime converteren
    df = yf.download(ticker, period='6mo', interval='1d')
    df.index = pd.to_datetime(df.index)

    # Doelvariabelen
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    df['Target_Price'] = df['Close'].shift(-horizon)

    # Technische indicatoren
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    ema
