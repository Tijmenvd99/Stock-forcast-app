
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title('📈 Aandelen Voorspeller met Technische Indicatoren')

ticker = st.text_input('Vul een aandeelsticker in (bijv. AMD, HPQ, ^GSPC)', 'AMD')

if ticker:
    df = yf.download(ticker, period='2y')
    df = df[['Close']]
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)

    # Technische indicatoren
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26

    df.dropna(inplace=True)

    X = df[['Close', 'SMA_10', 'EMA_10', 'RSI_14', 'MACD']]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.metric('Model Accuratesse', f'{accuracy * 100:.2f}%')

    latest_prediction = model.predict(X.tail(1))[0]
    voorspelling = '📈 Verwacht stijging' if latest_prediction == 1 else '📉 Verwacht daling'
    st.subheader(voorspelling)
