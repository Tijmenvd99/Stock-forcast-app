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

    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26

    df.dropna(inplace=True)

    # Features en labels
    features = ['Close', 'SMA_10', 'EMA_10', 'RSI_14', 'MACD']
    X = df[features]
    y_class = df['Target']
    y_reg = df['Target_Price']

    # Train/test split
    X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, shuffle=False)
    _, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, shuffle=False)

    # Classificatie-model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_class_train)
    y_class_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_class_test, y_class_pred)

    # Regressie-model
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_reg_train)
    y_reg_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)

    # Laatste datapunt
    last_data = X.tail(1)
    direction = clf.predict(last_data)[0]
    future_price = reg.predict(last_data)[0]
    current_price = df['Close'].iloc[-1]

    # Metrics tonen
    st.metric('Model Accuratesse (richting)', f'{accuracy * 100:.2f}%')
    st.metric('Gemiddelde prijsafwijking (MAE)', f'${mae:.2f}')

    # Voorspellingstekst
    st.subheader('🔮 Voorspelling:')
    if direction == 1:
        st.success(f"📈 Verwachte stijging in de komende {horizon_label}")
    else:
        st.error(f"📉 Verwachte daling in de komende {horizon_label}")

    # Prijsformattering gesplitst in aparte variabelen
    future_price_text = f"${future_price:.2f}"
    current_price_text = f"${current_price:.2f}"
    st.write(f"📌 Verwachte prijs: **{future_price_text}** (Huidige prijs: {current_price_text})")

    # Grafiek
    st.subheader("📉 Historische koers + voorspelling")
    fig, ax = plt.subplots()
    df['Close'].plot(ax=ax, label='Historisch')

    unit = "hours" if "uur" in horizon_label else "days"
    if unit == "hours":
        time_ahead = pd.Timedelta(hours=horizon)
    else:
        time_ahead = pd.Timedelta(days=horizon)

    ax.scatter(
        df.index[-1] + time_ahead,
        future_price,
        color='green' if direction == 1 else 'red',
        label='Voorspeld',
        zorder=5
    )
    ax.legend()
    st.pyplot(fig)
