
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
    '2 maanden': 40,
    '6 maanden': 120,
    '1 jaar': 252,
    '5 jaar': 1260
}

horizon_label = st.selectbox('Voorspel vooruit over:', list(horizon_map.keys()))
horizon = int(horizon_map[horizon_label])

if ticker:
    df = yf.download(ticker, period='10y', interval='1d')
    if df.empty:
        st.error("⚠️ Geen data gevonden voor dit aandeel. Controleer de ticker of probeer het later opnieuw.")
    else:
        df.index = pd.to_datetime(df.index)

        df['Return'] = df['Close'].pct_change()
        df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
        df['Target_Price'] = df['Close'].shift(-horizon)

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

        prediction_date = df.index[-1] + timedelta(days=horizon)
        if prediction_date <= datetime.now():
            st.warning("⚠️ Niet genoeg actuele data om {0} vooruit te voorspellen. (Laatste datum in data: {1})".format(
                horizon_label, df.index[-1].strftime("%Y-%m-%d")))
        else:
            features = ['Close', 'SMA_10', 'EMA_10', 'RSI_14', 'MACD']
            X = df[features]
            y_class = df['Target']
            y_reg = df['Target_Price']

            X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, shuffle=False)
            _, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, shuffle=False)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_class_train)
            y_class_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_class_test, y_class_pred)

            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            reg.fit(X_train, y_reg_train)
            y_reg_pred = reg.predict(X_test)
            mae = mean_absolute_error(y_reg_test, y_reg_pred)

            last_data = X.tail(1)
            direction = clf.predict(last_data)[0]
            future_price = reg.predict(last_data)[0]

            try:
                current_price = float(df['Close'].iloc[-1])
                current_price_text = f"${current_price:.2f}"
            except:
                current_price = None
                current_price_text = "Onbekend"

            future_price_text = f"${future_price:.2f}"

            st.metric('Model Accuratesse (richting)', f'{accuracy * 100:.2f}%')
            st.metric('Gemiddelde prijsafwijking (MAE)', f'${mae:.2f}')

            st.subheader('🔮 Voorspelling:')
            if direction == 1:
                st.success(f"📈 Verwachte stijging in de komende {horizon_label}")
            else:
                st.error(f"📉 Verwachte daling in de komende {horizon_label}")

            st.write(f"📌 Verwachte prijs: **{future_price_text}** (Huidige prijs: {current_price_text})")

            st.subheader("📉 Historische koers + voorspelling")
            fig, ax = plt.subplots()
            df['Close'].plot(ax=ax, label='Historisch')

            ax.scatter(
                prediction_date,
                future_price,
                color='green' if direction == 1 else 'red',
                label='Voorspeld',
                zorder=5
            )
            ax.legend()
            st.pyplot(fig)
