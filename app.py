import yfinance as yf
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime

st.set_page_config(page_title="AI Aandelen Voorspeller", layout="centered")
st.title("📈 AI Aandelen Voorspeller")

st.markdown("""
Voer een **aandelenticker** in (zoals `AMD`, `HPQ`, `^GSPC`) en klik op **'Zoek aandeel'** om een voorspelling te zien.
""")

# --- Formulier met knop ---
with st.form("stock_form"):
    ticker = st.text_input("Ticker symbool:", "AMD")
    submit = st.form_submit_button("🔍 Zoek aandeel")

if submit:
    start_date = "2018-01-01"
    end_date = datetime.date.today().strftime('%Y-%m-%d')

    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.warning("Geen data gevonden voor deze ticker. Probeer iets anders.")
        st.stop()

    data['Return'] = data['Close'].pct_change()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['Volatility'] = data['Return'].rolling(window=10).std()
    data.dropna(inplace=True)

    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    features = ['Close', 'SMA_10', 'SMA_50', 'Volatility']
    X = data[features]
    y = data['Target']

    if len(X) < 100:
        st.warning("Niet genoeg data om een betrouwbare voorspelling te maken.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    try:
        latest_data = X.tail(1)
        prediction = model.predict(latest_data)[0]
        pred_label = "📈 STIJGING verwacht" if prediction == 1 else "📉 DALING verwacht"
    except Exception as e:
        pred_label = f"⚠️ Voorspelling mislukt: {str(e)}"

    st.subheader(f"📊 Resultaten voor {ticker.upper()}")
    st.metric(label="Model Accuratesse", value=f"{accuracy*100:.2f}%")
    st.metric(label="Laatste voorspelling", value=pred_label)
    st.line_chart(data['Close'], height=300, use_container_width=True)
    st.caption("Deze voorspelling is gebaseerd op historische patronen en is geen beleggingsadvies.")
