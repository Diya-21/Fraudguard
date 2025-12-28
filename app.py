import streamlit as st
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="FraudGuard", layout="centered")

st.title("💳 FraudGuard – Credit Card Fraud Detection")
st.write("Simulate a transaction and check fraud risk using a behavior-based ML model.")

st.divider()

# -------------------------------
# USER INPUTS (TRANSACTION)
# -------------------------------
st.subheader("Enter Transaction Details")

amount = st.number_input("Transaction Amount", min_value=0.0, value=250.0)
hour = st.slider("Hour of Transaction (0–23)", 0, 23, 19)
time_gap = st.number_input(
    "Time since last transaction (seconds)", min_value=0.0, value=1800.0
)

# -------------------------------
# FEATURE ENGINEERING (LIVE)
# -------------------------------
is_night_txn = 1 if hour < 6 else 0
log_amount = np.log1p(amount)

# Simulated behavior features
amount_deviation = amount * 0.3
high_velocity = 1 if time_gap < 60 else 0

# -------------------------------
# LOAD & TRAIN MODEL (DEMO PURPOSE)
# -------------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("data/creditcard.csv")

    df = df.sort_values("Time").reset_index(drop=True)
    df['hour'] = (df['Time'] // 3600) % 24
    df['is_night_txn'] = (df['hour'] < 6).astype(int)
    df['log_amount'] = np.log1p(df['Amount'])

    df['rolling_mean_amount'] = df['Amount'].rolling(
        window=10, min_periods=1
    ).mean()
    df['amount_deviation'] = df['Amount'] - df['rolling_mean_amount']

    df['time_since_last_txn'] = df['Time'].diff().fillna(0)
    df['high_velocity'] = (df['time_since_last_txn'] < 60).astype(int)

    df.fillna(0, inplace=True)

    X = df[
        [
            'hour',
            'is_night_txn',
            'log_amount',
            'amount_deviation',
            'time_since_last_txn',
            'high_velocity',
        ]
    ]
    y = df['Class']

    model = RandomForestClassifier(
        n_estimators=50,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    return model


model = train_model()

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚨 Check Fraud Risk"):

    input_data = pd.DataFrame(
        [[
            hour,
            is_night_txn,
            log_amount,
            amount_deviation,
            time_gap,
            high_velocity
        ]],
        columns=[
            'hour',
            'is_night_txn',
            'log_amount',
            'amount_deviation',
            'time_since_last_txn',
            'high_velocity'
        ]
    )

    fraud_prob = model.predict_proba(input_data)[0][1]
    risk_score = fraud_prob * 100

    st.divider()
    st.subheader("📊 Fraud Risk Assessment")

    st.metric("Fraud Risk Score", f"{risk_score:.2f} / 100")

    # -------------------------------
    # SYSTEM DECISION (REAL BANK LOGIC)
    # -------------------------------
    st.subheader("🛡️ System Decision")

    if risk_score < 30:
        st.success("✅ LOW RISK: Transaction Approved")
        st.write("System Action: Transaction processed successfully.")

    elif risk_score < 70:
        st.warning("⚠️ MEDIUM RISK: Additional Verification Required")
        st.write("System Action: OTP / Customer verification triggered.")

    else:
        st.error("🚨 HIGH RISK: Possible Fraud Detected")
        st.write("System Action: Transaction blocked.")
        st.write("📩 Alert sent to customer via SMS / Banking App.")

    # -------------------------------
    # EXPLANATION (HUMAN READABLE)
    # -------------------------------
    st.subheader("🧠 Why was this decision made?")

    reasons = []
    if is_night_txn:
        reasons.append("Transaction occurred at night")
    if high_velocity:
        reasons.append("Rapid successive transactions detected")
    if amount_deviation > 0:
        reasons.append("Unusual transaction amount pattern")

    if reasons:
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.write("- Transaction behavior appears normal")

# -------------------------------
# FOOTER
# -------------------------------
st.divider()
st.caption(
    "FraudGuard simulates a real-world fraud detection decision engine. "
    "Alerts and actions shown here represent how a banking system would respond in production."
)
