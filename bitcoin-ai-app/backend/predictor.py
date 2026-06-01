"""
Bitcoin price predictor — loads the GRU champion model and serves predictions.
Model: GRU + RSI + MACD  |  MAPE: 3.05%  |  R²: 0.968
"""

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import joblib
import json
import os
from datetime import datetime, timedelta

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
WINDOW = 30


class BTCPredictor:
    def __init__(self):
        model_path   = os.path.join(MODEL_DIR, "bitcoin_champion_final.keras")
        scaler_path  = os.path.join(MODEL_DIR, "bitcoin_scaler_champion.pkl")
        features_path = os.path.join(MODEL_DIR, "bitcoin_features_champion.json")

        self.model  = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(features_path) as f:
            self.features = json.load(f)

        print(f"✅ Model loaded | Features: {self.features}")

    # ── Data ──────────────────────────────────────────────────────────────────

    def _fetch_data(self, period: str = "120d") -> pd.DataFrame:
        df = yf.download("BTC-USD", period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.rename(columns={"Close": "close", "Volume": "Volume USD"})

        # Technical indicators (manual — no extra dependency)
        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"]        = ema12 - ema26
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        return df.dropna()

    # ── Core prediction ───────────────────────────────────────────────────────

    def _inverse_transform(self, pred_scaled: float) -> float:
        """Inverse-transform a scaled prediction (column 0 = close)."""
        dummy = np.zeros((1, len(self.features)))
        dummy[0, 0] = pred_scaled
        return float(self.scaler.inverse_transform(dummy)[0, 0])

    def _predict_from_window(self, window_data: np.ndarray) -> float:
        scaled = self.scaler.transform(window_data)
        X = scaled.reshape(1, WINDOW, len(self.features))
        pred_scaled = float(self.model.predict(X, verbose=0)[0, 0])
        return self._inverse_transform(pred_scaled)

    # ── Endpoints ─────────────────────────────────────────────────────────────

    def predict_tomorrow(self) -> dict:
        df = self._fetch_data(period="120d")
        last_window    = df[self.features].tail(WINDOW).values
        predicted_price = self._predict_from_window(last_window)

        current_price = float(df["close"].iloc[-1])
        change_pct    = (predicted_price - current_price) / current_price * 100
        confidence    = round(min(99, max(60, 96.8 - abs(change_pct) * 0.5)), 1)

        return {
            "current_price":   round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "change_percent":  round(change_pct, 2),
            "confidence":      confidence,
            "model":           "GRU + RSI + MACD",
            "mape":            3.05,
            "r2":              0.968,
            "timestamp":       datetime.utcnow().isoformat() + "Z",
            "prediction_date": (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d"),
        }

    def get_current_price(self) -> dict:
        df = self._fetch_data(period="5d")
        current_price = float(df["close"].iloc[-1])
        prev_price    = float(df["close"].iloc[-2])
        change_pct    = (current_price - prev_price) / prev_price * 100
        return {
            "price":             round(current_price, 2),
            "change_percent_24h": round(change_pct, 2),
            "timestamp":         datetime.utcnow().isoformat() + "Z",
        }

    def get_history(self, days: int = 30) -> dict:
        fetch_period = f"{days + WINDOW + 30}d"
        df = self._fetch_data(period=fetch_period)
        df_slice = df.tail(days + WINDOW)

        history = []
        for i in range(WINDOW, len(df_slice)):
            window_data     = df_slice[self.features].iloc[i - WINDOW:i].values
            predicted_price = self._predict_from_window(window_data)
            real_price      = float(df_slice["close"].iloc[i])
            error_pct       = abs(predicted_price - real_price) / real_price * 100
            history.append({
                "date":            df_slice.index[i].strftime("%Y-%m-%d"),
                "real_price":      round(real_price, 2),
                "predicted_price": round(predicted_price, 2),
                "error_percent":   round(error_pct, 2),
            })

        return {"history": history[-days:], "days": days}
