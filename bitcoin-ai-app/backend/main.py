"""
Bitcoin AI Predictor — FastAPI backend
Run: uvicorn main:app --reload --port 8000
"""

import asyncio
import httpx
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from predictor import BTCPredictor

app = FastAPI(
    title="Bitcoin AI Predictor",
    description="Next-day Bitcoin price prediction using GRU Deep Learning (MAPE 3.05%, R² 0.968)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ────────────────────────────────────────────────
predictor = BTCPredictor()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "api":     "Bitcoin AI Predictor",
        "version": "1.0.0",
        "model":   "GRU + RSI + MACD",
        "mape":    "3.05%",
        "r2":      0.968,
    }


@app.get("/predict")
def predict():
    """
    Predict tomorrow's Bitcoin closing price.

    Returns current price, predicted price, % change, and confidence score.
    """
    try:
        return predictor.predict_tomorrow()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/current")
def current():
    """
    Returns the latest available BTC closing price and 24h change.
    """
    try:
        return predictor.get_current_price()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
def history(days: int = Query(default=30, ge=7, le=365, description="Number of past days")):
    """
    Returns the last N days of real prices vs model predictions.

    Each row: date, real_price, predicted_price, error_percent
    """
    try:
        return predictor.get_history(days=days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ── WebSocket live price (CoinGecko — no geo-restrictions) ────────────────────

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                try:
                    resp = await client.get(
                        "https://api.coingecko.com/api/v3/simple/price",
                        params={"ids": "bitcoin", "vs_currencies": "usd"},
                    )
                    if resp.status_code == 200:
                        price = resp.json()["bitcoin"]["usd"]
                        await websocket.send_json({"price": price})
                except Exception:
                    pass
                await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
