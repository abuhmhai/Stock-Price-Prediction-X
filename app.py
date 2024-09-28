from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
from joblib import load, dump
import yfinance as yf
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

# Initialize FastAPI app
app = FastAPI()

COINMARKETCAP_API_KEY = "ca1bf8a0-dece-42ac-9cd6-a242cb15209b"  # Set your CoinMarketCap API key here

stock_data = None

@app.get("/get_stock_data/{ticker}")
async def get_stock_data(ticker: str):
    global stock_data  # Declare stock_data as global
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol is required")

    ticker = ticker.upper() + "-USD"  # Convert to uppercase and append "-USD"

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')

    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date, interval='1d')

        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker: {ticker}")

        stock_data = [
            {
                'Date': index.strftime("%Y-%m-%d"),
                'Open': row['Open'],
                'High': row['High'],
                'Low': row['Low'],
                'Close': row['Close'],
                'Volume': row['Volume'],
            }
            for index, row in data.iterrows()
        ]

        return JSONResponse(content={'historicalData': stock_data})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/full-dataset")

async def full_dataset():

    try:
        if stock_data is None:
            raise HTTPException(status_code=404, detail="No stock data available. Please fetch data first.")

        filtered_data = [{'Date': record['Date'], 'Close': record['Close']} for record in stock_data]

        return {"data": filtered_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Load the pre-trained model
model = load('./model.joblib')

def train_model_from_full_data(full_data):
    closedf = pd.DataFrame(full_data)[['Close']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))

    training_size = int(len(closedf) * 0.80)
    train_data, test_data = closedf[0:training_size, :], closedf[training_size:len(closedf), :1]

    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=32, verbose=1)

    model.save('./trained_model.h5')  # Correctly save the model
    return history

@app.post("/train_model")
async def train_model_endpoint():
    try:
        full_data = await full_dataset()

        if 'data' not in full_data:
            raise HTTPException(status_code=404, detail="No data available for training")

        history = train_model_from_full_data(full_data['data'])
        return {"message": "Model trained successfully", "training_history": history.history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_next_30_days")
async def predict_next_30_days():
    try:
        # Load the pre-trained model
        trained_model = load_model('./trained_model.h5')  # Load the model each time

        # Fetch the full dataset
        full_data = await full_dataset()

        # Check if 'data' is present
        if 'data' not in full_data or not full_data['data']:
            raise HTTPException(status_code=404, detail="No data available for prediction")

        # Prepare the 'Close' prices for scaling
        closedf = pd.DataFrame(full_data['data'])[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        closedf_scaled = scaler.fit_transform(closedf)

        # Use the last time_step data to predict the next 30 days
        time_step = 15
        test_data = closedf_scaled[-time_step:]  # Get the last `time_step` data
        temp_input = test_data.flatten().tolist()  # Flatten to a list for processing

        lst_output = []
        pred_days = 30

        for _ in range(pred_days):
            if len(temp_input) > time_step:
                x_input = np.array(temp_input[-time_step:])  # Use the last `time_step` elements
            else:
                x_input = np.array(temp_input)  # Fallback if less than `time_step`

            x_input = x_input.reshape((1, time_step, 1))  # Reshape for LSTM input
            yhat = trained_model.predict(x_input, verbose=0)  # Make prediction
            temp_input.append(yhat[0][0])  # Append predicted value to input list
            lst_output.append(yhat[0][0])  # Only append the predicted value

        # Inverse scale to get the predicted stock prices
        predicted_stock_price = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)).flatten()

        return {"predicted_stock_price": predicted_stock_price.tolist()}

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
