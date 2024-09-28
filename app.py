from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from joblib import load
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf
import math
import requests
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

stock_data = None

@app.get("/get_stock_data/{ticker}")
async def get_stock_data(ticker: str):
    global stock_data
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol is required")

    # Convert to uppercase, and define the timeframe (timestamps for from/to)
    ticker = ticker.upper()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    # Convert datetime to Unix timestamps
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    # API URL with the new API
    api_url = f"https://histdatafeed.vps.com.vn/tradingview/history?symbol={ticker}&resolution=1D&from={start_timestamp}&to={end_timestamp}"

    try:
        response = requests.get(api_url)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Failed to fetch data for ticker: {ticker}")

        # Parse the response
        data = response.json()

        # If no data, return 404
        if data['s'] != 'ok':
            raise HTTPException(status_code=404, detail=f"No data found for ticker: {ticker}")

        # Transform data into the desired format
        stock_data = [
            {
                'date': datetime.utcfromtimestamp(t).strftime('%Y-%m-%d'),
                'open': o,
                'high': h,
                'low': l,
                'close': c,
                'volume': v
            }
            for t, o, h, l, c, v in zip(data['t'], data['o'], data['h'], data['l'], data['c'], data['v'])
        ]

        return JSONResponse(content={'historicalData': stock_data})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/full-dataset")
async def full_dataset():
    try:
        if stock_data is None:
            raise HTTPException(status_code=404, detail="No stock data available. Please fetch data first.")

        filtered_data = [{'date': record['date'], 'close': record['close']} for record in stock_data]
        return {"data": filtered_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Define model with the structure requested
model = Sequential([
        LSTM(200, input_shape=(5, 1), activation=tf.nn.leaky_relu, return_sequences=True),
        LSTM(200, activation=tf.nn.leaky_relu),
        Dense(200, activation=tf.nn.leaky_relu),
        Dense(100, activation=tf.nn.leaky_relu),
        Dense(50, activation=tf.nn.leaky_relu),
        Dense(5, activation=tf.nn.leaky_relu)
    ])
# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

    # Define the learning rate scheduler
def scheduler(epoch):
        if epoch <= 150:
            lrate = (10 ** -5) * (epoch / 150)
        elif epoch <= 400:
            initial_lrate = (10 ** -5)
            k = 0.01
            lrate = initial_lrate * math.exp(-k * (epoch - 150))
        else:
            lrate = (10 ** -6)
        return lrate
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

def train_model_from_full_data(full_data):
        closedf = pd.DataFrame(full_data)[['close']]

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

        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=32, callbacks=[callback], verbose=1)

        model.save('./trained_model.h5')
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
            closedf = pd.DataFrame(full_data['data'])[['close']]
            scaler = MinMaxScaler(feature_range=(0, 1))
            closedf_scaled = scaler.fit_transform(closedf)

            # Use the last time_step data to predict the next 30 days
            time_step = 15
            test_data = closedf_scaled[-time_step:]  # Get the last time_step data
            temp_input = test_data.flatten().tolist()  # Flatten to a list for processing

            lst_output = []
            pred_days = 30

            for _ in range(pred_days):
                if len(temp_input) > time_step:
                    x_input = np.array(temp_input[-time_step:])  # Use the last time_step elements
                else:
                    x_input = np.array(temp_input)  # Fallback if less than time_step

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

    # Run the app
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
