from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from datetime import datetime
import requests
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input

app = FastAPI()

stock_data = None
scaler = MinMaxScaler(feature_range=(0, 1))  # Đặt scaler làm biến toàn cục

@app.get("/get_stock_data/{ticker}")
async def get_stock_data(ticker: str):
    global stock_data
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol is required")

    ticker = ticker.upper()
    end_date = datetime.now()
    start_date = datetime(2019, 1, 1)
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    api_url = f"https://histdatafeed.vps.com.vn/tradingview/history?symbol={ticker}&resolution=1D&from={start_timestamp}&to={end_timestamp}"

    try:
        response = requests.get(api_url)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail=f"Failed to fetch data for ticker: {ticker}")

        data = response.json()
        if data['s'] != 'ok':
            raise HTTPException(status_code=404, detail=f"No data found for ticker: {ticker}")

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

def create_model():
    model = Sequential([
        LSTM(200, input_shape=(5, 1), activation=tf.nn.leaky_relu, return_sequences=True),
        LSTM(200, activation=tf.nn.leaky_relu),
        Dense(200, activation=tf.nn.leaky_relu),
        Dense(100, activation=tf.nn.leaky_relu),
        Dense(50, activation=tf.nn.leaky_relu),
        Dense(1, activation=tf.nn.leaky_relu)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model

def scheduler(epoch):
    if epoch <= 150:
        return (10 ** -5) * (epoch / 150)
    elif epoch <= 400:
        return (10 ** -5) * math.exp(-0.01 * (epoch - 150))
    else:
        return 10 ** -6

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

def train_model_from_full_data(data):
    global scaler  # Sử dụng biến toàn cục scaler

    # Convert data into DataFrame
    df = pd.DataFrame(data)

    # Convert 'close' column to numpy array and normalize
    close_prices = df['close'].values.reshape(-1, 1)

    # Normalize using MinMaxScaler
    scaled_close_prices = scaler.fit_transform(close_prices).astype(np.float32)

    # Split data into train/test (80% train)
    training_size = int(len(scaled_close_prices) * 0.8)
    train_data = scaled_close_prices[:training_size]

    # Create training dataset with time_step
    time_step = 15
    X_train, y_train = [], []
    for i in range(time_step, len(train_data)):
        X_train.append(train_data[i - time_step:i, 0])
        y_train.append(train_data[i, 0])

    # Convert to numpy array and ensure proper data types
    X_train, y_train = np.array(X_train).astype(np.float32), np.array(y_train).astype(np.float32)

    # Reshape input into [samples, time steps, features] for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Create model and train
    model = create_model()

    # Train with callback for learning rate scheduling
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1, callbacks=[callback])

    # Save the trained model
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


from datetime import timedelta

@app.post("/predict_next_30_days")
async def predict_next_30_days():
    global scaler  # Sử dụng biến toàn cục scaler
    try:
        trained_model = load_model('./trained_model.h5')
        full_data = await full_dataset()
        if 'data' not in full_data or not full_data['data']:
            raise HTTPException(status_code=404, detail="No data available for prediction")

        closedf = pd.DataFrame(full_data['data'])[['close']].values
        time_step = 15
        test_data = closedf[-time_step:]
        temp_input = test_data.flatten().tolist()

        lst_output = []
        pred_days = 30

        for _ in range(pred_days):
            # Ensure correct shape and type
            x_input = np.array(temp_input[-time_step:]).reshape((1, time_step, 1)).astype(np.float32)

            # Predict the next value
            yhat = trained_model.predict(x_input, verbose=0)

            # Append the predicted value
            temp_input.append(float(yhat[0][0]))
            lst_output.append(float(yhat[0][0]))

        # Reshape predicted values to match input for inverse transform
        predicted_stock_price = np.array(lst_output).reshape(-1, 1)

        # Inverse transform to original scale
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

        # Lấy ngày cuối cùng từ dữ liệu gốc
        last_date = datetime.strptime(full_data['data'][-1]['date'], "%Y-%m-%d")

        # Chuẩn bị danh sách kết quả dự đoán với định dạng yêu cầu
        prediction_list = []
        for i in range(pred_days):
            next_date = last_date + timedelta(days=(i + 1))
            prediction_list.append({
                "time": next_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),  # Định dạng thời gian ISO
                "open": 0,
                "high": 0,
                "low": 0,
                "close": round(predicted_stock_price[i][0], 2),  # Giá trị close được dự đoán
                "volume": 0
            })

        return {"predicted_stock_price": prediction_list}

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model file not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
