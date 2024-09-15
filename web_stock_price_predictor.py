import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Stock Price Predictor App")

#stock = st.text_input("Enter the Stock ID", "GOOG")
#Get ID stock from URL
query_params = st.experimental_get_query_params()
stock = query_params.get("stock", ["GOOG"])[0]  # Mặc định là "GOOG" nếu không có

st.write(f"Đang dự đoán cho mã cổ phiếu: {stock}")

from datetime import datetime, timedelta
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

google_data = yf.download(stock, start, end)

model = load_model("Latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data,0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# ploting_data = pd.DataFrame(
#  {
#     'original_test_data': inv_y_test.reshape(-1),
#     'predictions': inv_pre.reshape(-1)
#  } ,
#     index = google_data.index[splitting_len+100:]
# )
# st.subheader("Original values vs Predicted values")
# st.write(ploting_data)

# st.subheader('Original Close Price vs Predicted Close price')
# fig = plt.figure(figsize=(15,6))
# plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data], axis=0))
# plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
# st.pyplot(fig)

# Plot original vs predicted values
ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    } ,
    index = google_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)

# Predict the next 10 days
st.subheader("Predicted Stock Prices for the Next 10 Days")
last_100_days = scaled_data[-100:]  # Last 100 days of scaled data
# Giả sử current_date là ngày hiện tại
current_date = datetime.now()
# Số ngày dự đoán
prediction_days = 10
# Biến để đếm số lần lặp
counter = 0
# Danh sách để lưu trữ dự đoán cho các ngày tương lai
future_predictions = []
# Sử dụng last_100_days là dữ liệu đầu vào cuối cùng từ ngày gần nhất trong dữ liệu thực tế
last_100_days = scaled_data[-100:]

while counter < prediction_days:
    # Predict price
    prediction = model.predict(last_100_days.reshape(1, 100, 1))
    future_predictions.append(prediction[0, 0])
    last_100_days = np.append(last_100_days[1:], prediction, axis=0)
    counter += 1
    current_date += timedelta(days=1)

# Đảo chuẩn hóa dự đoán tương lai
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Tạo DataFrame để hiển thị kết quả
future_dates = [datetime.now() + timedelta(days=i) for i in range(1, prediction_days + 1)]
future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Close'])

st.subheader("10-Day Future Predictions")

st.write(future_df)