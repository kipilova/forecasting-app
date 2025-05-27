import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
import pywt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences_multifeature(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

def run_dwt_lstm_forecast(df, mode='test', seq_length=10):
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Индекс датафрейма должен быть pd.DatetimeIndex")

    df['day_of_week'] = df.index.dayofweek
    df['views'] = np.log1p(df['views'])

    # DWT
    wavelet = 'dmey'
    A, D = pywt.dwt(df['views'].values, wavelet)
    dow = df['day_of_week'].values[:len(A)]

    scaler_A = MinMaxScaler()
    scaler_D = MinMaxScaler()
    scaler_dow = MinMaxScaler()

    A_scaled = scaler_A.fit_transform(A.reshape(-1, 1))
    D_scaled = scaler_D.fit_transform(D.reshape(-1, 1))
    dow_scaled = scaler_dow.fit_transform(dow.reshape(-1, 1))

    combined_data = np.hstack((A_scaled, D_scaled, dow_scaled))

    # разделение данных
    split = int(0.8 * len(combined_data))
    train_data = combined_data[:split]
    test_data = combined_data[split:]

    X_train, y_train = create_sequences_multifeature(train_data, seq_length)
    X_test, y_test = create_sequences_multifeature(test_data, seq_length)

    # === Обучение ===
    start_fit = time.perf_counter()
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)
    end_fit = time.perf_counter()
    fit_time = end_fit - start_fit

    # === Прогноз ===
    start_pred = time.perf_counter()
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    end_pred = time.perf_counter()
    predict_time = end_pred - start_pred

    # Обратное масштабирование и логарифмирование
    y_pred = scaler_A.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_A.inverse_transform(y_test.reshape(-1, 1)).flatten()

    y_pred_final = np.expm1(y_pred)
    y_true_final = np.expm1(y_true)

    # Подготовка дат
    start_index = seq_length + split
    forecast_dates = df.index[start_index:start_index + len(y_pred_final)]

    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'real': y_true_final,
        'forecast': y_pred_final
    }).reset_index(drop=True)

    return forecast_df, fit_time, predict_time
