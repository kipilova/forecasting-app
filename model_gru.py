import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time

# Преобразование данных в окна
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length][0])
    return np.array(X), np.array(y)

def run_gru_forecast(df, epochs=50, batch_size=32, mode='test', future_days=365):

    df['day_of_week'] = df.index.dayofweek

    df['views'] = np.log1p(df['views'])

    # Нормализация данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    views_data_scaled = scaler.fit_transform(df['views'].values.reshape(-1, 1))

    scaler_dow = MinMaxScaler()
    dow_data_scaled = scaler_dow.fit_transform(df['day_of_week'].values.reshape(-1, 1))
    data_scaled = np.hstack((views_data_scaled, dow_data_scaled))
    input_shape = (30, 2)

    seq_length = 30  # от этого зависит точность
    if len(data_scaled) <= seq_length:
        print("Недостаточно данных для создания последовательностей.")
        return None
    X, y = create_sequences(data_scaled, seq_length)

    # Разделение данных на тренировочные и тестовые
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Построение модели GRU #
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        GRU(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # === Обучение ===
    start_fit = time.perf_counter()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),
              callbacks=[early_stop], verbose=0)
    end_fit = time.perf_counter()
    fit_time = end_fit - start_fit

    if mode == 'test':
        # === Прогноз ===
        start_pred = time.perf_counter()
        y_pred = model.predict(X_test)
        end_pred = time.perf_counter()
        predict_time = end_pred - start_pred

        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_rescaled = scaler.inverse_transform(y_pred)

        y_pred_final = np.expm1(y_pred_rescaled).flatten()
        y_true_final = np.expm1(y_test_rescaled).flatten()

        forecast_dates = df.index[-len(y_test):].to_series().reset_index(drop=True)
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'real': y_true_final,
            'forecast': y_pred_final
        })
        return forecast_df, fit_time, predict_time

    else:
        raise ValueError("mode должен быть 'test' или 'future'")
