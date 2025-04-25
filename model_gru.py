import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Преобразование данных в окна
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def run_gru_forecast(df, epochs=50, batch_size=32, mode='test', future_days=365):

    # Нормализация данных
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df['views'].values.reshape(-1, 1))

    seq_length = 60  # от этого зависит точность
    X, y = create_sequences(data_scaled, seq_length)

    # Разделение данных на тренировочные и тестовые
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Построение модели GRU #
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=(seq_length, 1)),  # Первый GRU слой
        GRU(50, return_sequences=False),  # Второй GRU слой
        Dense(25),  # Полносвязный слой
        Dense(1)  # Выходной слой
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)

    if mode == 'test':
        y_pred = model.predict(X_test)
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_rescaled = scaler.inverse_transform(y_pred)

        forecast_dates = df['date'].iloc[-len(y_test):].reset_index(drop=True)
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'real': y_test_rescaled.flatten(),
            'forecast': y_pred_rescaled.flatten()
        })
        return forecast_df

    else:
        raise ValueError("mode должен быть 'test' или 'future'")
