import numpy as np
import pandas as pd
import time
from pmdarima import auto_arima
import pywt

def run_dwt_arima_forecast(df, mode='test'):
    df = df.copy()

    # === Логарифмирование и очистка выбросов ===
    df['views'] = np.log1p(df['views'])

    Q1 = df['views'].quantile(0.25)
    Q3 = df['views'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df['views'] >= lower) & (df['views'] <= upper)]

    # === Разделение данных ===
    split = int(len(df) * 0.8)
    df_train = df.iloc[:split]
    df_test = df.iloc[split:]

    train_values = df_train['views'].values
    test_values = df_test['views'].values

    # === DWT разложение ===
    wavelet = 'dmey'
    A, D = pywt.dwt(train_values, wavelet)

    # === Обучение моделей ARIMA для A и D ===
    forecast_horizon = min(len(test_values), len(A), len(D))

    start_fit = time.perf_counter()
    model_A = auto_arima(A, seasonal=False, stepwise=True, suppress_warnings=True)
    model_D = auto_arima(D, seasonal=False, stepwise=True, suppress_warnings=True)
    end_fit = time.perf_counter()
    fit_time = end_fit - start_fit

    # === Прогноз ===
    start_pred = time.perf_counter()
    dwt_steps = forecast_horizon // 2
    A_forecast = model_A.predict(n_periods=dwt_steps)
    D_forecast = model_D.predict(n_periods=dwt_steps)
    forecast_log = pywt.idwt(A_forecast, D_forecast, wavelet)
    forecast_final = np.expm1(forecast_log)
    end_pred = time.perf_counter()
    predict_time = end_pred - start_pred

    # === Синхронизация длин ===
    y_true = np.expm1(test_values)[:len(forecast_final)]
    y_pred = forecast_final[:len(y_true)]
    forecast_index = df_test.index[:len(y_true)]

    forecast_df = pd.DataFrame({
        'date': forecast_index,
        'real': y_true,
        'forecast': y_pred
    }).reset_index(drop=True)

    return forecast_df, fit_time, predict_time
