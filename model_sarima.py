from pmdarima import auto_arima
import time
import numpy as np
import pandas as pd

def run_sarima_forecast(df, mode='test'):
    df = df.copy()

    # === Подготовка данных ===
    df['views'] = np.log1p(df['views'])  # логарифмируем

    # Удаление выбросов
    Q1 = df['views'].quantile(0.25)
    Q3 = df['views'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_clean = df[(df['views'] >= lower) & (df['views'] <= upper)]

    # Разделение на train/test
    split = int(len(df_clean) * 0.8)
    df_train = df_clean.iloc[:split]
    df_test = df_clean.iloc[split:]

    y_train = df_train['views'].values
    y_test = df_test['views'].values

    # === Обучение модели ===
    model = auto_arima(
        y_train,
        seasonal=True,
        m=7,  # недельная сезонность
        stepwise=True,
        trace=True,
        suppress_warnings=True,
        start_p=0, start_q=0, start_P=0, start_Q=0,
        max_p=3, max_q=3, max_d=2,
        max_P=2, max_Q=2, max_D=1,
        error_action='ignore',
    )
    start_fit = time.perf_counter()
    model.fit(y_train)
    end_fit = time.perf_counter()
    fit_time = end_fit - start_fit

    # === Прогноз ===
    start_pred = time.perf_counter()
    forecast_log = model.predict(n_periods=len(y_test))
    end_pred = time.perf_counter()
    predict_time = end_pred - start_pred

    # Обратное логарифмирование
    y_pred = np.expm1(forecast_log)
    y_true = np.expm1(y_test)

    # Подготовка DataFrame для вывода и графиков
    forecast_df = pd.DataFrame({
        'date': df_test.index[:len(y_true)],
        'real': y_true,
        'forecast': y_pred
    }).reset_index(drop=True)

    return forecast_df, fit_time, predict_time

