import numpy as np
import pandas as pd
import time

def run_analog_forecast(df, mode='test'):
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Индекс датафрейма должен быть pd.DatetimeIndex")

    # Логарифмирование
    df['views'] = np.log1p(df['views'])

    # Разделение на train/test
    split = int(len(df) * 0.8)
    df_train = df.iloc[:split]
    df_test = df.iloc[split:]

    start_fit = time.perf_counter()

    predictions = []
    for current_date in df_test.index:
        weekday = current_date.weekday()
        month = current_date.month
        day = current_date.day

        analog_days = df_train[
            (df_train.index.month == month) &
            (df_train.index.day == day) &
            (df_train.index.weekday == weekday)
        ]

        if analog_days.empty:
            analog_days = df_train[
                (df_train.index.month == month) &
                (df_train.index.weekday == weekday)
            ]
        if analog_days.empty:
            analog_days = df_train[df_train.index.weekday == weekday]

        pred = analog_days['views'].mean()
        predictions.append(pred)

    end_fit = time.perf_counter()
    fit_time = end_fit - start_fit

    start_pred = time.perf_counter()
    y_pred = np.array(predictions)
    y_true = df_test['views'].values
    end_pred = time.perf_counter()
    predict_time = end_pred - start_pred

    # Обратное логарифмирование
    y_pred_final = np.expm1(y_pred)
    y_true_final = np.expm1(y_true)

    forecast_df = pd.DataFrame({
        'date': df_test.index,
        'real': y_true_final,
        'forecast': y_pred_final
    }).reset_index(drop=True)

    return forecast_df, fit_time, predict_time