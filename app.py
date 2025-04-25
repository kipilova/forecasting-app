import streamlit as st
from utils import get_wikipedia_pageviews, prepare_data_csv
from model_gru import run_gru_forecast
from model_arima import run_arima_forecast
from model_lstm import run_lstm_forecast
from model_dwt import run_dwt_forecast
from model_analog import run_analog_forecast
import pandas as pd
import numpy as np
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(layout="wide")
st.title("Прогноз посещаемости сайта с помощью GRU")

# Выбор источника данных
source = st.radio("Источник данных:", ["CSV-файл", "Wikipedia API"])

df = None

if 'df' not in st.session_state:
    st.session_state.df = None

# Загрузка данных #
if source == "CSV-файл":
    uploaded_file = st.file_uploader("Загрузите CSV-файл с колонками 'date' и 'views'", type="csv")
    if uploaded_file:
        st.session_state.df = prepare_data_csv(uploaded_file)
        st.success("Данные загружены!")

elif source == "Wikipedia API":
    article = st.text_input("Название статьи Wikipedia (англ.)", "Machine_learning")
    start_date_txt = st.text_input("Введите начальную дату (формат ГГГГ-ММ-ДД)", "2023-01-01")
    end_date_txt = st.text_input("Введите конечную дату (формат ГГГГ-ММ-ДД)", "2023-12-31")

    start_date = start_date_txt.replace("-", "")
    end_date = end_date_txt.replace("-", "")
    
    if st.button("Загрузить данные"):
        st.session_state.df = get_wikipedia_pageviews(article, start_date, end_date)
        if st.session_state.df is not None:
            st.success(f"Данные успешно получены для статьи {article} с {start_date_txt} по {end_date_txt}")

# Работа с фильтрацией данных #
if st.session_state.df is not None:
    df = st.session_state.df
    df['date'] = pd.to_datetime(df['date'])

    try:
        # Проверяем правильность ввода
        start_date_parsed = pd.to_datetime(start_date)
        end_date_parsed = pd.to_datetime(end_date)

        df_filtered = df[(df['date'] >= start_date_parsed) & (df['date'] <= end_date_parsed)]

        st.write(f"Количество точек в выбранном интервале: {len(df_filtered)}")
        st.line_chart(df_filtered.set_index('date')['views'])

        # === Прогнозы === #
        # Выбор модели
        model = st.radio("Модель:", ["GRU", "ARIMA", "LSTM", "DWT", "Прогнозирование по аналогии"])

        if st.button("Построить прогноз по тестовой выборке"):
            with st.spinner("Модель обучается и делает прогноз..."):
                if model == "GRU":
                    forecast_df = run_gru_forecast(df_filtered, mode='test')
                elif model == "ARIMA":
                    forecast_df = run_arima_forecast(df_filtered, mode='test')
                elif model == "LSTM":
                    forecast_df = run_lstm_forecast(df_filtered, mode='test')
                elif model == "DWT":
                    forecast_df = run_dwt_forecast(df_filtered, mode='test')
                elif model == "Прогнозирование по аналогии":
                    forecast_df = run_analog_forecast(df_filtered, mode='test')

            st.success("Прогноз построен!")
            st.subheader("Прогноз по тестовой выборке")
            st.line_chart(forecast_df.set_index('date')[['real', 'forecast']])

            # Вычисление метрик
            if 'real' in forecast_df.columns and 'forecast' in forecast_df.columns:
                real_values = forecast_df['real'].dropna().values
                forecast_values = forecast_df['forecast'].dropna().values

                mse = mean_squared_error(real_values, forecast_values)
                mae = mean_absolute_error(real_values, forecast_values)
                r2 = r2_score(real_values, forecast_values)
                mape = np.mean(np.abs((real_values - forecast_values) / real_values)) * 100
                bias = np.mean(real_values - forecast_values)

                st.subheader("Метрики оценки модели")
                st.write(f"MSE (среднеквадратическая ошибка): {mse}")
                st.write(f"MAE (средняя абсолютная ошибка): {mae}")
                st.write(f"R² (коэффициент детерминации): {r2}")
                st.write(f"MAPE (средняя абсолютная процентная ошибка): {mape:.2f}%")
                st.write(f"Bias (смещение): {bias}")

       

    except Exception as e:
        st.error("Ошибка в формате даты. Пожалуйста, введите в формате 'YYYY-MM-DD'.")
