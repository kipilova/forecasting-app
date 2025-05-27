import streamlit as st
from utils import get_wikipedia_pageviews, prepare_data_csv
from model_gru import run_gru_forecast
from model_arima import run_arima_forecast
from model_sarima import run_sarima_forecast
from model_lstm import run_lstm_forecast
from model_arima_dwt import run_dwt_arima_forecast
from model_lstm_dwt import run_dwt_lstm_forecast
from model_analog import run_analog_forecast
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(layout="wide")
st.title("Методы прогнозирования посещаемости веб-сайтов")

source = st.radio("Источник данных:", ["CSV-файл", "Wikipedia API"])

df = None

if 'df' not in st.session_state:
    st.session_state.df = None

if source == "CSV-файл":
    uploaded_file = st.file_uploader("Загрузите CSV-файл с колонками 'date' и 'views'", type="csv")
    if uploaded_file:
        st.session_state.df = prepare_data_csv(uploaded_file)
        st.success("Данные загружены!")

elif source == "Wikipedia API":
    article = st.text_input("Название статьи Wikipedia (англ.)", "Machine learning")
    start_date_txt = st.text_input("Введите начальную дату (формат ГГГГ-ММ-ДД)", "2021-01-01")
    end_date_txt = st.text_input("Введите конечную дату (формат ГГГГ-ММ-ДД)", "2023-12-31")

    api_start = start_date_txt.replace("-", "")
    api_end = end_date_txt.replace("-", "")

    if st.button("Загрузить данные"):
        df = get_wikipedia_pageviews(article, api_start, api_end)
        if df is not None and not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            st.session_state.df = df
            st.success(f"Данные успешно получены для статьи '{article}' с {start_date_txt} по {end_date_txt}")
        else:
            st.error("Не удалось получить данные с Wikipedia API")

# === Отображение данных ===
if st.session_state.df is not None:
    df = st.session_state.df.copy()

    if 'date' not in df.columns and df.index.name == 'date':
        df = df.reset_index()

    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    st.subheader("График просмотров")
    st.line_chart(df['views'])

    # === Прогноз ===
    model = st.radio("Модель:", ["ARIMA", "SARIMA", "LSTM", "GRU", "DWT+ARIMA", "DWT+LSTM", "Прогнозирование по аналогии"])

    if st.button("Построить прогноз по тестовой выборке"):
        with st.spinner("Модель обучается и делает прогноз..."):
            if model == "ARIMA":
                forecast_df, fit_time, predict_time = run_arima_forecast(df, mode='test')
            elif model == "SARIMA":
                forecast_df, fit_time, predict_time = run_sarima_forecast(df, mode='test')
            elif model == "LSTM":
                forecast_df, fit_time, predict_time = run_lstm_forecast(df, mode='test')
            elif model == "GRU":
                forecast_df, fit_time, predict_time = run_gru_forecast(df, mode='test')
            elif model == "DWT+ARIMA":
                forecast_df, fit_time, predict_time = run_dwt_arima_forecast(df, mode='test')
            elif model == "DWT+LSTM":
                forecast_df, fit_time, predict_time = run_dwt_lstm_forecast(df, mode='test')
            elif model == "Прогнозирование по аналогии":
                forecast_df, fit_time, predict_time = run_analog_forecast(df, mode='test')

        # График прогноза
        st.success("Прогноз построен!")
        st.subheader("Прогноз по тестовой выборке")
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        st.line_chart(forecast_df.set_index('date')[['real', 'forecast']])

        # Метрики
        if 'real' in forecast_df.columns and 'forecast' in forecast_df.columns:
            real_values = forecast_df['real'].dropna().values
            forecast_values = forecast_df['forecast'].dropna().values

            mse = mean_squared_error(real_values, forecast_values)
            mae = mean_absolute_error(real_values, forecast_values)
            r2 = r2_score(real_values, forecast_values)
            mape = np.mean(np.abs((real_values - forecast_values) / real_values)) * 100
            bias = np.mean(real_values - forecast_values)

            st.subheader("Метрики оценки модели")
            st.write(f"MSE: {mse}")
            st.write(f"MAE: {mae}")
            st.write(f"R²: {r2}")
            st.write(f"MAPE: {mape:.2f}%")
            st.write(f"Bias: {bias}")
            st.write(f"Время обучения (сек): {fit_time:.3f}")
            st.write(f"Время прогноза (сек): {predict_time:.3f}")