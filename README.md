# Прогнозирование посещаемости веб-сайтов

Прототип веб-приложения разработан с использованием Streamlit, предназначен для просмотра работы различных моделей для прогнозирования посещаемости веб-сайтов, в том числе ARIMA, SARIMA, LSTM, GRU, ARIMA+DWT, LSTM+DWT и прогнозирование по аналогии. Архитектура сайта позволяет загрузить .csv файл с датасетом или вставить название Wikipedia-страницы и временный интервал, из которого будут извлекаться данные о посещаемости из API Wikipedia.

## Как запустить локально
1. Клонируйте репозиторий:
   
   git clone https://github.com/yourusername/forecasting-app.git
   
   cd forecasting-app
   
3. Создайте и активируйте виртуальное окружение (не обьязательно):
   
   python -m venv venv
   
   source venv/bin/activate  # Linux / macOS
   
   venv\Scripts\activate     # Windows
   
5. Установите зависимости:
   
   pip install -r requirements.txt
   
7. Запустите приложение:
   
   streamlit run app.py

## Онлайн-версия
Приложение развернуто на Streamlit Community Cloud и доступно по ссылке:

https://forecasting-app-kipilova.streamlit.app

## Возможности приложения
- Загрузка данных из CSV-файла или с Wikipedia API (по названию статьи и датам).
- Выбор модели для прогнозирования.
- Прогноз по тестовой выборке.
- Визуализация реальных данных и прогноза.
- Вычисление метрик точности и качества.

## Пример работы приложения

![image](https://github.com/user-attachments/assets/6295d75f-3a3d-4114-8ad9-2ec5f9853c1d)
![image](https://github.com/user-attachments/assets/c4c63d26-0224-47ba-8a4c-524314111b1c)
![image](https://github.com/user-attachments/assets/b35a3316-fa6d-4cb6-8d81-9b80713dceab)
![image](https://github.com/user-attachments/assets/834b1aa3-1089-47f2-aead-47776717beb6)
![image](https://github.com/user-attachments/assets/467173e6-202a-4ccd-8d85-71e447b39c95)
![image](https://github.com/user-attachments/assets/8dc819e7-973c-4212-bebf-832e8aa5aa8c)
![image](https://github.com/user-attachments/assets/d8bff5e3-515d-448b-8a3f-9c371f6e943b)

