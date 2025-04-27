# Прогнозирование посещаемости Wikipedia-статей

Веб-приложение разработанное с использованием Streamlit, предназначено для прогнозирования посещаемости Wikipedia-страниц с помощью различных моделей машинного обучения.
В текущей версии реализована модель GRU (Gated Recurrent Unit), в дальнейшем планируется добавление ARIMA, LSTM, DWT и аналогичных методов.

## Как запустить локально
1. Клонируйте репозиторий:
   git clone https://github.com/yourusername/forecasting-app.git
   cd forecasting-app
2. Создайте и активируйте виртуальное окружение (не обьязательно):
   python -m venv venv
   source venv/bin/activate  # Linux / macOS
   venv\Scripts\activate     # Windows
3. Установите зависимости:
   pip install -r requirements.txt
4. Запустите приложение:
   streamlit run app.py

## Онлайн-версия
Приложение развернуто на Streamlit Community Cloud и доступно по ссылке:
https://forecasting-app-kipilova.streamlit.app

## Возможности приложения
- Загрузка данных из CSV-файла или с Wikipedia API (по названию статьи и датам).
- Выбор модели для прогнозирования (сейчас доступна только GRU).
- Прогноз по тестовой выборке.
- Визуализация реальных данных и прогноза.
- Вычисление метрик качества.

## Пример работы приложения

![image](https://github.com/user-attachments/assets/40d1ab73-ce55-4b17-b000-2e58a4e67ceb)
![image](https://github.com/user-attachments/assets/61601dc0-638d-432e-a4a9-f33f9aea4c88)
![image](https://github.com/user-attachments/assets/219453a6-19fd-45a5-a34e-fb26c4e27018)
![image](https://github.com/user-attachments/assets/93867da0-09b3-4372-afa9-77cd75b82559)
![image](https://github.com/user-attachments/assets/1eb02763-8b03-4a3c-bfdb-29583747a23c)




