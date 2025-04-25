import pandas as pd
import requests

# Сбор данных #
def get_wikipedia_pageviews(article, start_date, end_date, project="en.wikipedia"):
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/all-agents/{article}/daily/{start_date}/{end_date}"
    headers = {'User-Agent': 'WikipediaPageviewsCollector/1.0 (akipilova@gmail.com)'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        dates = [item['timestamp'][:8] for item in data['items']]
        views = [item['views'] for item in data['items']]
        df = pd.DataFrame({'date': pd.to_datetime(dates), 'views': views})
        return df
    else:
        print("Ошибка при запросе данных:", response.status_code)
        return None

def prepare_data_csv(file):
    df = pd.read_csv(file)
    df.columns = ["date", "views"]
    df["date"] = pd.to_datetime(df["date"])
    return df
