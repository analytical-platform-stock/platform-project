import requests
import csv
import time

def fetch_news(ticker, headers):
    """Получаем json с новостями по компании"""
    url = f'https://be.broker.ru/bcsexpress-partners/bcsexpress-article-aggregate/api/v2/articles?instruments%5B0%5D=TQBR%3A{ticker}&limit=9000&lastValue=7ba6e894-a345-4ac2-baf5-210c5f865e02'
    response = requests.get(url, headers=headers)
    time.sleep(5)  # Установка задержки между запросами
    return response.json()

def parse_news_item(news_item, ticker):
    """Парсит новость в словарь"""
    return {
        'title_news': news_item.get('title', ''),
        'link_news': news_item.get('hyperLink', ''),
        'announce': news_item.get('announce', ''),
        'publish_date': news_item.get('publishDate', ''),
        'rubric': news_item.get('rubric', {}).get('name', '') if news_item.get('rubric') else '',
        'ticket': ticker
    }

def collect_news(tickers, headers):
    """Собирает новости для списка тикеров и возвращает список с новостями"""
    all_news = []
    for ticker in tickers:
        response = fetch_news(ticker, headers)
        for item in response.get('data', []):
            parsed_news = parse_news_item(item, ticker)
            all_news.append(parsed_news)
    return all_news

def save_news_to_csv(news, filename):
    with open(filename, 'w', newline='', encoding='UTF-8-sig') as file:
        fieldnames = news[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(news)

def main():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 YaBrowser/24.10.0.0 Safari/537.36'}
    tickers = ['ROSN', 'GAZP', 'LKOH', 'NVTK', 'SNGS', 'TATN']
    news = collect_news(tickers, headers)
    save_news_to_csv(news, 'news_bcs_express.csv')

if __name__ == "__main__":
    main()
