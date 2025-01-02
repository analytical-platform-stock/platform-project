from newsapi import NewsApiClient
from datetime import date, timedelta
import pandas as pd
import json


with open("newsapi_sources.json") as file:
    file = json.load(file)

newsapi = NewsApiClient(api_key="9788d84de83449e99e1dc0493ed97650")

ticket_dict = {
    "en": [
        "Rosneft",
        "Gazprom",
        "NOVATEK",
        "LUKOIL",
        "SNGS",
        "Tatneft"
    ],
    "ru": [
        "Роснефть",
        "Газпром",
        "НОВАТЭК",
        "Лукоил",
        "Сургутнефтегаз",
        "Татнефть"
    ]
}


def parse(lang):
    source_list = ', '.join(file[lang])
    df = pd.DataFrame(columns=["ticket", "title", "description", "publishedAt", "source", "link"])
    my_date = date.today() - timedelta(days = 30)
    for ticket in ticket_dict[lang]:
        articles = newsapi.get_everything(q=ticket,
                                          from_param = my_date.isoformat(),
                                          sort_by="relevancy")
        en_source_articles = [item for item in articles["articles"] if item["source"]["name"] in source_list]
        df_iter = pd.DataFrame({"ticket": ticket,
                                "title": [x["title"] for x in en_source_articles],
                                "description": [x["description"] for x in en_source_articles],
                                "publishedAt": [x["publishedAt"] for x in en_source_articles],
                                "source": [x["source"]["name"] for x in en_source_articles],
                                "link": [x["url"] for x in en_source_articles]})
        df = pd.concat([df, df_iter], ignore_index=True)

    return df



newsapi_ru_data = parse("ru")
newsapi_ru_data.to_csv("newsapi_ru_data.csv", encoding='utf-8',
                       index=False, sep=';')
newsapi_en_data = parse("en")
newsapi_en_data.to_csv("newsapi_en_data.csv", encoding='utf-8',
                       index=False, sep=';')
