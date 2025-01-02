import pandas as pd
import datetime
from config import brokerage_account
from tinkoff.invest import Client, CandleInterval, HistoricCandle, RequestError
from tinkoff.invest.constants import INVEST_GRPC_API


figi_dict = {
    'ROSN': 'BBG004731354',
    'GAZP': 'BBG004730RP0',
    'NVTK': 'BBG00475KKY8',
    'LKOH': 'BBG004731032',
    'SNGS': 'BBG0047315D0',
    'TATN': 'BBG004RVFFC0'
}

def convert_MoneyValue(s):
    # преобразование данных типа MoneyValue во float
    return s.units + (s.nano/1e9)


def data_processing(candles: [HistoricCandle]):
    df = pd.DataFrame([{
        'open': convert_MoneyValue(p.open),
        'close': convert_MoneyValue(p.close),
        'high': convert_MoneyValue(p.high),
        'low': convert_MoneyValue(p.low),
        'time': p.time,
        'volume': p.volume
    } for p in candles])
    return df

def get_data(figi_dict):
    try:
        all_df = []
        for figi_keys, figi_values in figi_dict.items():
            with Client(brokerage_account, target=INVEST_GRPC_API) as c:
                request = c.get_all_candles(
                    figi=figi_values,
                    from_=datetime.datetime(2006, 1, 1),
                    to=datetime.datetime.now(),
                    interval=CandleInterval.CANDLE_INTERVAL_DAY
                )
                df = data_processing(request)
                df['company'] = figi_keys
                all_df.append(df)
        combined_df = pd.concat(all_df, ignore_index=True)
        combined_df.to_csv(f"data.csv", encoding='utf-8',
                  index=False, sep=';')

    except RequestError as e:
        print(str(e))

get_data(figi_dict)
