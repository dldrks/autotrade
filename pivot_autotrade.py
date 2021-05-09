import time
import pyupbit
import datetime

access = ""
secret = ""




def get_start_time(ticker):
    """시작 시간 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
    start_time = df.index[0]
    return start_time


def get_pivot(ticker):
    '''피봇포인트'''
    df = pyupbit.get_ohlcv(ticker, interval="minute10", count=1)
    pivot = (df.iloc[0]['close']+ df.iloc[0]['high'] + df.iloc[0]['low'])/3
    return pivot


def get_s_pivot(ticker):
    '''1차 지지선'''
    df = pyupbit.get_ohlcv(ticker, interval="minute10", count=1)
    s_pivot = ((df.iloc[0]['close']+ df.iloc[0]['high'] + df.iloc[0]['low'])/3)*2 - df.iloc[0]['high']
    return s_pivot

def get_r_pivot(ticker):
    '''1차 저항선'''
    df = pyupbit.get_ohlcv(ticker, interval="minute10", count=1)
    support_pivot = ((df.iloc[0]['close']+ df.iloc[0]['high'] + df.iloc[0]['low'])/3)*2 - df.iloc[0]['low']
    return support_pivot

def get_balance(ticker):
    """잔고 조회"""
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == ticker:
            if b['balance'] is not None:
                return float(b['balance'])
            else:
                return 0

def get_current_price(ticker):
    """현재가 조회"""
    return pyupbit.get_orderbook(tickers=ticker)[0]["orderbook_units"][0]["ask_price"]

# 로그인
upbit = pyupbit.Upbit(access, secret)
print("autotrade start")

# 자동매매 시작
while True:
    try:
        now = datetime.datetime.now()
        start_time = get_start_time("KRW-ETH")
        end_time = start_time + datetime.timedelta(days=1)

        if start_time < now < end_time - datetime.timedelta(seconds=10):
            pivot = get_pivot("KRW-ETH")
            r_pivot = get_r_pivot("KRW-ETH")
            s_pivot = get_s_pivot("KRW-ETH")
            current_price = get_current_price("KRW-ETH")
            avg_buy_price = upbit.get_avg_buy_price("ETH")
            if s_pivot > current_price:
                krw = get_balance("KRW")
                if krw > 5000: 
                    upbit.buy_market_order("KRW-ETH", krw*0.9995)
            elif avg_buy_price < r_pivot < current_price:
                  eth = get_balance("ETH")
                  if eth > 0.002:
                        upbit.sell_market_order("KRW-ETH", eth*0.9995)
            
        else:
            eth = get_balance("ETH")
            if eth > 0.002:
                upbit.sell_market_order("KRW-ETH", eth*0.9995)
        time.sleep(1)
    except Exception as e:
        print(e)
        time.sleep(1)
