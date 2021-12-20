# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv
import datetime
import json
import sys
import time

import requests
import twint
import dateutil.parser as parser

import pandas


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def get_csv():
    companies = ["Tesla",
                 "Apple",
                 "Google",
                 "Bitcoin",
                 "Facebook"
                 "Samsung",
                 "Amazon"]
    ind = 1
    for company in companies:
        c = twint.Config()
        c.Lang = "en"
        c.Verified = True
        c.Search = f"@{company}"
        if company == "Bitcoin":
            c.Search = f"#{company}"
        c.Limit = 500
        c.Min_likes = 1000
        c.Store_csv = True
        c.Output = "file.csv"
        while len(list(csv.reader(open("file.csv")))) < ind * c.Limit:
            try:
                twint.run.Search(c)
                c.Resume = "file.csv"
            except:
                c.Resume = "file.csv"
        ind += 1


def read_dataset(dataset_name):
    with open(dataset_name, "r") as f:
        return list(map(lambda x: x.strip().split(','), f.readlines()))


def load_prices(company, year, month, api_key):
    with open(f'prices_{company}_{year}_{month}.csv', 'a') as f:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={company}&slice=year{year}month{month}&interval=1min&apikey={api_key}'
        r = requests.get(url)
        if "Thank you for using Alpha Vantage" in r.text:
            print("HUI", company, year, month, api_key)
        f.write(r.text)


def binary_search(a, x, lo=0, hi=None):
    if hi is None:
        hi = len(a)
    while (hi - lo) > 1:
        mid = (lo + hi) // 2
        midval = parser.parse(a[mid][0])
        if midval > x:
            lo = mid + 1
        else:
            hi = mid
    return lo


companies = [("tesla", "TSLA"), ("apple", "AAPL"), ("facebook", "FB"), ("google", "GOOGL"), ("amazon", "AMZN")]

datasets = {stock: read_dataset(f"prices_{stock}.csv") for _, stock in companies}


def get_price(company, time):
    tweet_time = datetime.datetime.utcfromtimestamp(parser.parse(time).timestamp()).replace(second=0)
    # print(tweet_time)

    if company not in datasets:
        return None, None
    dtsset = datasets[company]

    pos = binary_search(dtsset, tweet_time)
    return dtsset, pos


def get_company(tweet_text, tweet_author):
    for company_name, stock_name in companies:
        # print(tweet_text, company_name, tweet_author)
        if company_name == tweet_author or company_name in tweet_text:
            return stock_name, company_name
    print("TI HUI", file=sys.stderr)
    return "TI HUI", "Gavno"


if __name__ == '__main__':

    with open("dataset.csv", "w") as f:
        tweets = pandas.read_csv("file.csv").values.tolist()
        for tweet in tweets:
            tweet_text = tweet[10].lower()
            tweet_author = tweet[8].lower()
            tweet_date = tweet[2]

            stock, company = get_company(tweet_text, tweet_author)

            dtst, pos = get_price(stock, tweet_date)
            if dtst is None:
                continue
            if pos+5 >= len(dtst):
                continue
            f.write(",".join([f'"{tweet_text}"', company, str((float(dtst[pos - 5][2]) + float(dtst[pos - 5][2])) / 2),
                              str((float(dtst[pos + 5][2]) + float(dtst[pos + 5][2])) / 2)]) + "\n")

        # print(tweet_text, dtst[pos])

    # companies = ["TSLA", "AAPL", "MSFT"]

    # keys = ["301WO9K1YQ97AVVN", "CULGKENKCCM9LW82", "TTPZJJ506CIXEN0U", "NVH5GZFABN3IGX4K", "0BOFTJFOQNUMKP3C",
    #         "33SS5F2NRROWMKSB", "EM1L7GOWOV4QZB78", "1XKF6GK39K8A4D8G", "AUUI5WZS13XS6FOA", "PWX810YR8BFF1EGK",
    #         "3T4QLEIZ0QA0HXIE", "2ZMS069BXQBZCBSW", "HIVN74LGX8JV4FZ8", "L5S1U57Y8LCJC92W", "HDALGYBSXCFD4TOR",
    #         "CKY4KGMWKWHDMTAX", "730H56GXTYX7CSE7", "V4I4WPU0QPES4FOT", "53ELYJTX5N53YFO1", "J4IK46J8WHKRVRMJ",
    #         "11EOYJF3SYAR3BSO", "QQ2CO6HISV62Q28M", "V4FMXHGP6EXVN3HV", "7M8NKC06NXRUUM5C", "APYR5WLDPC7QGDFL",
    #         "KMADT346UPLSF96H", "GXQU7FO19QGEHFMY", "KSQGHNTM8V4TCOGN", "7GP3F7A9FBXTOF47", "Q09DFH0OQEDEWQ9T"]
    # ind = 0
    # keys_ind = 0
    # for company in companies:
    #     for year in range(1, 3):
    #         for month in range(1, 13):
    #             ind += 1
    #             if (ind == 6):
    #                 ind = 0
    #                 time.sleep(60)
    #             keys_ind = (keys_ind + 1) % len(keys)
    #             print(company, year, month, keys[keys_ind])
    #             load_prices(company, year, month, keys[keys_ind])

    # load_prices("TSLA", 2, 5, keys[0])
    # load_prices("TSLA", 2, 11, keys[0])
    # load_prices("AAPL", 1, 5, keys[0])
    # load_prices("AAPL", 1, 11, keys[0])
    # load_prices("AAPL", 2, 11, keys[0])
    # load_prices("MSFT", 1, 5, "hui")
    # load_prices("MSFT", 2, 5, keys[0])
    # load_prices("MSFT", 2, 11, keys[0])

    # load_prices("MSFT", 1, 1, keys[0])

    # for i in range(6, 11):
    #     load_prices("TSLA", 1, i, "301WO9K1YQ97AVVN")
    # dataset = read_dataset("file.csv")
    # get_price("TSLA", dataset[120][2])
