# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv
import datetime
import json

import requests
import twint
import dateutil.parser as parser


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
    with open(f'prices_{company}.csv', 'a') as f:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={company}&slice=year{year}month{month}&interval=1min&apikey={api_key}'
        r = requests.get(url)
        if "Thank you for using Alpha Vantage" in r.text:
            print("HUI", company, year, month, api_key)
        else:
            f.write(r.text)


# def get_prices(company, diff_year, diff_month, api_key):
#     api_key = "33SS5F2NRROWMKSB"
#     url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={company}&slice=year{diff_year}month{diff_month}&interval=1min&apikey={api_key}'
#     r = requests.get(url)
#     return list(map(lambda x: x.strip().split(','), r.text.split("\n")))[1:]


def get_price(company, time):
    tweet_time = datetime.datetime.utcfromtimestamp(parser.parse(time).timestamp()).replace(second=0)
    print(tweet_time)

    cur_time = datetime.datetime.utcnow()
    diff_year = cur_time.year - tweet_time.year + 1
    diff_month = (cur_time.month - tweet_time.month) + 1
    print(cur_time, diff_year, diff_month)
    while diff_month < 0:
        diff_month += 12

    api_key = "33SS5F2NRROWMKSB"
    while True:
        prices = get_prices(company, diff_year, diff_month, api_key)

        first_day = parser.parse(prices[0][0])
        last_day = parser.parse(prices[-2][0])

        if first_day < tweet_time < last_day:
            for i in range(len(prices)):
                try:
                    if tweet_time == parser.parse(prices[i][0]):
                        print(i)
                except:
                    pass
            break
        if tweet_time < first_day:
            diff_month += 1
            if diff_month == 13:
                diff_month = 1
                diff_year += 1
            continue
        if tweet_time > last_day:
            diff_month -= 1
            if diff_month == 0:
                diff_month = 12
                diff_year -= 1
            continue
    # with open("broker_data.csv", "w") as f:
    #     f.write(r.text)

    # print(r.text)


if __name__ == '__main__':
    companies = ["TSLA", "AAPL", "SSNLF", "GOOGL", "FB", "AMZN"]
    keys = ["301WO9K1YQ97AVVN", "CULGKENKCCM9LW82", "TTPZJJ506CIXEN0U", "NVH5GZFABN3IGX4K", "0BOFTJFOQNUMKP3C",
            "33SS5F2NRROWMKSB", "EM1L7GOWOV4QZB78", "1XKF6GK39K8A4D8G", "AUUI5WZS13XS6FOA", "PWX810YR8BFF1EGK",
            "3T4QLEIZ0QA0HXIE", "2ZMS069BXQBZCBSW", "HIVN74LGX8JV4FZ8", "L5S1U57Y8LCJC92W", "HDALGYBSXCFD4TOR",
            "CKY4KGMWKWHDMTAX", "730H56GXTYX7CSE7", "V4I4WPU0QPES4FOT", "53ELYJTX5N53YFO1", "J4IK46J8WHKRVRMJ",
            "11EOYJF3SYAR3BSO", "QQ2CO6HISV62Q28M", "V4FMXHGP6EXVN3HV", "7M8NKC06NXRUUM5C", "APYR5WLDPC7QGDFL",
            "KMADT346UPLSF96H", "GXQU7FO19QGEHFMY", "KSQGHNTM8V4TCOGN", "7GP3F7A9FBXTOF47", "Q09DFH0OQEDEWQ9T"]
    ind = 0
    keys_ind = 0
    for company in companies:
        for year in range(1, 3):
            for month in range(1, 13):
                keys_ind = (keys_ind + 1) % len(keys)
                load_prices(company, year, month, keys[keys_ind])

    # for i in range(6, 11):
    #     load_prices("TSLA", 1, i, "301WO9K1YQ97AVVN")
    # dataset = read_dataset("file.csv")
    # get_price("TSLA", dataset[120][2])
