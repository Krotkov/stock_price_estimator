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
import flair

import pandas

import csv

from sklearn import linear_model


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


COMPANIES = [("tesla", "TSLA"), ("apple", "AAPL"), ("facebook", "FB"), ("google", "GOOGL"), ("amazon", "AMZN")]


def get_market_data():
    from config import KEYS
    # print(tweet_text, dtst[pos])

    companies = ["TSLA", "AAPL", "MSFT"]

    keys = KEYS
    ind = 0
    keys_ind = 0
    for company in companies:
        for year in range(1, 3):
            for month in range(1, 13):
                ind += 1
                if (ind == 6):
                    ind = 0
                    time.sleep(60)
                keys_ind = (keys_ind + 1) % len(keys)
                print(company, year, month, keys[keys_ind])
                load_prices(company, year, month, keys[keys_ind])

    load_prices("TSLA", 2, 5, keys[0])
    load_prices("TSLA", 2, 11, keys[0])
    load_prices("AAPL", 1, 5, keys[0])
    load_prices("AAPL", 1, 11, keys[0])
    load_prices("AAPL", 2, 11, keys[0])
    load_prices("MSFT", 1, 5, "hui")
    load_prices("MSFT", 2, 5, keys[0])
    load_prices("MSFT", 2, 11, keys[0])

    load_prices("MSFT", 1, 1, keys[0])

    for i in range(6, 11):
        load_prices("TSLA", 1, i, "301WO9K1YQ97AVVN")
    dataset = read_dataset("file.csv")
    get_price("TSLA", dataset[120][2])


AUTHORS = {"tesla": 1, "elonmusk": 2, "apple": 3, "tim_cook": 4, "amazon": 5, "unofficial": 6}

DATASETS = {stock: read_dataset(f"prices/prices_{stock}.csv") for _, stock in COMPANIES}

AUTHORS_COLUMNS = [f"author{v}" for v in AUTHORS.values()]


def get_price(company, time):
    tweet_time = datetime.datetime.utcfromtimestamp(parser.parse(time).timestamp()).replace(second=0)
    # print(tweet_time)

    if company not in DATASETS:
        return None, None
    dtsset = DATASETS[company]

    pos = binary_search(dtsset, tweet_time)
    return dtsset, pos


def get_company(tweet_text, tweet_author):
    for company_name, stock_name in COMPANIES:
        # print(tweet_text, company_name, tweet_author)
        if company_name == tweet_author or company_name in tweet_text:
            return stock_name, company_name
    return "unfound", "unfound"

def get_author(tweet_author):
    id = AUTHORS.get(tweet_author, -1)
    if id == -1:
        id = AUTHORS.get("unofficial")
    return tuple(int(id == i) for i in AUTHORS.values())


def ff(a):
    if a > 0.5:
        return 1
    if a < -0.5:
        return -1
    return 0

def fill_df():
    sentiment_model = flair.models.TextClassifier.load('en-sentiment')

    with open("dataset_v2.csv", "w") as f:
        with open("testtest_v2.csv", "w") as f1:
            writer = csv.writer(f)
            writertest = csv.writer(f1)
            writer.writerow(["text", "company", "delta", "sentiment", *AUTHORS_COLUMNS])
            writertest.writerow(["text", "company", "delta", "sentiment", *AUTHORS_COLUMNS])
            tweets = pandas.read_csv("file.csv").values.tolist()
            pos = 0
            for tweet in tweets:
                tweet_text = tweet[10].lower()
                tweet_author = tweet[8].lower()
                tweet_account = tweet[7].lower()
                tweet_date = tweet[2]

                stock, company = get_company(tweet_text, tweet_author)
                author_vectorized = get_author(tweet_account)

                sentence = flair.data.Sentence(tweet_text)
                prediction = sentiment_model.predict(sentence)
                # print(sentence['label'])

                dtst, pos = get_price(stock, tweet_date)
                if dtst is None:
                    continue
                if pos + 24 * 60 >= len(dtst):
                    continue
                score = sentence.labels[0].score
                if sentence.labels[0].value == "NEGATIVE":
                    score = 1 - sentence.labels[0].score
                prev = (float(dtst[pos - 5][2]) + float(dtst[pos - 5][3])) / 2
                next = (float(dtst[pos + 24 * 60][2]) + float(dtst[pos + 24 * 60][3])) / 2
                delta = ff((next - prev) / prev * 100)
                if pos % 10 == 1:
                    writertest.writerow([tweet_text, company, delta, score, *author_vectorized])
                else:
                    writer.writerow([tweet_text, company, delta, score, *author_vectorized])

                pos += 1


def predict_deltas(train_x, test_x, train_y, test_y):
    from sklearn.svm import LinearSVC, SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    # clf.fit(train_x, train_y)
    regr = SVC(tol=10 ** -8)
    regr.fit(train_x, train_y)

    import sklearn.metrics
    pred_company = regr.predict(test_x)
    # clf_pred_company = clf.predict(test_x)

    print(sklearn.metrics.classification_report(test_y, pred_company))
    # print(sklearn.metrics.classification_report(test_y, clf_pred_company))


MODES = {1: "sentiment", 2: "sentiment_author", 3: "fill_df", 4: "text", 5: "text_sentiment"}
PROMPT = [f"{row[0]} for {row[1]}" for row in MODES.items()]


if __name__ == '__main__':
    import numpy as np
    from stonks import get_vectorized_text_attributes

    train_data = pandas.read_csv("dataset_v2.csv")
    test_data = pandas.read_csv("testtest_v2.csv")

    train_x = []
    test_x = []

    train_y = train_data["delta"]
    test_y = test_data["delta"]

    mode = int(input(f"Print the id of mode ({PROMPT})"))


    if mode == 1:
        train_x = np.reshape(train_data["sentiment"].values, (-1, 1))
        test_x = np.reshape(test_data["sentiment"].values, (-1, 1))

        predict_deltas(train_x, test_x, train_y, test_y)
    elif mode == 2:
        train_x = train_data[["sentiment", *AUTHORS_COLUMNS]].values
        test_x = test_data[["sentiment", *AUTHORS_COLUMNS]].values

        predict_deltas(train_x, test_x, train_y, test_y)
    elif mode == 3:
        fill_df()
    elif mode == 4 or mode == 5:
        train_x, test_x, train_y, test_y = get_vectorized_text_attributes(mode)
        print(test_x.shape, test_y.shape)
        print(train_x.shape, train_y.shape)
        predict_deltas(train_x, test_x, train_y, test_y)

