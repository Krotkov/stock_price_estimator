# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import csv

import twint


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
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
        # c.Profile_full = True
        c.Store_csv = True
        c.Output = "file.csv"
        while len(list(csv.reader(open("file.csv")))) < ind * c.Limit:
            try:
                twint.run.Search(c)
                c.Resume = "file.csv"
            except:
                c.Resume = "file.csv"
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        ind += 1
        # print(twint.run.Search(c))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
