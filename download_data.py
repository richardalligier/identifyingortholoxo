from traffic.data import aixm_airspaces, opensky
import pandas as pd
import datetime
import argparse
import os
from traffic.data.samples import savan


def daterange(start, end, step=datetime.timedelta(1)):
    curr = start
    while curr < end:
        yield curr
        curr += step

def down_day(day):
    ldf = []
    for i in range(24):
        t1 = f"{day} {i:02d}:00:00+00:00"
        t2 = f"{day} {i:02d}:59:59+00:00"
        # print(t1,t2)
        ldf.append(opensky.history(t1,t2,bounds=aixm_airspaces["LFBBBDX"]).data)
    return pd.concat(ldf,ignore_index=True)


def mainsavan():
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-foldertrajs')
    args = parser.parse_args()
    savan.to_parquet(os.path.join(args.foldertrajs,"savan.parquet"))

def main():
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-foldertrajs')
    args = parser.parse_args()
    start = datetime.datetime(2022,7,14)
    end = datetime.datetime(2022,8,11)
    for day in daterange(start,end):
        date = day.strftime("%Y-%m-%d")
        print(date)
        df = down_day(date)
        df = df.sort_values(by=["callsign","icao24","timestamp"]).reset_index()
        df.to_parquet(os.path.join(args.foldertrajs,date)+".parquet")

mainsavan()
