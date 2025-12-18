import pandas as pd

import argparse
from detect_classic import FlightPlanDatabase, NoFlightPlan, Detect, filter_trajectories

def convert(df):
    for col in df.select_dtypes(include="string[pyarrow]"):
        df[col] = df[col].astype("string")#.astype("string[python]")#.astype("string[python]")  # or "object"
    for col in df.select_dtypes(include="float[pyarrow]"):
        df[col] = df[col].astype("float")#.astype("string[python]")#.astype("string[python]")  # or "object"
    return df


def read_trajectories(filein,queries=None):
    if filein.endswith(".parquet"):
        flights = pd.read_parquet(filein)
    else:
        flights = pd.read_csv(filein)
    if queries is not None:
        for q in queries:
            flights = flights.query(q)
        flights = flights.reset_index(drop=True)
    todrop = ["onground"]
    for v in todrop:
        if v in flights:
            flights = flights.drop(columns=v)
    flights = convert(flights)
    # print(flights)
    # print(flights.dtypes)
    if flights.empty:
        return flights
    flights["tunix"] = flights["timestamp"].astype(int)//10**9
    flights["date"] = flights["timestamp"].dt.date
    flights = filter_trajectories(flights, "classic")
    flights = flights.dropna(subset=["track","latitude"])
    return flights


def main():
    parser = argparse.ArgumentParser(
        description='filter out measurements that are likely erroneous',
    )
    parser.add_argument("-trajsin",required=True)
    parser.add_argument("-trajsout")
    parser.add_argument("-flightplans",required=True)
    parser.add_argument("-timesplit",type=float,required=True)
    args = parser.parse_args()
    flights = read_trajectories(args.trajsin)
    basedetector = Detect()
    fpdatabase = FlightPlanDatabase(args.flightplans)
    def keep_with_fp(df):
        try:
            fpdatabase.extract_flightplan(df)
            return df
        except NoFlightPlan:
            return pd.DataFrame()
    groupby = ["icao24","callsign","date"]
    def split_and_keep(df):
        return df.groupby(by=groupby).apply(keep_with_fp,include_groups=True).reset_index(drop=True)#.drop(columns="index")
    flights = basedetector.apply_splitted(flights,split_and_keep,args.timesplit)
    print(list(flights))
    if not flights.empty:
        flights = flights.reset_index(drop=True).sort_values(by=groupby).drop(columns=["index","splitted","date","tunix"])
    flights.to_parquet(args.trajsout,index=False)



if __name__ == '__main__':
    main()
