import pandas as pd

import argparse
from detect_classic import extract_flightplan, NoFlightPlan, DetectOrthodromyWithBeacons
from detect_orthodromy import filter_trajectories

def convert(df):
    for col in df.select_dtypes(include="string[pyarrow]"):
        df[col] = df[col].astype("string")#.astype("string[python]")#.astype("string[python]")  # or "object"
    for col in df.select_dtypes(include="float[pyarrow]"):
        df[col] = df[col].astype("float")#.astype("string[python]")#.astype("string[python]")  # or "object"
    return df


def read_trajectories(filein,queries=None):
    flights = pd.read_parquet(filein)
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
# def read_trajectories(f, strategy):
#     ''' read a trajectory file named @f, and filters points using a @strategy'''
#     df = pd.read_parquet(f)
#     for v in ["flight_id"]:
#         df[v] = df[v].astype(np.int64)
#     df = df.drop_duplicates(["flight_id","timestamp"]).sort_values(["flight_id","timestamp"]).reset_index(drop=True)#.head(10_000)
#     if strategy == "classic":
#         filter = FilterCstLatLon()|FilterCstPosition()|FilterCstSpeed()|MyFilterDerivative()|FilterIsolated()
#     else:
#         raise Exception(f"strategy '{strategy}' not implemented")
#     dftrafficin = Traffic(df).filter(filter=filter,strategy=nointerpolate).eval(max_workers=1).data
#     dico_tomask = {
#         # "track":["track_unwrapped"],
#         "latitude":["u_component_of_wind","v_component_of_wind","temperature"],
#         "altitude":["u_component_of_wind","v_component_of_wind","temperature"],
#     }
#     for k,lvar in dico_tomask.items():
#         for v in lvar:
#             dftrafficin[v] = dftrafficin[[v]].mask(dftrafficin[k].isna())
#     return dftrafficin

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
    flightplans = pd.read_parquet(args.flightplans)
    detector = DetectOrthodromyWithBeacons(flightplans=flightplans,timesplit=args.timesplit)
    def keep_with_fp(df):
        try:
            extract_flightplan(flightplans,df)
            return df
        except NoFlightPlan:
            return pd.DataFrame()
    groupby = ["icao24","callsign","date"]
    def split_and_keep(df):
        return df.groupby(by=groupby).apply(keep_with_fp,include_groups=True).reset_index(drop=True)#.drop(columns="index")
    flights = detector.apply_splitted(flights,split_and_keep)
    print(list(flights))
    if not flights.empty:
        flights = flights.reset_index(drop=True).sort_values(by=groupby).drop(columns=["index","splitted","date","tunix"])
    flights.to_parquet(args.trajsout,index=False)



if __name__ == '__main__':
    main()
