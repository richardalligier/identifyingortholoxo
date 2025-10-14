import pyproj
from pyproj import Geod
import pandas as pd
import numpy as np
import argparse
import os
import time

from traffic.core import Traffic
# from filterclassic import FilterCstLatLon
# from geosphere import orthodromy, loxodromy, distance_ortho_pygplates, distance_without_time_exact,distance_loxo,distance_loxo_ortho#distance_degree
import tqdm
import detect_orthodromy
import detect_classic
import detect_longest
import filter_trajs


class TimeInterval:
    def __init__(self,start,stop):
        self.start = start.timestamp()
        self.stop = stop.timestamp()
    def dt(self):
        return self.stop-self.start
    def __repr__(self):
        return f"{self.start} {self.stop} {self.dt()}"



def main():
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-trajsin')
    parser.add_argument('-detectedout')
    # parser.add_argument('-smooth',type=float)
    # parser.add_argument('-timesplit',type=float)
    # parser.add_argument('-thresh_border',type=float)
    subparsers = parser.add_subparsers(dest="command", required=True)
    # parser_ortholoxo = subparsers.add_parser("ortholoxo", help="Say hello")
    # parser_ortholoxo.add_argument('-track_tolerance_degrees',type=float)
    # parser_ortholoxo.add_argument('-thresh',type=float)
    # parser_ortholoxo.add_argument('-thesh_',type=float)
    parser_longest = subparsers.add_parser("longest", help="Say hello")
    detect_longest.add_parser(parser_longest)
    parser_aligned = subparsers.add_parser("aligned", help="Say hello")
    detect_classic.add_parser(parser_aligned)
    args = parser.parse_args()
    # flights = pd.read_parquet("airacsmall.parquet").query("icao24=='47c1f7'").reset_index()
    # flights = pd.read_parquet("airacsmall.parquet")
    flights = filter_trajs.read_trajectories(args.trajsin)
    if flights.empty:
        flights.to_parquet(args.detectedout)
    else:
        print(flights["date"].unique())
        print(flights)
        #print(df.latitude.isna().mean())
        t0 = time.time()
        # if args.command == "ortholoxo":
        #     detector = detect_orthodromy.DetectOrthodromyLoxodromy(timesplit=args.timesplit,smooth=args.smooth,thresh_border=args.thresh_border,thresh=args.thresh)
        if args.command == "longest":
            kwargs = detect_longest.extract_args(args)
            detector = detect_longest.DetectLongestOrthodromyLoxodromy(**kwargs)
        elif args.command == "aligned":
            kwargs = detect_classic.extract_args(args,extract_flightplan=detect_classic.extract_flightplan)
            detector = detect_classic.DetectOrthodromyWithBeacons(**kwargs)
        else:
            raise Exception
        groupby = ["icao24","callsign","date"]
        res = flights.groupby(by=groupby).apply(detector.apply,include_groups=True)#.reset_index()
        print(time.time()-t0)
        df=res.sort_values(by=["start"]).reset_index(drop=True)
        print(df)
        print(df.dtypes)
        df.to_parquet(args.detectedout)

    # d = {k:[] for k in (["iswhat","start","stop","icao24","callsign","date"]+[f"{v}_{s}" for v in ["altitude","track_angle"] for s in ["start","stop"]])}
    # def process(df,iswhat):
    #     nsegs = df[iswhat].max()
    #     if nsegs>1:
    #         pf = None
    #         for i in range(1,nsegs+1):
    #             nf = df.query(f"{iswhat}==@i")
    #             start = nf.timestamp.min()
    #             stop = nf.timestamp.max()
    #             # print(list(traj))
    #             # print(g)
    #             d["start"].append(int(start.timestamp()))
    #             d["stop"].append(int(stop.timestamp()))
    #             d["iswhat"].append(iswhat)
    #             for k in ["icao24","callsign","date"]:
    #                 d[k].append(g[groupby.index(k)])
    #             for q in ["start","stop"]:
    #                 traj = nf.query(f"timestamp==@{q}")
    #                 assert(traj.shape[0]==1)
    #                 for _,line in traj.iterrows():
    #                     d[f"altitude_{q}"].append(line.altitude)
    #                     d[f"track_angle_{q}"].append(line.track)
    #             # for k in ["dolmax","domax","dlmax"]:
    #             #     d[k].append(line[k])
    #         # for i in range(2,nsegs+1):
    #         #     pf = nf if i > 2 else df.query(f"{iswhat}==1")
    #         #     nf = df.query(f"{iswhat}==@i")
    #         #     start = pf.timestamp.max()
    #         #     stop = nf.timestamp.min()
    #         #     traj = pf.query("timestamp==@start")
    #         #     # print(list(traj))
    #         #     # print(g)
    #         #     assert(traj.shape[0]==1)
    #         #     d["start"].append(int(start.timestamp()))
    #         #     d["stop"].append(int(stop.timestamp()))
    #         #     d["iswhat"].append(iswhat)
    #         #     for k in ["icao24","callsign","date"]:
    #         #         d[k].append(g[groupby.index(k)])
    #         #     for _,line in traj.iterrows():
    #         #         d["altitude"].append(line.altitude)
    #         #         d["track_angle"].append(line.track)
    # for g,resdf in res.groupby(by=groupby):
    #     process(resdf,"is_orthodromy")
    #     process(resdf,"is_loxodromy")
    #     # nsegs = df["is_orthodromy"].max()
    #     # if nsegs>1:
    #     #     pf = None
    #     #     for i in range(2,nsegs+1):
    #     #         pf = nf if i > 2 else df.query("is_orthodromy==1")
    #     #         nf = df.query("is_orthodromy==@i")
    #     #         start = pf.timestamp.max()
    #     #         stop = nf.timestamp.min()
    #     #         traj = pf.query("timestamp==@start")
    #     #         # print(list(traj))
    #     #         # print(g)
    #     #         assert(traj.shape[0]==1)
    #     #         d["start"].append(int(start.timestamp()))
    #     #         d["stop"].append(int(stop.timestamp()))
    #     #         for k in ["icao24","callsign","date"]:
    #     #             d[k].append(g[groupby.index(k)])
    #     #         for _,line in traj.iterrows():
    #     #             d["altitude"].append(line.altitude)
    #     #             d["track_angle"].append(line.track)
    # df=pd.DataFrame(d).sort_values(by=["start"]).reset_index(drop=True)
    # print(df)
    # df.to_parquet(args.detectedout)
    # # raise Exception
main()
