import pyproj
from pyproj import Geod
import pandas as pd
import numpy as np

from traffic.core import Traffic
from filterclassic import FilterCstLatLon
#from geosphere import orthodromy, loxodromy, distance_without_time_exact,distance_loxo_ortho,  my_distance_ortho,my_distance_loxo#distance_ortho_pygplates,distance_loxo
import tqdm
import time
from scipy.sparse import SparseEfficiencyWarning
FutureWarning
from collections import deque
import matplotlib.pyplot as plt

from traffic import algorithms
from detect_classic import build, Segment, Detect, compute_angle,reducetrack
import detect_classic
#posl = loxodromy(lat1,lon1,lat2,lon2,npts)
#poso = orthodromy(lat1,lon1,lat2,lon2,npts)
#SparseEfficiencyWarning:
import warnings
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
from enum import Enum,auto
from pyproj import CRS, Transformer
import csaps
import piecewise

import sys
sys.setrecursionlimit(10000000)
print("After:", sys.getrecursionlimit())

DEBUG = False
SAVEFIG = False





# def enlarge(i,j,angle,params):
#     k=(i+j)//2
#     forward=np.maximum.accumulate(angle[k:])-np.minimum.accumulate(angle[k:])
#     backward=np.maximum.accumulate(angle[k::-1])-np.minimum.accumulate(angle[k::-1])
#     for r,dr in enumerate(forward):
#         if dr>params["track_tolerance_degrees"]/2:
#             break
#     newj = k+r
#     for r,dr in enumerate(backward):
#         if dr>params["track_tolerance_degrees"]/2:
#             break
#     newi = k-r
#     return (newi,newj)


# def apply_enlarge(lower,upper,angle,params):
#     l=set()
#     for (i,j) in zip(lower,upper):
#         newi,newj =enlarge(i,j,angle,params)
#         l.add((newi,newj))
#     l=sorted(l)
#     return tuple(map(np.array, zip(*l)))#,np.array(p)




def extract_longest(arr,hit_tolerance, max_width):
    n = len(arr)
    min_dq, max_dq = deque(), deque()
    l = 0
    r = 0
    while r < n:
        # remove all old max and add r as new max
        while max_dq and arr[max_dq[-1]] < arr[r]:
            max_dq.pop()
        max_dq.append(r)
        # remove all old min and add r as new min
        while min_dq and arr[min_dq[-1]] > arr[r]:
            min_dq.pop()
        min_dq.append(r)

        # increase left till tunnnel is small
        while arr[max_dq[0]] - arr[min_dq[0]] > max_width:
            l += 1
            if max_dq[0] < l:
                max_dq.popleft()
            if min_dq[0] < l:
                min_dq.popleft()

        # check if tunnel ends at r
        assert((arr[max_dq[0]] - arr[min_dq[0]] <= max_width))
        is_end = (
            r == n-1 or
            arr[max_dq[0]] - arr[min_dq[0]] > max_width or
            (arr[r+1:r+2+hit_tolerance] - arr[min_dq[0]] > max_width).all() or
            (arr[max_dq[0]] - arr[r+1:r+2+hit_tolerance] > max_width).all()
        )
        if is_end and r >= l:
            yield (l,r)
        r += 1

def keep(angle,ij,params):
    (i,j)=ij
    a = angle[i:j+1]
    v = np.mean(a)
    # lever = piecewise.compute_lever(a-v)/params["track_tolerance_degrees"]
    return np.mean(np.abs(a-v)) < params["thresh_abs_mean_ratio"]

def groupbyintersection(l,angle,thresh_iou):
    criteria = lambda seg,g: detect_classic.check_iou(seg,g,thresh_iou)
    def f(seg):
        (i,j) = seg.interval
        a = angle[i:j+1]
        v = np.mean(a)
        return np.mean(np.abs(a-v))
    select = lambda x: max(x,key=lambda e:(e.interval[1]-e.interval[0],-f(e)))
    return detect_classic.groupbyCriteriaThenSelect(l,criteria,select)


def sortbysizeandfilterbyintersection(l,angle,thresh_iou):
    def filterother(x,y):
        return detect_classic.iou(x.interval,y.interval)<=thresh_iou
    def criteria(x):
        i,j=x.interval
        a = angle[i:j+1]
        v = np.mean(a)
        return (j-i) ,-np.mean(np.abs(v-a))
    return detect_classic.sortCriteriaThenSelect(l,criteria,filterother)


def filterbylever(seg,angle,params):
    i,j = seg.interval
    a = angle[i:j+1]
    return piecewise.compute_lever(a)<params["thresh_lever"]

def douglas_peucker_xy(t,track,xr,yr,params,what):
    n = xr.shape[0]

    angle = np.unwrap(compute_angle(t,xr,yr,params),period=360)#np.degrees(np.unwrap(np.arctan2(dy,dx)))
    assert(xr.shape[0]==angle.shape[0])
    lowerupper = list(extract_longest(angle,params["hit_tolerance"], params["track_tolerance_degrees"]))# if keep(angle,ij,params)]
    # print(f"{len(lowerupper)=}")
    lowerupper = [Segment(iseg,dict(),i,j) for iseg,(i,j) in enumerate(lowerupper)]
    if piecewise.DEBUG:
        itoplot = min(np.arange(len(lowerupper)),key=lambda k:lowerupper[k].interval[0])
        (lower,upper) = tuple(map(np.array, zip(*[s.interval for s in lowerupper])))
        lowi = np.array([0])
        uppi = np.array([upper[itoplot]-lower[itoplot]])
        piecewise.plotdebug(angle[lower[itoplot]:upper[itoplot]+1],(lowi,uppi),"projIsole",what)
        del lower
        del upper
    newres = []
    for s in lowerupper:
        slope,r = reducetrack(s.interval,t,angle,params)
        if r is not None:
            s.debugdata["slope"]=slope
            newres.append(Segment(s.iseg,s.debugdata,r[0],r[1]))
    lowerupper = newres
    if piecewise.DEBUG:
        itoplot = min(np.arange(len(lowerupper)),key=lambda k:lowerupper[k].interval[0])
        (lower,upper) = tuple(map(np.array, zip(*[s.interval for s in lowerupper])))
        lowi = np.array([0])
        uppi = np.array([upper[itoplot]-lower[itoplot]])
        piecewise.plotdebug(angle[lower[itoplot]:upper[itoplot]+1],(lowi,uppi),"projIsoleAfterCut",what)
        del lower
        del upper

    # lowerupper = [Segment(iseg,dict(),i,j) for iseg,(i,j) in enumerate(lowerupper)]
    # lowerupper = [Segment(iseg,dict(),i,j) for iseg,(i,j) in enumerate(lowerupper)]
    # print(f"before {len(lowerupper)=}")
    # lowerupper = groupbyintersection(lowerupper,angle,params["thresh_iou"])
    # lowerupper = [seg for seg in lowerupper if filterbylever(seg,angle,params)]
    lowerupper = sortbysizeandfilterbyintersection(lowerupper,angle,params["thresh_iou"])
    # print(f"after  {len(lowerupper)=}")
    if lowerupper == []:
        return []
    else:
        (lower,upper) = tuple(map(np.array, zip(*[s.interval for s in lowerupper])))
    # (lower,upper) = apply_enlarge(lower,upper,angle,params)
    if piecewise.DEBUG:
        piecewise.plotdebug(angle,(lower,upper),"proj",what)
    assert(upper.max()<n)
    # print(thresholds)
    return [s.totuple() for s in lowerupper]
    # indexes = set()
    # for iseg,(i,j) in enumerate(zip(lower,upper)):
    #     if i+1<j:
    #         for k,l in reducetrack(t[i:j+1],track[i:j+1],thresh=params["thresh"]):
    #             indexes.add((i+k,i+l))
    # return [(iseg,dict(),i,j,angle[i:j+1]) for iseg,(i,j) in enumerate(indexes)]



# def douglas_peucker_proj(crs_dest,t,track,lats,lons,criterias,params):
#     n =lats.shape[0]
#     clat = lats[n//2]
#     clon = lons[n//2]
#     crs_geo = CRS.from_epsg(4326)
#     transformer = Transformer.from_crs(crs_geo, crs_dest, always_xy=True)
#     x,y = transformer.transform(lons,lats)
#     indexes = douglas_peucker_xy(t,track,x,y,criterias,params)
#     s = [build(lats,lons,iseg,cstep,i,j,a) for iseg,cstep,i,j,a in indexes]
#     return s

class DetectLongestOrthodromyLoxodromy(Detect):
    _old=[]
    default = dict(
        track_tolerance_degrees = 0.5,
        name_is_orthodromy = "orthodromy",
        name_is_loxodromy = "loxodromy",
        smooth=1e-2,
        thresh_iou=0.1,
        model="mean",
        hit_tolerance=0,
        thresh_slope = 0.001,
        thresh_border = 0.1,
        timesplit=3600.,
    )
    def __init__(self, **kwargs):
        super().__init__()
        self.params = {**self.default, **kwargs}
        assert(self.params["model"] is not None)
    def extract_segments_xy(self,crs_dest,df,what):
        lats = df.latitude.values
        lons = df.longitude.values
        t = (df.timestamp.astype(int)//10**9).values
        t = t - t[0]
        track = df.track.values
        crs_geo = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs_geo, crs_dest, always_xy=True)
        x,y = transformer.transform(lons,lats)
        angle = np.unwrap(compute_angle(t,x,y,self.params),period=360)
        segments = douglas_peucker_xy(t,track,x,y,self.params,what)
        s = [build(lats,lons,iseg,cstep,i,j,angle[i:j+1]) for iseg,cstep,i,j in segments]
        return s

    def extract_segments(self,df):
        # print(df.head(3))
        lats = df.latitude.values
        lons = df.longitude.values
        n =lats.shape[0]
        clat = lats[n//2]
        clon = lons[n//2]
        dcrs = {
            self.params["name_is_orthodromy"]:CRS.from_proj4(f"+proj=gnom +lat_0={clat} +lon_0={clon} +datum=WGS84 +units=m +no_defs"),
            self.params["name_is_loxodromy"]:CRS.from_proj4("+proj=merc +datum=WGS84 +units=m +no_defs"),
        }
        ds ={}
        for k,crs in dcrs.items():
            ds[k]=self.extract_segments_xy(crs,df,k)#,self.criterias,self.params)
        return ds


def add_parser(parser):
    for k,v in DetectLongestOrthodromyLoxodromy.default.items():
        if isinstance(v,float):
            parser.add_argument(f'-{k}',type=float,default=v)
        elif isinstance(v,int):
            parser.add_argument(f'-{k}',type=int,default=v)
        else:
            parser.add_argument(f'-{k}',type=str,default=v)

def extract_args(args):
    d=vars(args)
    kwargs={k:d[k] for k in DetectLongestOrthodromyLoxodromy.default}
    return kwargs

def mainold():

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from traffic.data import opensky, navaids
    import time
    from datetime import datetime
    from figures import read_config
    import filter_trajs
    # print(list(navaids))
    # raise Exception
    import pandas as pd
    import argparse
    config = read_config()
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    SAVEFIG=False
    add_parser(parser)
    parser.add_argument('-r',type=float,default=0.5)
    parser.add_argument('-dolmax',type=float,default=30)
    # parser.add_argument('-smooth',type=float,default=0.01)
    # parser.add_argument('-track_tolerance_degrees',type=float,default=200000)
    # parser.add_argument('-thresh_border',type=float,default=0.1)
    # # parser.add_argument('-thresh_abs_mean_ratio',type=float,default=0.25)
    # parser.add_argument('-model',default="quantile")
    # parser.add_argument('-thresh_iou',type=float,default=0.1)
    # parser.add_argument('-thresh_slope',type=float,default=1)
    # parser.add_argument('-hit_tolerance',type=int,default=10)
    # parser.add_argument('-timesplit',type=float)
    parser.add_argument('-folderfigures',type=str)
    parser.add_argument('-trajfile',type=str)
    args = parser.parse_args()
    if args.folderfigures is not None:
        SAVEFIG = True
        piecewise.SAVEFIG = True
        piecewise.DEBUG = True
        piecewise.FOLDER_FIGURES = args.folderfigures
    #FILE = "source/raw_shortcut_flight.csv"
    #FILE = "source/raw_neighbour_flight.csv"
    #flights = read_format(FILE)
    #flights = opensky.history("2022-07-15 00:44:24",stop="2022-07-15 23:44:24",departure_airport="LFBO").data#,callsign="FOX50A")#.data#.iloc[:10000]
#    flights = opensky.history("2022-07-15 00:44:24",stop="2022-07-15 23:44:24",callsign="EDC749").data#.iloc[:10000]
    #flights = opensky.history("2022-07-15 00:44:24",stop="2022-07-15 23:44:24",callsign="EZY18AN").data#.iloc[:10000]
    #flights = opensky.history("2024-06-09 09:44:24",stop="2024-06-09 15:44:24",departure_airport="LFBO",arrival_airport="LIPQ").data#.iloc[:10000]
#    flights = opensky.history("2024-05-09 03:44:24",stop="2024-05-09 12:44:24",callsign="RYR1338").data#.iloc[:10000]
    #flights = opensky.history("2024-06-09 09:44:24",stop="2024-06-09 15:44:24",callsign="RYR3630").data#.iloc[:10000]
    #AFR87GJ
    #flights = pd.read_parquet("airac.parquet")#.query("icao24=='3415cf'")#.head(0000)#0a004b#0a0026#3415cf.query("icao24=='0a0026'")
    # flights = pd.read_parquet("pb.parquet")
    #flights.to_parquet("pb.parquet")
    # flights = pd.read_parquet("airac.parquet").iloc[:50000]
    #flights = pd.read_parquet(f"{config.FOLDER}/trajs/2022-07-14.parquet").query("icao24=='34364e'").query("callsign=='ANE46SQ'")
    # icao24='4cad0f';start=1658052173;stop=1658052992;selecteddate='2022-07-17' # loxo-ortho ok ! 24414
    # icao24='3964e4';start=1658222384;stop=1658222580#; 24413 loxo-ortho
    # icao24='40751c';start=1658661774;stop=1658662581;#24419 ok, mais bof qd meme
    # icao24='4841d8';start=1658053448;stop=1658053797; 24426 nope !
    # icao24='4ca242';start=1658776939;stop=1658777238; 24433 presque mais tres nordsur
    # icao24='406c43';start=1658998169;stop=1658998835#; 24443 1er ortho
    # icao24='406b6d';start=1658735962;stop=1658736457#;
    # icao24='40655b';start=1658994771;stop=1658995575#1658736457#;24459
    # icao24='4ca8fb';start=1659792582;stop=1659792924#1658487865
    # icao24='aa5f41';start=1658757581;stop=1658758152# bof 270 400
    # icao24='4bb186';start=1659093475;stop=1659093962# ok mais bof 90 400
    # icao24='4bb186';start=1659093475;stop=1659093962# ok mais bof 90 400
    # icao24='aa9093';start=1658487531;stop=1658487865#duration=334track=90.1085; bof
#    icao24='4ca8fb';start=1659792582;stop=1659792924#duration=342track=270.1483;
#    icao24='aa9801';start=1658915682;stop=1658916359#duration=677track=279.9327;
    # icao24='7380c2';start=1660055547;stop=1660055817#duration=270track=259.5085;
#    icao24='a1f1a0';start=1658208924;stop=1658209371#duration=447track=85.751;
    #icao24='a1f1a0';start=1659854804;stop=1659855207#duration=403track=100.0782;
    #icao24='4aca83';start=1658473349;stop=1658473552#duration=403track=100.0782;
    #icao24='48597d';start=1658502732;stop=1658503007#duration=403track=100.0782;
    # icao24='4d2311';start=1659523048;stop=1659523175#duration=270track=259.5085;
    # icao24='4ac9eb';start=1659950920;stop=1659953185#duration=270track=259.5085;
    # icao24='34364e';start=1657776663;stop=1657776663
    # icao24='4ac9eb';start=1659950970;stop=1659953145 # needs 0.5
    # icao24='a1f1a0';start=1659854804;stop=1659855207#duration=403track=100.0782;
    icao24='4bb0eb';start=1659357150;stop=1659357150#duration=403track=100.0782;
    icao24='3944ef';start=1657972484;stop=1657972484#duration=403track=100.0782;
    icao24='a1f1a0';start=1659854804;stop=1659855207
    icao24='43edf0';start=1659424603;stop=1659425670
    icao24='501db3';start=1659692865;stop=1659425670
    icao24='aaed0d';start=1659701797;stop=1659425670
    icao24='43edf0';start=1659424603;stop=1659425670
    icao24='a1f1a0';start=1659854804;stop=1659855207
    # icao24='40660c';start=1659507285;stop=1659509317#1659855207
    # icao24='4caf7e';start=1658907489;stop=1658909950#1659509317#1659855207
    if args.trajfile is None:
        selecteddate=datetime.fromtimestamp(start).strftime("%Y-%m-%d")
        flights = filter_trajs.read_trajectories(f"{config.FOLDER}/trajs/{selecteddate}.parquet",queries=[f"icao24=={repr(icao24)}"])#,"callsign=='AAL111'","callsign=='2NAOM'"
        # flights = filter_trajs.read_trajectories(f"{config.FOLDER}/savan.parquet").query("callsign=='SAVAN07'")
        print(flights.icao24.unique())
        # flights["tunix"] = flights["timestamp"].astype(int)//10**9#timestamp()
        start = start - 1200
        stop = stop + 1200
        flights = flights.query("@start<=tunix<=@stop")
    else:
        flights = filter_trajs.read_trajectories(args.trajfile)
    #.query("callsign=='RAM1668'").query("icao24=='02006f'")
    #flights = pd.read_parquet("/disk2/newjson/trajs/2022-08-10.parquet").query("icao24=='c0799a'").query("callsign=='TSC9434'")
    #flights = pd.read_parquet("airacsmall.parquet").query("callsign=='RAM1617'").query("icao24=='02006f'")#.query("icao24=='34364e'")
    #flights = flights.query("icao24 == @flights.icao24.unique()[2]")
    # flights.to_parquet("ech.parquet")
    # raise Exception
    # flights["date"] = flights["timestamp"].dt.date
    print(flights["date"])
    # flights = flights.query("icao24=='4cc0df'").query("callsign=='BCS142'").query("date=='2022-07-14'").reset_index()
    # flights.to_parquet("pb.parquet")
    # flights = flights.iloc[0:40000]
    # flights = filter_trajectories(flights, "classic")#.query("icao24=='3415cf'")#.head(0000)#0a004b#0a0026#3415cf.query("icao24=='0a0026'")
    # raise Exception
    #0a004b#0a0026
    #02a18e
    print(flights)
    #print(df.latitude.isna().mean())
    t0 = time.time()
    groupby = ["icao24","callsign","date"]
    print(flights.groupby(by=groupby).count())
    kwargs = extract_args(args)
    res = flights.groupby(by=groupby).apply(DetectLongestOrthodromyLoxodromy(**kwargs).apply,include_groups=True).reset_index(drop=True)
    print(list(res))
    res["dangle"] = res["maxangle"]-res["minangle"]
    #res = flights.filter(DetectOrthodromyLoxodromy()).data

    print(time.time()-t0)
    print(res)
    # res = res.query("dolmax==0.")
    for g,df in res.groupby(by=groupby):
        print(g)
        # print(df.query("npts>10"))
        dfl=df.query("iswhat=='loxodromy'")#.query("dlmax<=@r*dolmax").query("dlmax<=@r*domax").query("npts>10")
        dfo=df.query("iswhat=='orthodromy'")#.query("domax<=@r*dolmax").query("domax<=@r*dlmax").query("npts>10")
        header = ["iswhat","dolmax","domax","dlmax","lever","slope","icao24","callsign",'start',"stop","npts"]
        print(dfo[header])
        print(dfl[header])
        # if what == "is_loxodromy":
        # else:

        print(df.dtypes)

        #what="is_orthodromy"
        traj=flights
        fig = plt.figure()
        go = plt.scatter(traj.longitude,traj.latitude,c="black")
        go.set_label("ADS-B trajectory")
        toiter = [(dfl,"is_loxodromy","+",20),(dfo,"is_orthodromy","x",1)]
        for (nf,what,marker,s) in toiter:
            i=0
            for _,line in nf.iterrows():
                i+=1
                seg = traj.query("@line.start<=tunix").query("tunix<=@line.stop")
                go = plt.scatter(seg.longitude,seg.latitude,marker=marker,s=s)
                go.set_label(f"{what[3:]} #{i}")
        plt.xlabel("longitude [°]")
        plt.ylabel("latitude [°]")
        plt.gca().set_aspect("equal")
        plt.legend(frameon=False,handletextpad=0.2)
        if SAVEFIG:
            fig.set_tight_layout({'pad':0})
            fig.set_figwidth(4)
            plt.savefig(f"{args.folderfigures}/latlon.pdf", dpi=300, bbox_inches='tight')
            plt.clf()
        else:
            plt.show()
        fig = plt.figure()
        go=plt.scatter(traj.timestamp,traj.track,c="black")
        go.set_label("ADS-B trajectory")

        for (nf,what,marker,s) in toiter:
            i=0
            numero = []
            for _,line in nf.iterrows():
                i+=1
                seg = traj.query("@line.start<=tunix").query("tunix<=@line.stop")
                seg["timestamp"]=seg["timestamp"].astype("datetime64[ns, UTC]")
                # print(seg.dtypes)
                # print(seg.timestamp.dt)
                numero.append(i)
                go=plt.scatter(seg.timestamp,seg.track,marker=marker,s=s)
                go.set_label(f"{what[3:]} #{i}")
            nf["segment number"] = numero
            plt.xlabel("time")
            plt.ylabel("ADS-B track angle [°]")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.xticks(rotation=45)
        plt.legend(frameon=False,handletextpad=0.2)
        if SAVEFIG:
            fig.set_tight_layout({'pad':0})
            fig.set_figwidth(4)
            plt.savefig(f"{args.folderfigures}/timetrack.pdf", dpi=300, bbox_inches='tight')
            # plt.clf()
        else:
            plt.show()
        dfl["me"] = dfl["dlmax"]
        dfo["me"] = dfo["domax"]
        dfl["other"] = dfl["domax"]
        dfo["other"] = dfo["dlmax"]
        nf = pd.concat([dfl,dfo],ignore_index=True)
        nf["maxloxo"]=dfl["segment number"].max()
        nf["maxortho"]=dfo["segment number"].max()
        nf = nf.query("dolmax>@args.dolmax").query("me<dolmax*@args.r").query("me<other*@args.r")
        nf["identified"] = nf["iswhat"]
        nf["ortho-loxo [m]"]=nf["dolmax"]
        nf["adsb-loxo [m]"]=nf["dlmax"]
        nf["ortho-adsb [m]"]=nf["domax"]
        nf["duration [s]"]=nf["stop"]-nf["start"]
        if args.folderfigures is not None:
            with open(f"{args.folderfigures}/tablelongest.tex",'w') as f:
                f.write(nf[["identified","segment number","duration [s]","ortho-loxo [m]","adsb-loxo [m]","ortho-adsb [m]"]].to_latex(index=False,float_format="%.2f"))
        nf[["identified","maxloxo","maxortho","segment number","start","stop"]].to_csv("segments_longest.csv",index=False)
    dlatlon = 3
    minlat = flights.latitude.min() - dlatlon
    maxlat = flights.latitude.max() + dlatlon
    minlon = flights.longitude.min() - dlatlon
    maxlon = flights.longitude.max() + dlatlon
    def isok(nav):
        return nav.type == "DME" and minlon<=nav.longitude <=maxlon and minlat<=nav.latitude <=maxlat
    # lon = [nav.longitude for nav in navaids if isok(nav)]
    # lat = [nav.latitude for nav in navaids if isok(nav)]
    # plt.scatter(lon,lat,c="red")
    # plt.show()
#         if nsegs>0:
#             plt.scatter(df.longitude,df.latitude)
#             for i in range(1,nsegs+1):
#                 dfi = df.query(f"{what}==@i")
#                 print(f"{dfi.shape=}")
#                 assert(dfi.latitude.values[0]==dfi.latitude.values[0])
#                 assert(dfi.latitude.values[-1]==dfi.latitude.values[-1])
#                 o = orthodromy(dfi.latitude.values[0],dfi.longitude.values[0],dfi.latitude.values[-1],dfi.longitude.values[-1],dfi.shape[0])
#                 print(f"{o.shape=}")
# #                plt.scatter(dfi.longitude,dfi.latitude)
#                 plt.scatter(o[:,1],o[:,0])
#             plt.gca().axis('equal')
#             plt.show()
#             proj = pyproj.Proj(proj="merc",ellps='sphere')
#             # lon=0
#             # for lat in range(0,40):
#             #     print(f"{proj.transform(lon,lat)[1]=} {mercator(lat,lon)[0]=} {proj.transform(lon,lat)[1]/mercator(lat,lon)[0]=}")
#             # lat = 0
#             # for lon in range(0,40):
#             #     print(f"{proj.transform(lon,lat)[0]=} {mercator(lat,lon)[1]=} {proj.transform(lon,lat)[0]/mercator(lat,lon)[1]=}")
#             # raise Exception
#             plt.plot(df.timestamp.values,df.track.values)
#             for i in range(1,nsegs+1):
#                 dfi = df.query(f"{what}==@i")
#                 if what == "is_orthodromy":
#                     o = orthodromy(dfi.latitude.values[0],dfi.longitude.values[0],dfi.latitude.values[-1],dfi.longitude.values[-1],dfi.shape[0])
#                     otrack, _, _ = geod.inv(o[:-1,1],o[:-1,0],o[1:,1],o[1:,0])
#                 else:
#                     proj = pyproj.Proj(proj="merc",ellps='sphere')
#                     y1,x1=np.array(proj.transform(dfi.latitude.values[0],dfi.longitude.values[0]))
#                     y2,x2=np.array(proj.transform(dfi.latitude.values[-1],dfi.longitude.values[-1]))
#                     dx = x2-x1
#                     dy = y2-y1
#                     #https://en.wikipedia.org/wiki/Rhumb_line
#                     m = dy/dx
#                     lambda0 = dfi.latitude.values[0]-y1/m
#                     lambda02 = dfi.latitude.values[-1]-y2/m
#                     print(f"{lambda02=} {lambda0=}")
#                     m = np.arctan2(dx,dy)
#                     print(f"{(dfi.latitude.values[0],dfi.longitude.values[0])=} {x1=} {y1=} {(dfi.latitude.values[-1],dfi.longitude.values[-1])=} {x2=} {y2=} {dx=} {dy=}")
#                     y1,x1=mercator(dfi.latitude.values[0],dfi.longitude.values[0])
#                     y2,x2=mercator(dfi.latitude.values[-1],dfi.longitude.values[-1])
#                     dx = x2-x1
#                     dy = y2-y1
#                     m = np.arctan2(dx,dy)
#                     print(f"{(dfi.latitude.values[0],dfi.longitude.values[0])=} {x1=} {y1=} {(dfi.latitude.values[-1],dfi.longitude.values[-1])=} {x2=} {y2=} {dx=} {dy=}")
#                     otrack = np.full(dfi.shape[0]-1,np.degrees(m))
#                 print(dfi.timestamp.values[0],dfi.timestamp.values[-1])
#                 # plt.plot(dfi.timestamp.values[:-1],zeroto360(otrack))
#                 plt.plot(dfi.timestamp.values[:-1],zeroto360(otrack))
#             plt.show()
#         else:
#             print("no loxo")
    #plt.plot(d2*100000)
#import matplotlib
#matplotlib.use('nbagg')
#%pylab inline
#mpld3.enable_notebook()
#print(df.shape[0],(df.timestamp.max()-df.timestamp.min()).total_seconds())
#%matplotlib widget
if __name__ == '__main__':
    mainold()
#[x for x in s if min(x.dl.max(),x.do.max())<25000 and x.interval[1]-x.interval[0]>100]
#for line in s:
    #plt.scatter()

#print(distance(posl[:,0],posl[:,1],poso[:,0],poso[:,1]))


#p1,p2
#proj.transform(p1[0],p1[1],direction=pyproj.enums.TransformDirection.INVERSE),(lat1,lon1)
