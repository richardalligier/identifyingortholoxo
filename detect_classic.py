# from detect_orthodromy import Detect, filter_trajectories, compute_angle
from geosphere import orthodromy, loxodromy, distance_without_time_exact,distance_loxo_ortho,  my_distance_ortho,my_distance_loxo#distance_ortho_pygplates,distance_loxo
import numpy as np
import pandas as pd
import time
from pyproj import CRS, Transformer
from traffic.core import Traffic, mixins
from filterclassic import FilterCstLatLon
import csaps
import piecewise
#from traffic.algorithms.prediction.flightplan import Point

DEBUG = False
SAVEFIG = False

def filter_trajectories(df, strategy):
    df = df.copy()
    # df = df.drop_duplicates(["timestamp"]).sort_values(["timestamp"]).reset_index(drop=True)#.head(10_000)
    #print(df.shape)
    df = df.drop_duplicates(["icao24","callsign","timestamp"]).sort_values(["icao24","callsign","timestamp"]).reset_index(drop=True)#.head(10_000)
    #print(df.shape)
    if strategy == "classic":
        filter = FilterCstLatLon()#|FilterCstPosition()#|FilterCstSpeed()#|MyFilterDerivative()#|FilterIsolated()
    else:
        raise Exception(f"strategy '{strategy}' not implemented")
    nointerpolate = lambda x:x
    return Traffic(df).filter(filter=filter,strategy=nointerpolate).eval(max_workers=1).data.copy()

def get_navaid(flightplan,name):
    for x in flightplan:
        if x.name==name:
            return x
    raise Exception

def intersection(a,b):
    (la,ua) = a
    (lb,ub) = b
    return (max(la,lb), min(ua,ub))

def union(a,b):
    (la,ua) = a
    (lb,ub) = b
    return (min(la,lb), max(ua,ub))

def iou(a,b):
    i = intersection(a,b)
    u = union(a,b)
    res= (i[1]-i[0])/(u[1]-u[0])
    return res

def reducetrack(ij,tin,trackin,params):#thresh_border,thresh_slope):
    from sklearn import linear_model
    # return [(0,len(track)-1)]
    track = np.unwrap(trackin,period=360)
    i,j = ij
    n = j-i+1
    t = tin[i:j+1]-tin[i]
    track = trackin[i:j+1]
    if n<=3:
        m = linear_model.LinearRegression()
    else:
        if params["model"]=="quantile":
            m = linear_model.QuantileRegressor(alpha=1e-6)
        elif params["model"]=="mean":
            m = linear_model.LinearRegression()
        else:
            raise Exception
    # m = linear_model.LinearRegression()
    m.fit(t[:,None],track)
    strack = m.predict(t[:,None])
    slope = abs(m.coef_[0])
    if params["thresh_slope"] is not None and slope > params["thresh_slope"]:
        return None,None
    isok = np.abs(track-strack)
    for k,ck in enumerate(isok):
        if ck < params["thresh_border"]:
            break
    for l in range(len(isok)-1,-1,-1):
        if isok[l] < params["thresh_border"]:
            break
    if k<l:
        # print(isok<thresh_border)
        # print(k,len(isok)-1-l)
        if k==0 and l==len(isok)-1:
            return slope,(i+k,i+l)
        else:
            return reducetrack((i+k,i+l),tin,trackin,params)#slope,(i+k,i+l)
    else:
        return None,None

class ResLine:
    def __init__(self,iseg,debugdata,i,j,dolmax,do,dl,angle):#,lats,lons):#,dolmax):
        #print(i,j,dolmax,do.max(),dl.max())
        self.interval=(i,j)
        # self.do=do
        # self.dl=dl
        self.debugdata=debugdata
        self.iseg = iseg
        self.idlmax = np.argmax(dl)
        self.idomax = np.argmax(do)
        self.dlmax = dl[self.idlmax]
        self.domax = do[self.idomax]
        self.dolmax = dolmax
        self.dlmean = dl.mean()
        self.domean = do.mean()
        self.v = np.mean(angle)
        assert(j-i+1==angle.shape[0])
        self.lever = piecewise.compute_lever(angle)
        # if self.domax == 0.:
        #     raise Exception
        self.dmax = min(self.domax,self.dlmax)

        self.maxangle = angle.max()
        self.minangle = angle.min()
        self.stdangle = angle.std()
        self.meanabsangleerror = np.abs(angle-self.v).mean()
    def length(self):
        i,j = self.interval
        return j-i+1
    # def reindex(self,indexes):
    #     i,j=self.interval
    #     return ResLine(i=indexes[i],j=indexes[j],do=self.do,dl=self.dl,dolmax=self.dolmax)
    def __repr__(self):
        return f"{self.interval}{Line.LOXO if self.dlmax<self.domax else Line.ORTH} {self.domax} {self.dlmax}"


def build(lats,lons,iseg,cstep,i,j,a):
    dl = my_distance_loxo(lats[i],lons[i],lats[j],lons[j],lats[i:j+1],lons[i:j+1])
    assert(dl.shape[0]==j-i+1)
    do = my_distance_ortho(lats[i],lons[i],lats[j],lons[j],lats[i:j+1],lons[i:j+1])
    if lats[i]==lats[j] and lons[i]==lons[j]:
        dolmax = 0.
    else:
        dolmax = distance_loxo_ortho(lats[i],lons[i],lats[j],lons[j])
    return ResLine(iseg,cstep,i,j,dolmax,do,dl,a)#,dol.max())




class Segment:
    def __init__(self,iseg,debugdata,i,j):
        self.iseg = iseg
        self.debugdata = debugdata
        self.interval = (i,j)
    def totuple(self):
        (i,j) = self.interval
        return (self.iseg,self.debugdata,i,j)
    def is_included(self,other):
        (i,j) = self.interval
        (k,l) = other.interval
        return k<=i and j<=l
    def __repr__(self):
        return f"Segement({self.interval})"

def check_iou(seg,g,thresh_iou):
    for x in g:
        if iou(x.interval,seg.interval) < thresh_iou:
            return False
    return True


def groupbyCriteriaThenSelect(l,criteria,select):
    res = []
    for seg in l:
        added = False
        for g in res:
            # print(criteria(seg,g))
            if criteria(seg,g):#check_iou(seg,g,thresh_iou):
                g.append(seg)
                added = True
        # assert(not False)
        if not added:
            res.append([seg])
    # print(res)
    return [select(x) for x in res]


def sortCriteriaThenSelect(l,criteria,filterother):
    l = sorted(l,key=criteria)
    res = []
    while l!=[]:
        x = l.pop()
        res.append(x)
        l = [y for y in l if filterother(x,y)]
    return res


def sortbysizeandfilterbyintersection(l,thresh_iou):
    def filterother(x,y):
        return iou(x.interval,y.interval)<=thresh_iou
    def criteria(x):
        i,j=x.interval
        # a = angle[i:j+1]
        # v = np.mean(a)
        return (j-i)#,-np.mean(np.abs(v-a))
    return sortCriteriaThenSelect(l,criteria,filterother)

def groupbyintersection(l,thresh_iou):
    criteria = lambda seg,g: check_iou(seg,g,thresh_iou)
    select = lambda x: max(x,key=lambda e:e.debugdata["distance"])
    return groupbyCriteriaThenSelect(l,criteria,select)


def remove_included(segs):
    res = set()
    for s in segs:
        toremove=set()
        isincluded=False
        for resi in res:
            if s.is_included(resi):
                toremove=set()
                isincluded=True
                break
            if resi.is_included(s):
                toremove.add(resi)
        if not isincluded:
            res.add(s)
        # print(s,toremove)
        res.difference_update(toremove)
    # print(res)
    return list(res)


def detect(tf,flightplan,params):
    global old
    l=list(tf)
    assert(len(l)==1)
    # print(flightplan)
    # print(l[0].closest_point(flightplan))
    # raise Exception
    aligned = l[0].aligned_on_navpoint(
        flightplan,
        angle_precision=params["angle_precision"],
        min_distance=params["min_distance"],
        time_precision=params["time_precision"],
    )
    # print(tf.data)
    # print(list(tf.data))
    res = []
    for iseg,f in enumerate(aligned):
        debugdata = {k: f.data[k].max() for k in  ['distance', 'bearing', 'shift', 'delta', ]}
        debugdata['navaid']=f.data['navaid'].iloc[0]
        p = get_navaid(flightplan,debugdata['navaid'])
        debugdata['navaid_latitude']=p.latitude
        debugdata['navaid_longitude']=p.longitude
        # print(f.data)
        res.append(Segment(iseg,debugdata,f.data.index[0],f.data.index[-1]))
    # res = groupbyintersection(res,params["thresh_iou"])
    newres = []
    t = tf.data["tunix"].values
    track = tf.data["track"].values
    for s in res:
        slope,r = reducetrack(s.interval,t,track,params)
        if r is not None:
            s.debugdata["slope"]=slope
            newres.append(Segment(s.iseg,s.debugdata,r[0],r[1]))
    res = newres
    res = remove_included(res)
    res = sortbysizeandfilterbyintersection(res,params["thresh_iou"])
    return [x.totuple() for x in res]


def compute_angle(t,xr,yr,params):
    n = xr.shape[0]
    xy = [xr,yr]
    sxy = csaps.csaps(t, xy, smooth=params["smooth"])
    pxy = sxy(t,nu=0)
    if DEBUG:
        print(((pxy[0]-xr)**2+(pxy[1]-yr)**2).mean())
        print(np.sqrt(((pxy[0]-xr)**2+(pxy[1]-yr)**2).max()))
    dx,dy = sxy(t,nu=1)
    # print(xr)
    # print(yr)
    # plt.scatter(xr,yr)
    # plt.scatter(*pxy,s=1)
    # plt.show()
    angle = np.degrees((np.arctan2(dy,dx)))
    return angle



# def extractor(flightplan,df,params):
#     tf = Traffic(df)
#     return detect(tf,flightplan,params)



# class Point(mixins.PointMixin):
#     def __init__(self,name,latitude,longitude):
#         self.longitude = longitude
#         self.latitude = latitude
#         self.name = name
#     def __repr__(self):
#         return f"({self.longitude}, {self.latitude})"
class NoFlightPlan(Exception): pass

old = None
def extract_flightplan(flightplans,df):
    global old
    icao24 = df.icao24.values[0]
    # print(type(icao24))
    # print(df.dtypes)
    # print(flightplans.dtypes)
    # print(icao24)
    # print(list(df))
    # print(list(df))
    # print(list(flightplans))
    # print(flightplans.icao24.unique())
    fp = flightplans.query("icao24==@icao24").query("@df.tunix.min()<=start").query("stop<=@df.tunix.max()")
    res = []
    if fp.shape[0]>0:
        for _,line in fp.iterrows():
            res.append([mixins.PointBase(name=name,latitude=x[1],longitude=x[0],altitude=float("nan")) for x,name in zip(line["flight_plan"],line["flight_plan_names"])])
    # print(f"{len(res)=} {df.icao24.iloc[0]=} {df.date.iloc[0]=} {df.callsign.iloc[0]=} {df.shape=}")
    # if old is not None:
    #     isnotok = df.icao24.iloc[0]==old[0] and df.date.iloc[0]==old[1] and df.callsign.iloc[0]==old[2]
    #     print(f"{df.icao24.iloc[0]==old[0]} {df.date.iloc[0]==old[1]} {df.callsign.iloc[0]==old[2]}")
    if len(res)==1:
        # print(df.shape,df.icao24.unique(),df.callsign.unique(),"ok")
        return res[0]
    # print(df.shape)
    # print(list(df.icao24.unique()))
    # print(list(df.callsign.unique()))
    # print(f"{len(res)=}")
    if len(res)>1:
        names = [p.name for p in res[0]]
        for x in range(1,len(res)):
            if [p.name for p in res[x]] != names:
                print(res)
                raise NoFlightPlan
        return res[0]
    raise NoFlightPlan


class Detect:
    _constantv = ["icao24","callsign","date"]
    _integersv = ["start","stop","npts"]
    _all = ["iswhat","dolmax","domax","dlmax","domean","dlmean","v","maxangle","minangle","stdangle","meanabsangleerror","lever"]+_integersv+_constantv+[f"{v}_{s}" for v in ["altitude","track"] for s in ["start","stop"]]
    def apply(self, df):
        return self.apply_splitted(df,lambda x: self._apply(x).astype({k:np.int64 for k in self._integersv}))
    def apply_splitted(self,df,f):
        isok = (df["timestamp"].diff().dt.total_seconds().to_numpy()>self.params["timesplit"])
        df["splitted"]= isok.cumsum()
        res=df.groupby(by="splitted").apply(f)
        return res
    def isvalid(self,r,df):#,latsin,lonsin,indexes):
        latsin = df.latitude.values
        lonsin = df.longitude.values
        i,j = r.interval
        res = i+1<j and latsin[i]!=latsin[j] and lonsin[i]!=lonsin[j]
        return res
    # def process(self,s,iswhat):
    #         for r in s:
    #             i,j = r.interval
    #             if self.isvalid(r,latsin,lonsin,indexes):#i+1<j and latsin[i]!=latsin[j] and lonsin[i]!=lonsin[j]:
    #                 assert(latsin[i]==lats[indexes[i]])
    #                 assert(lonsin[i]==lons[indexes[i]])
    #                 assert(latsin[j]==lats[indexes[j]])
    #                 assert(lonsin[j]==lons[indexes[j]])
    def _apply(self, df):
        # print("in Dectect._apply")
        df = df.reset_index(drop=True)
        lats = df.latitude.values
        lons = df.longitude.values
        track = df.track.values
        nonnan = np.logical_and(track==track,np.logical_and(lats==lats,lons==lons))#np.ones(lats.shape,dtype=bool)
        indexes = np.arange(lats.shape[0])[nonnan]
        dfnonan = df.loc[nonnan].reset_index(drop=True)
        d = {k:[] for k in self._all}
        n = dfnonan.shape[0]
        if n<=2:
            return pd.DataFrame(d)#.sort_values(by=["start"]).reset_index(drop=True)
        t0 = time.time()
        def process(s,iswhat):
            for r in s:
                i,j = r.interval
                if self.isvalid(r,dfnonan):#,latsin,lonsin,indexes):#i+1<j and latsin[i]!=latsin[j] and lonsin[i]!=lonsin[j]:
                    self.process_one(dfnonan,d,r,iswhat)
        for k,s in self.extract_segments(dfnonan).items():
            process(s,k)
        print(df.shape[0],time.time()-t0)
        for k in ["stop","start"]:
            d[k] = np.array(d[k])
        res = pd.DataFrame(d).sort_values(by=["start"]).reset_index(drop=True)
        for k,v in self.params.items():
            res[k]=v
        return res#.astype({k:np.int64 for k in self._integersv})
    def process_one(self,df,d,r,iswhat):
        df = df.reset_index(drop=True)
        i,j = r.interval
        nf = df.iloc[i:j+1]
        start = nf.timestamp.min()
        stop = nf.timestamp.max()
        d["start"].append(np.int64(start.timestamp()))
        d["stop"].append(np.int64(stop.timestamp()))
        d["iswhat"].append(iswhat)
        for q in ["start","stop"]:
            traj = nf.query(f"timestamp==@{q}")
            assert(traj.shape[0]==1)
            for _,line in traj.iterrows():
                d[f"altitude_{q}"].append(line.altitude)
                d[f"track_{q}"].append(line.track)
        for k,v in r.debugdata.items():
            if k in d:
                d[k].append(v)
            else:
                d[k]=[v]
        for k in self._constantv:
            d[k].append(line[k])
        d["dolmax"].append(r.dolmax)
        d["domax"].append(r.domax)
        d["dlmax"].append(r.dlmax)
        d["domean"].append(r.domean)
        d["dlmean"].append(r.dlmean)
        d["maxangle"].append(r.maxangle)
        d["minangle"].append(r.minangle)
        d["stdangle"].append(r.stdangle)
        d["v"].append(r.v)
        d["lever"].append(r.lever)#/self.params["track_tolerance_degrees"])
        d["meanabsangleerror"].append(r.meanabsangleerror)
        d["npts"].append(j-i+1)
                    # d["cstep"].append(r.cstep)
                    # d["iseg"].append(r.iseg)


class DetectOrthodromyWithBeacons(Detect):
    default = dict(
        name_is_orthodromy = "orthodromy",
        smooth = 1e-2,
        angle_precision = 2.,
        time_precision="20s",
        min_distance = 200.,
        model="quantile",
        timesplit=3600.,
        thresh_iou = 0.9,
        thresh_border = 0.1,
    )
    def __init__(self,extract_flightplan,flightplans, **kwargs):
        super().__init__()
        self.flightplans = flightplans
        self.params = {**self.default, **kwargs}
        self.params["thresh_slope"]=None
        self.extract_flightplan = extract_flightplan
    def extract_segments(self,df):
        flightplan = self.extract_flightplan(self.flightplans,df)
        if flightplan == []:
            return {}
        t = (df.timestamp.astype(int)//10**9).values
        t = t - t[0]
        track = df.track.values
        lats = df.latitude.values
        lons = df.longitude.values
        n =lats.shape[0]
        clat = lats[n//2]
        clon = lons[n//2]
        crs_dest = CRS.from_proj4(f"+proj=gnom +lat_0={clat} +lon_0={clon} +datum=WGS84 +units=m +no_defs")
        crs_geo = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs_geo, crs_dest, always_xy=True)
        x,y = transformer.transform(lons,lats)
        angle = np.unwrap(compute_angle(t,x,y,self.params),period=360)
        tf = Traffic(df)
        indexes = detect(tf,flightplan,self.params)#extractor(df) #douglas_peucker_xy(t,track,x,y,criterias,params)
        s = {self.params["name_is_orthodromy"]:[build(lats,lons,iseg,cstep,i,j,angle[i:j+1]) for iseg,cstep,i,j in indexes]}
        # print(s)
        return s

def add_parser(parser):
    parser.add_argument('-flightplans',type=str,required=True)
    for k,v in DetectOrthodromyWithBeacons.default.items():
        if isinstance(v,float):
            parser.add_argument(f'-{k}',type=float,default=v)
        elif isinstance(v,int):
            parser.add_argument(f'-{k}',type=int,default=v)
        else:
            parser.add_argument(f'-{k}',type=str,default=v)


def extract_args(args,extract_flightplan):
    d={k:v for k,v in vars(args).items()}
    kwargs={k:d[k] for k in DetectOrthodromyWithBeacons.default}
    kwargs["flightplans"] = pd.read_parquet(args.flightplans)
    kwargs["extract_flightplan"]=extract_flightplan
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
    add_parser(parser)
        # parser.add_argument('-angle_precision',type=float,default=1.)
        # parser.add_argument('-thresh_border',type=float,default=0.1)
        # parser.add_argument('-thresh_iou',type=float,default=0.1)
        # parser.add_argument('-model',default="quantile")
        # parser.add_argument('-min_distance',type=float)
        # parser.add_argument('-timesplit',type=float)
    parser.add_argument('-folderfigures',type=str)
    # parser.add_argument('-lever_crit',type=float,default=5)
    args = parser.parse_args()
    args = parser.parse_args()
    SAVEFIG=False
    if args.folderfigures is not None:
        SAVEFIG = False
        piecewise.SAVEFIG = True
        piecewise.DEBUG = True
        piecewise.FOLDER_FIGURES = args.folderfigures
    # icao24='a1f1a0';start=1659854804;stop=1659855207#duration=403track=100.0782;
#    icao24='3944ef';start=1657972484;stop=1657972484#duration=403track=100.0782;
    icao24='a1f1a0';start=1659854804;stop=1659855207
#    icao24='4caf7e';start=1658907489;stop=1658909950#1659509317#1659855207
    selecteddate=datetime.fromtimestamp(start).strftime("%Y-%m-%d")
    flights = filter_trajs.read_trajectories(f"{config.FOLDER}/trajs/{selecteddate}.parquet",queries=[f"icao24=={repr(icao24)}"])#.query("callsign=='TRA618T'")
    flightplans = filter_trajs.convert(pd.read_parquet(f"{config.FOLDER}/detectedref.parquet"))
    flightplans["icao24"]=flightplans["icao24"].astype("string[python]")
    print(flightplans.dtypes)
    print(flights.dtypes)
    # raise Exception
    # print(flightplans)
    # raise Exception
    # flights["tunix"] = flights["timestamp"].astype(int)//10**9#timestamp()
    start = start - 1200
    stop = stop + 1200
    flights = flights.query("@start<=tunix<=@stop")
    #.query("callsign=='RAM1668'").query("icao24=='02006f'")
    #flights = pd.read_parquet("/disk2/newjson/trajs/2022-08-10.parquet").query("icao24=='c0799a'").query("callsign=='TSC9434'")
    #flights = pd.read_parquet("airacsmall.parquet").query("callsign=='RAM1617'").query("icao24=='02006f'")#.query("icao24=='34364e'")
    #flights = flights.query("icao24 == @flights.icao24.unique()[2]")
    # flights.to_parquet("ech.parquet")
    # raise Exception
    flights["date"] = flights["timestamp"].dt.date
    print(flights["date"])
    # flights = flights.query("icao24=='4cc0df'").query("callsign=='BCS142'").query("date=='2022-07-14'").reset_index()
    # flights.to_parquet("pb.parquet")
    # flights = flights.iloc[0:40000]
    # flights = filter_trajectories(flights, "classic")#.query("icao24=='3415cf'")#.head(0000)#0a004b#0a0026#3415cf.query("icao24=='0a0026'")
    # flights = flights.dropna(subset=["track","latitude"])#.reset_index(drop=True)

    # print(flights["date"])
    # raise Exception
    #0a004b#0a0026
    #02a18e
    print(flights)
    #print(df.latitude.isna().mean())
    t0 = time.time()
    groupby = ["icao24","callsign","date"]
    print(flights.groupby(by=groupby).count())
    # kwargs = {k:v for k,v in vars(args).items()}
    # del kwargs["folderfigures"]
    kwargs=extract_args(args,extract_flightplan)#["flightplans"] = pd.read_parquet(args.flightplans)
    detector = DetectOrthodromyWithBeacons(**kwargs)
#    res = flights.groupby(by=groupby).apply(DetectOrthodromyWithBeacons(extract_flightplan=extract_flightplan,thresh_border=args.thresh_border,flightplans=flightplans,smooth=args.smooth,angle_precision=args.angle_precision,thresh=args.thresh,thresh_iou=args.thresh_iou,timesplit=args.timesplit,min_distance=args.min_distance).apply,include_groups=True).reset_index(drop=True)
    res = flights.groupby(by=groupby).apply(detector.apply,include_groups=True).reset_index(drop=True)
    print(list(res))
    res["dangle"] = res["maxangle"]-res["minangle"]

    print(time.time()-t0)
    print(res)
    header = ["iswhat","dolmax","domax","dlmax","lever","slope","icao24","callsign",'start',"stop","npts"]
    # res = res.query("dolmax==0.")
    for g,df in res.groupby(by=groupby):
        print(g)
        print(df.query("npts>10"))
        dfl=df.query("iswhat=='loxodromy'")#.query("dlmax<=0.5*dolmax").query("dlmax<=0.5*domax").query("npts>10")
        dfo=df.query("iswhat=='orthodromy'")#.query("domax<=0.5*dolmax").query("domax<=0.5*dlmax").query("npts>10")
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
        for (nf,what) in [(dfl,"is_loxodromy"),(dfo,"is_orthodromy")]:
            i=0
            for _,line in nf.iterrows():
                i+=1
                seg = traj.query("@line.start<=tunix").query("tunix<=@line.stop")
                go = plt.scatter(seg.longitude,seg.latitude)
                go.set_label(f"{what[3:]} #{i}")
        plt.xlabel("longitude [°]")
        plt.ylabel("latitude [°]")
        plt.gca().set_aspect("equal")
        plt.legend(frameon=False,handletextpad=0.2)
        if SAVEFIG:
            fig.set_tight_layout({'pad':0})
            fig.set_figwidth(4)
            plt.savefig("figures/latlon.pdf", dpi=300, bbox_inches='tight')
            plt.clf()
        else:
            plt.show()
        fig = plt.figure()
        go=plt.scatter(traj.timestamp,traj.track,c="black")
        go.set_label("ADS-B trajectory")

        for (nf,what) in [(dfl,"is_loxodromy"),(dfo,"is_orthodromy")]:
            i=0
            numero = []
            for _,line in nf.iterrows():
                i+=1
                seg = traj.query("@line.start<=tunix").query("tunix<=@line.stop")
                seg["timestamp"]=seg["timestamp"].astype("datetime64[ns, UTC]")
                # print(seg.dtypes)
                # print(seg.timestamp.dt)
                numero.append(i)
                go=plt.scatter(seg.timestamp,seg.track)
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
            plt.savefig("figures/timetrack.pdf", dpi=300, bbox_inches='tight')
            # plt.clf()
        else:
            plt.show()
        nf = pd.concat([dfl,dfo],ignore_index=True)
        nf["identified"] = nf["iswhat"]
        nf["ortho-loxo [m]"]=nf["dolmax"]
        nf["adsb-loxo [m]"]=nf["dlmax"]
        nf["ortho-adsb [m]"]=nf["domax"]
        nf["duration [s]"]=nf["stop"]-nf["start"]
        print(nf[["identified","segment number","duration [s]","ortho-loxo [m]","adsb-loxo [m]","ortho-adsb [m]"]].to_latex(index=False,float_format="%.2f"))
        nf[["identified","segment number","start","stop"]].to_csv("segments_classic.csv",index=False)
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
