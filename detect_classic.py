from geosphere import distance_loxo_ortho,  my_distance_ortho,my_distance_loxo
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from traffic.core import Traffic, mixins
from filterclassic import FilterCstLatLon
import csaps
import piecewise


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
    if params["thresh_slope"] is None and params["thresh_border"] is None:
        return slope,(i,j)
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
    l=list(tf)
    assert(len(l)==1)
    aligned = l[0].aligned_on_navpoint(
        flightplan,
        angle_precision=params["angle_precision"],
        min_distance=params["min_distance"],
        time_precision=params["time_precision"],
    )
    res = []
    for iseg,f in enumerate(aligned):
        debugdata = {k: f.data[k].max() for k in  ['distance', 'bearing', 'shift', 'delta', ]}
        debugdata['navaid']=f.data['navaid'].iloc[0]
        p = get_navaid(flightplan,debugdata['navaid'])
        debugdata['navaid_latitude']=p.latitude
        debugdata['navaid_longitude']=p.longitude
        res.append(Segment(iseg,debugdata,f.data.index[0],f.data.index[-1]))
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
    xy = [xr,yr]
    sxy = csaps.csaps(t, xy, smooth=params["smooth"])
    pxy = sxy(t,nu=0)
#    if DEBUG:
#        print(((pxy[0]-xr)**2+(pxy[1]-yr)**2).mean())
#        print(np.sqrt(((pxy[0]-xr)**2+(pxy[1]-yr)**2).max()))
    dx,dy = sxy(t,nu=1)
    angle = np.degrees((np.arctan2(dy,dx)))
    return angle


class NoFlightPlan(Exception): pass

class FlightPlanDatabase:
    def __init__(self,filename):
        self.flightplans = pd.read_parquet(filename)
    def extract_flightplan(self,df):
        icao24 = df.icao24.values[0]
        fp = self.flightplans.query("icao24==@icao24").query("@df.tunix.min()<=start").query("stop<=@df.tunix.max()")
        res = []
        if fp.shape[0]>0:
            for _,line in fp.iterrows():
                res.append([mixins.PointBase(name=name,latitude=x[1],longitude=x[0],altitude=float("nan")) for x,name in zip(line["flight_plan"],line["flight_plan_names"])])
        if len(res)==1:
            return res[0]
        if len(res)>1:
            names = [p.name for p in res[0]]
            for x in range(1,len(res)):
                if [p.name for p in res[x]] != names:
                    print(res)
                    raise NoFlightPlan
            return res[0]
        raise NoFlightPlan


class Detect:
    _constantv = ("icao24","callsign")
    _integersv = ("start","stop","npts")
    _all = ("iswhat","dolmax","domax","dlmax","domean","dlmean","v","maxangle","minangle","stdangle","meanabsangleerror","lever")+_integersv+_constantv+tuple(f"{v}_{s}" for v in ["altitude","track"] for s in ["start","stop"])
    name_is_orthodromy = "orthodromy"
    name_is_loxodromy = "loxodromy"
    @classmethod
    def add_parser(cls,parser):
        for k,v in cls.default.items():
            if isinstance(v,float):
                parser.add_argument(f'-{k}',type=float,default=v)
            elif isinstance(v,int):
                parser.add_argument(f'-{k}',type=int,default=v)
            else:
                parser.add_argument(f'-{k}',type=str,default=v)
    @classmethod
    def extract_args(cls,args):
        d={k:v for k,v in vars(args).items()}
        kwargs={k:d[k] for k in cls.default}
        return kwargs
    def compute_tunix(self,timestamp):
        return timestamp.astype(int)//10**9
    def compute_t_lats_lons(self,df):
        lats = df.latitude.values
        lons = df.longitude.values
        t = self.compute_tunix(df.timestamp).values
        t = t - t[0]
        return t,lats,lons
    def compute_angle(self,crs_dest,t,lats,lons):
        crs_geo = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs_geo, crs_dest, always_xy=True)
        x,y = transformer.transform(lons,lats)
        return np.unwrap(compute_angle(t,x,y,self.params),period=360)
    def groupby_and_apply(self, df,by=None):
        if by is None:
            by=self._constantv
        return df.groupby(by=list(by))[list(df)].apply(self.apply).reset_index(drop=True)
    def apply(self, df):
        return self.apply_splitted(df,lambda x: self._apply(x).astype({k:np.int64 for k in self._integersv}),self.params["timesplit"])
    def apply_splitted(self,df,f,timesplit):
        isok = (df["timestamp"].diff().dt.total_seconds().to_numpy()>timesplit)
        df["splitted"]= isok.cumsum()
        res=df.groupby(by="splitted")[list(df)].apply(f,include_groups=False)
        return res
    def isvalid(self,r,df):#,latsin,lonsin,indexes):
        latsin = df.latitude.values
        lonsin = df.longitude.values
        i,j = r.interval
        res = i+1<j and latsin[i]!=latsin[j] and lonsin[i]!=lonsin[j]
        return res
    def _apply(self, df):
        # print("in Dectect._apply")
        df = df.reset_index(drop=True)
        lats = df.latitude.values
        lons = df.longitude.values
        track = df.track.values
        #nonnan = (track==track) & (lats==lats) &(lons==lons)#np.ones(lats.shape,dtype=bool)
        nonnan = (~np.isnan(track)) & (~np.isnan(lats)) &(~np.isnan(lons))
        #print(nonnan.dtype)
        #assert(not nonnan.isna().any())
        dfnonan = df.loc[nonnan].reset_index(drop=True)
        d = {k:[] for k in self._all}
        n = dfnonan.shape[0]
        if n<=2:
            return pd.DataFrame(d)#.sort_values(by=["start"]).reset_index(drop=True)
        def process(s,iswhat):
            for r in s:
                if self.isvalid(r,dfnonan):#,latsin,lonsin,indexes):#i+1<j and latsin[i]!=latsin[j] and lonsin[i]!=lonsin[j]:
                    self.process_one(dfnonan,d,r,iswhat)
        for k,s in self.extract_segments(dfnonan).items():
            process(s,k)
        for k in ["stop","start"]:
            d[k] = np.array(d[k])
        res = pd.DataFrame(d).sort_values(by=["start"]).reset_index(drop=True)
        for k,v in self.params.items():
            res[k]=v
        return res#.astype({k:np.int64 for k in self._integersv})
    
    def tag_pure(self,detected,r=0.5,dolmax=30.):
        dfl = detected.query('iswhat==@self.name_is_loxodromy').copy()
        dfo = detected.query('iswhat==@self.name_is_orthodromy').copy()
        dfl["me"] = dfl["dlmax"]
        dfo["me"] = dfo["domax"]
        dfl["other"] = dfl["domax"]
        dfo["other"] = dfo["dlmax"]
        nf = pd.concat([dfl,dfo],ignore_index=True)
        nf["pure"]= (nf.dolmax>dolmax) & (nf.me <nf.other * r) & (nf.me < nf.other * r)
        return nf
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
    )
    @classmethod
    def add_parser(cls,parser):
        parser.add_argument('-flightplans',type=str,required=True)
        super().add_parser(parser)
    @classmethod
    def extract_args(cls,args):
        kwargs = super().extract_args(args)
        kwargs['fpdatabase']=FlightPlanDatabase(args.flightplans)
        return kwargs
    def __init__(self,fpdatabase, **kwargs):
        super().__init__()
        self.params = {**self.default, **kwargs}
        self.params["thresh_slope"]=None
        self.params["thresh_border"] = None
        self.fpdatabase=fpdatabase
    def extract_segments(self,df):
        flightplan = self.fpdatabase.extract_flightplan(df)#self.extract_flightplan(self.flightplans,df)
        if flightplan == []:
            return {}
        t,lats,lons = self.compute_t_lats_lons(df)
        track = df.track.values
        n =lats.shape[0]
        clat = lats[n//2]
        clon = lons[n//2]
        crs_dest = CRS.from_proj4(f"+proj=gnom +lat_0={clat} +lon_0={clon} +datum=WGS84 +units=m +no_defs")
        angle = self.compute_angle(crs_dest,t,lats,lons)
        tf = Traffic(df)
        indexes = detect(tf,flightplan,self.params)#extractor(df) #douglas_peucker_xy(t,track,x,y,criterias,params)
        s = {self.params["name_is_orthodromy"]:[build(lats,lons,iseg,cstep,i,j,angle[i:j+1]) for iseg,cstep,i,j in indexes]}
        # print(s)
        return s


def test_one(cls,prefix):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import time
    from datetime import datetime
    from figures import read_config
    import filter_trajs
    import pandas as pd
    import argparse

    config = read_config()
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    cls.add_parser(parser)
    parser.add_argument('-r',type=float,default=0.5)
    parser.add_argument('-dolmax',type=float,default=30)
    parser.add_argument('-folderfigures',type=str)
    parser.add_argument('-trajfile',type=str)
    parser.add_argument('-identifiedcsv',type=str)
    args = parser.parse_args()
    if args.folderfigures is not None:
        SAVEFIG = True
        DEBUG = True
        FOLDER_FIGURES = args.folderfigures
    icao24='a1f1a0';start=1659854804;stop=1659855207
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
    t0 = time.time()
    groupby = ["icao24","callsign"]
    kwargs = cls.extract_args(args)
    res = cls(**kwargs).groupby_and_apply(flights)
    res.to_csv(f"{args.folderfigures}/{args.identifiedcsv}")
    res["dangle"] = res["maxangle"]-res["minangle"]
    print(time.time()-t0)
    print(res)
    for _,df in res.groupby(by=groupby):
        dfl=df.query("iswhat=='loxodromy'").copy()
        dfo=df.query("iswhat=='orthodromy'").copy()
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
            plt.savefig(f"{args.folderfigures}/{prefix}latlon.pdf", dpi=300, bbox_inches='tight')
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
                numero.append(i)
                go=plt.scatter(seg.timestamp,seg.track,marker=marker,s=s)
                go.set_label(f"{what[3:]} #{i}")
            nf.loc[:,"segment number"]=numero
            #raise Exception(nf)
            plt.xlabel("time")
            plt.ylabel("ADS-B track angle [°]")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.xticks(rotation=45)
        plt.legend(frameon=False,handletextpad=0.2)
        if SAVEFIG:
            fig.set_tight_layout({'pad':0})
            fig.set_figwidth(4)
            plt.savefig(f"{args.folderfigures}/{prefix}timetrack.pdf", dpi=300, bbox_inches='tight')
            # plt.clf()
        else:
            plt.show()
        dfl["me"] = dfl["dlmax"]
        dfo["me"] = dfo["domax"]
        dfl["other"] = dfl["domax"]
        dfo["other"] = dfo["dlmax"]
        nf = pd.concat([dfl,dfo],ignore_index=True)
        nf["pure"]= (nf.dolmax>args.dolmax) & (nf.me <nf.other * args.r) & (nf.me < nf.other * args.r)
        nf["maxloxo"]=dfl["segment number"].max()
        nf["maxortho"]=dfo["segment number"].max()
#        nf = nf.query("dolmax>@args.dolmax").query("me<dolmax*@args.r").query("me<other*@args.r")
        nf["identified"] = nf["iswhat"]
        nf["ortho-loxo [m]"]=nf["dolmax"]
        nf["adsb-loxo [m]"]=nf["dlmax"]
        nf["ortho-adsb [m]"]=nf["domax"]
        nf["duration [s]"]=nf["stop"]-nf["start"]
        if args.folderfigures is not None:
            with open(f"{args.folderfigures}/table{prefix}.tex",'w') as f:
                tf = nf.query("pure")
                f.write(tf[["identified","segment number","duration [s]","ortho-loxo [m]","adsb-loxo [m]","ortho-adsb [m]"]].to_latex(index=False,float_format="%.2f"))
        nf[["identified","pure","maxloxo","maxortho","segment number","start","stop"]].to_csv(f"{args.folderfigures}/withpure{args.identifiedcsv}",index=False)

if __name__ == '__main__':
    test_one(DetectOrthodromyWithBeacons,"classic")
