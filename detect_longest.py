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

def isolate_longest_constant_heading(t,track,xr,yr,params,what):
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
        segments = isolate_longest_constant_heading(t,track,x,y,self.params,what)
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


if __name__ == '__main__':
    detect_classic.test_one(DetectLongestOrthodromyLoxodromy,"longest")
