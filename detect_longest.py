import numpy as np
from scipy.sparse import SparseEfficiencyWarning
from collections import deque
from detect_classic import build, Segment, Detect, compute_angle,reducetrack
import detect_classic
import warnings
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
from pyproj import CRS, Transformer
import piecewise


def extract_longest(arr, max_width):
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
            (arr[r+1] - arr[min_dq[0]] > max_width) or
            (arr[max_dq[0]] - arr[r+1] > max_width)
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


def extract_longest_segments(angle,params):
    lowerupper = list(extract_longest(angle, params["track_tolerance_degrees"]))# if keep(angle,ij,params)]
    return [Segment(iseg,dict(),i,j) for iseg,(i,j) in enumerate(lowerupper)]


def isolate_longest_constant_heading(t,angle,params):
    n=angle.shape[0]
    #angle = np.unwrap(compute_angle(t,xr,yr,params),period=360)#np.degrees(np.unwrap(np.arctan2(dy,dx)))
    lowerupper = extract_longest_segments(angle,params)
    newres = []
    for s in lowerupper:
        slope,r = reducetrack(s.interval,t,angle,params)
        if r is not None:
            s.debugdata["slope"]=slope
            newres.append(Segment(s.iseg,s.debugdata,r[0],r[1]))
    lowerupper = newres
    lowerupper = sortbysizeandfilterbyintersection(lowerupper,angle,params["thresh_iou"])
    if lowerupper == []:
        return []
    else:
        (lower,upper) = tuple(map(np.array, zip(*[s.interval for s in lowerupper])))
    assert(upper.max()<n)
    return [s.totuple() for s in lowerupper]

class DetectLongestOrthodromyLoxodromy(Detect):
    default = dict(
        track_tolerance_degrees = 0.5,
        smooth=1e-2,
        thresh_iou=0.1,
        model="mean",
        thresh_slope = 0.001,
        thresh_border = 0.1,
        timesplit=3600.,
    )
    def __init__(self, **kwargs):
        super().__init__()
        self.params = {**self.default, **kwargs}
        assert(self.params["model"] is not None)
    def extract_segments_xy(self,crs_dest,df,what):
        t,lats,lons = self.compute_t_lats_lons(df)
        angle = self.compute_angle(crs_dest,t,lats,lons)
        segments = isolate_longest_constant_heading(t,angle,self.params)
        s = [build(lats,lons,iseg,cstep,i,j,angle[i:j+1]) for iseg,cstep,i,j in segments]
        return s
    def get_dcrs(self,df):
        t,lats,lons = self.compute_t_lats_lons(df)
        n =lats.shape[0]
        clat = lats[n//2]
        clon = lons[n//2]
        dcrs = {
            self.name_is_orthodromy:CRS.from_proj4(f"+proj=gnom +lat_0={clat} +lon_0={clon} +datum=WGS84 +units=m +no_defs"),
            self.name_is_loxodromy:CRS.from_proj4("+proj=merc +datum=WGS84 +units=m +no_defs"),
        }
        return dcrs
    def extract_segments(self,df):
        dcrs = self.get_dcrs(df)
        ds ={}
        for k,crs in dcrs.items():
            ds[k]=self.extract_segments_xy(crs,df,k)#,self.criterias,self.params)
        return ds



if __name__ == '__main__':
    class DetectLongestOrthodromyLoxodromyDebug(DetectLongestOrthodromyLoxodromy):
        def extract_segments(self,df):
            t,lats,lons = self.compute_t_lats_lons(df)
            dcrs = self.get_dcrs(df)
            ds ={}
            for k,crs in dcrs.items():
                ds[k]=self.extract_segments_xy(crs,df,k)#,self.criterias,self.params)
                angle= self.compute_angle(crs,t,lats,lons)
                (lower,upper) = tuple(map(np.array, zip(*[s.interval for s in ds[k]])))
                piecewise.plotdebug(angle,(lower,upper),k,f"{k}after.pdf")
            return ds
    detect_classic.test_one(DetectLongestOrthodromyLoxodromyDebug,"longest")
