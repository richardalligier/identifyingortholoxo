import numpy as np
import matplotlib.pyplot as plt

def compute_lever(a):
    n = a.shape[0]
    if n<=1:
        return 0.
    v = a - np.mean(a)
    x = 2*np.linspace(-1,1,num=n)#2*np.arange(n)/(n-1)
    return np.abs(np.mean(v*x))



def thresh_tree(tree,X):
    # print(tree.feature!=0)
    thresholds = sorted(tree.threshold[tree.feature==0])
    lower = np.array([X[0,0]] + list(np.array(thresholds)+1),dtype=np.int64)
    upper = np.array(thresholds + [X[-1,0]],dtype=np.int64)
    return lower, upper




class Node:
    def __init__(self,interval,l,r):
        self.l=l
        self.r=r
        self.interval=interval
    def depth(self):
        if isleaf():
            return 0
        return 1+max(self.r.depth(),self.l.depth())
    def isleaf(self):
        return self.l is None and self.r is None
    def __repr__(self):
        return f"Node({self.interval},\n{self.l},\n{self.r})"
    def getinterval(self,a):
        i,j = self.interval
        j = None if j is None else j+1
        return a[i:j]
    def getXrange(self,X):
        i,j = self.interval
        i = 0 if i is None else i
        j = int(X.max()) if j is None else j
        return i,j
    def extract_lower_upper(self,X):
        if self.isleaf():
            i,j = self.getXrange(X)
            return [(i,j)]
        else:
            l = [] if self.l is None else self.l.extract_lower_upper(X)
            r = [] if self.r is None else self.r.extract_lower_upper(X)
            return l+r
    def predict(self,X,angle):
        res = np.zeros_like(X)*np.nan
        def aux(t):
            if t.isleaf():
                m = np.mean(t.getinterval(angle))
                i,j = t.getXrange(X)
                res[np.logical_and(i<=X,X<=j)]=m
            else:
                if t.r is not None:
                    aux(t.r)
                if t.l is not None:
                    aux(t.l)
        aux(self)
        assert(np.isnan(res).sum()==0)
        return res

def translatetree(tree):
    def aux(i,lb,ub):
        t = tree.threshold[i]
        if tree.children_left[i]==-1:
            l=None
        else:
            l = aux(tree.children_left[i],lb,int(t))
        if tree.children_right[i]==-1:
            r=None
        else:
            r = aux(tree.children_right[i],int(t+1),ub)
        return Node((lb,ub),l,r)
    return aux(0,None,None)

def check_tree(tree,criterias,angle):
    if tree.isleaf():
        a = tree.getinterval(angle)
        print(a.shape[0])
        print(a)
        e = a - np.mean(a)
        assert(check_one(criterias,a,e))
    else:
        if tree.l is not None:
            check_tree(tree.l,criterias,angle)
        if tree.r is not None:
            check_tree(tree.r,criterias,angle)


class Criteria_Range:
    def __init__(self,eps):
        self.eps = eps
    def __call__(self,angle,err):
        mi = np.min(angle)
        ma = np.max(angle)
        return ma-mi < self.eps

class Criteria_RangeOld:
    def __init__(self,eps):
        self.eps = eps
    def __call__(self,angle,err):
        return np.abs(err).max() < self.eps


class Criteria_MeanAbs:
    def __init__(self,eps):
        self.eps = eps
    def __call__(self,angle,err):
        v = np.mean(angle)
        return np.mean(np.abs(angle-v)) < self.eps

def check_one(criterias,a,e):
    for c in criterias:
        if not c(a,e):
            return False
    return True

# def check_one(criterias,a,e):
#     for c in criterias:
#         if not c(a,e):
#             return False
#     return True

def check(criterias,angle,err,lowerupper):
    lower, upper = lowerupper
    for l,u in zip(lower,upper):
        a = angle[l:u+1]
        e = err[l:u+1]
        if not check_one(criterias,a,e):
            return False
    return True



def plotdebug(angle,lowerupper,what,fname):
    lower,upper = lowerupper
    # if iloxortho%2==0:
    #     proj="gnomonic"
    # else:
    #     proj="Mercator"
    fig = plt.figure()
    line,=plt.plot(angle,linewidth=3,c="black")
    line.set_label("ADS-B trajectory")
    # for (i,j) in zip(thresholds[:-1],thresholds[1:]):
    for (i,j) in zip(lower,upper):
        x =np.arange(i,j+1)
        line,=plt.plot(x,np.ones_like(x)*np.mean(angle[i:j+1]),c="red")
    line.set_label("fitted step-wise function")
    plt.xlabel("point index [-]")
    plt.ylabel(f"track angle after {what} projection [°]")
    plt.legend(frameon=False)
    if True:
        fig.set_tight_layout({'pad':0})
        fig.set_figwidth(4)
        plt.savefig(f"{fname}.pdf", dpi=300, bbox_inches='tight')
    else:
        plt.show()
