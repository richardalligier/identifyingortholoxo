from sklearn import tree
import numpy as np

import matplotlib.pyplot as plt

DEBUG = False
SAVEFIG = False
FOLDER_FIGURES = "./figures"

def compute_lever(a):
    n = a.shape[0]
    if n<=1:
        return 0.
    v = a - np.mean(a)
    x = 2*np.linspace(-1,1,num=n)#2*np.arange(n)/(n-1)
    # x = np.sign(x)
    # v = np.sign(v)
    # print(x)
    return np.abs(np.mean(v*x))

# def compute_lever(err):
#     n = err.shape[0]
#     if n == 1:
#         return 0.
#     m = (n-1)/2
#     x = (np.arange(n)-m)/(n-1)
#     return np.sum(x*err)





def thresh_tree(tree,X):
    # print(tree.feature!=0)
    thresholds = sorted(tree.threshold[tree.feature==0])
    lower = np.array([X[0,0]] + list(np.array(thresholds)+1),dtype=np.int64)
    upper = np.array(thresholds + [X[-1,0]],dtype=np.int64)
    return lower, upper


# def check_interval(err, eps, lever_crit):
#     return np.abs(err).max()< eps and np.abs(compute_lever(err))/eps<lever_crit

# def split_global(X,angle,ccp_alphas):
#     itrain = len(ccp_alphas)-1 if g(len(ccp_alphas)-1) else dicho(0,len(ccp_alphas)-1)
#     clf = tree.DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alphas[itrain])
#     clf.fit(X, angle)
#     if DEBUG:
#         print(clf.get_n_leaves())
#     lower,upper = thresh_tree(clf.tree_,X)
#     return lower,upper


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

def simplifytree(tree,criterias,angle):
    # print(tree)
    # check_tree(tree,criterias,angle)
    def aux(t):
        if t.isleaf():
            a = t.getinterval(angle)
            e = a - np.mean(a)
            # assert(check_one(criterias,a,e))
            return t
        else:
            l = aux(t.l)
            r = aux(t.r)
            if r.isleaf() and l.isleaf():
                a = t.getinterval(angle)
                e = a - np.mean(a)
                if check_one(criterias,a,e):
                    return Node(t.interval,None,None)
            return Node(t.interval,l,r)
    return aux(tree)

# class Criteria_Lever:
#     def __init__(self,eps,lever_crit):
#         self.eps = eps
#         self.lever_crit = lever_crit
#     def __call__(self,angle,err):
#         # err = angle - np.mean(angle)
#         # assert(np.abs(compute_lever(err))/self.eps < self.lever_crit)
#         return np.abs(compute_lever(angle))/self.eps < self.lever_crit

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



def identify_constant_with_tree_simpler(angle,criterias):
    clf = tree.DecisionTreeRegressor(random_state=0,criterion="absolute_error")
    X = np.arange(angle.shape[0])[:,None]
    clf.fit(X,angle)
    mytree =  translatetree(clf.tree_)
    # preds = clf.predict(X)
    # lowerupper = mytree.extract_lower_upper(X[:,-1])
    # lowerupper = tuple(map(np.array, zip(*lowerupper)))
    # plotdebug(angle,lowerupper,preds)
    stree = simplifytree(mytree,criterias,angle)
    lowerupper = stree.extract_lower_upper(X[:,-1])
    lowerupper = tuple(map(np.array, zip(*lowerupper)))
    preds = stree.predict(X[:,-1],angle)
    # print(stree)
    # print(preds)
    # raise Exception
    plotdebug(angle,lowerupper)
    return lowerupper,preds


def plotdebug(angle,lowerupper,fname,what):
    lower,upper = lowerupper
    if DEBUG:
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
        if SAVEFIG:
            fig.set_tight_layout({'pad':0})
            fig.set_figwidth(4)
            plt.savefig(f"{FOLDER_FIGURES}/{fname}{what}.pdf", dpi=300, bbox_inches='tight')
        else:
            plt.show()

# def identify_constant_with_tree(angle,criterias):
#     global iloxortho
#     clf = tree.DecisionTreeRegressor(random_state=0,criterion="absolute_error")
#     X = np.arange(angle.shape[0])[:,None]
#     path = clf.cost_complexity_pruning_path(X, angle)
#     ccp_alphas, impurities = path.ccp_alphas, path.impurities

#     def g(i):
#         clf = tree.DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alphas[i])
#         clf.fit(X, angle)
#         lowerupper = thresh_tree(clf.tree_,X)
#         pred = clf.predict(X)
#         return check(criterias,angle,angle-pred,lowerupper)

#     def dicho(i,j):#g(i) True, g(j) False
#         if j-i<=1:
#             # clf = tree.DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alphas[i])
#             # clf.fit(X, angle)
#             return i
#         else:
#             m = (i+j)//2
#             if g(m):
#                 return dicho(m,j)
#             else:
#                 return dicho(i,m)
#     if DEBUG:
#         print(f"{eps=} {g(0)=} {g(len(ccp_alphas)-1)=}")
#     assert(g(0))
#     # assert(not g(len(ccp_alphas)-1))
#     itrain = len(ccp_alphas)-1 if g(len(ccp_alphas)-1) else dicho(0,len(ccp_alphas)-1)
#     clf = tree.DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alphas[itrain])
#     clf.fit(X, angle)
#     translatetree(clf.tree_)
#     # raise Exception
#     # for ccp_alpha in reversed(ccp_alphas):
#     #     # ccp_alpha = ccp_alphas[-23]
#     #     clf = tree.DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
#     #     clf.fit(X, angle)
#     #     # print(ccp_alpha)
#     #     # print(np.abs(clf.predict(X)-angle).max())
#     #     if np.abs(clf.predict(X)-angle).max()<eps:
#     #         break
#     if DEBUG:
#         print(clf.get_n_leaves())
#     lowerupper = thresh_tree(clf.tree_,X)
#     preds = clf.predict(X)
#     plotdebug(angle,lowerupper,preds)
#     return lowerupper, preds
