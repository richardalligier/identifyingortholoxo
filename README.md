# Identifying Loxodromy and Orthodromy in ADS-B Trajectories
## What is this repo ?
This repo contains code for paper LINKTOPAPER. In this paper a method is presented to identify orthodromy and loxodromy segments on ADS-B aircraft trajectory. The method `groupby_and_apply` of the class  `DetectLongestOrthodromyLoxodromy` returns a dataframe of the identified orthodromies and loxodromies, one row per segment. Each row contains statistics on the identified segment.


## How to use it
```
from traffic.data.samples import savan
from detect_longest import DetectLongestOrthodromyLoxodromy

df = savan['SAVAN02'].data
detect = DetectLongestOrthodromyLoxodromy()
detected = detect.tag_pure(detect.groupby_and_apply(df))
```
Output
```
>>> detected[["iswhat","start","stop","dolmax","domax","dlmax","pure"]]
        iswhat       start        stop        dolmax         domax         dlmax   pure
0    loxodromy  1647874366  1647874825  7.459927e+01  3.000271e+01  5.932373e+01  False
1    loxodromy  1647875192  1647875806  9.746142e+01  2.788599e+01  8.135097e+01  False
2    loxodromy  1647877929  1647878389  5.362667e+01  6.939144e+00  5.544109e+01  False
3    loxodromy  1647878460  1647878772  2.684648e+01  2.856393e+01  1.128200e+01  False
4    loxodromy  1647878942  1647879468  7.734586e+01  4.075125e+01  5.682153e+01  False
5    loxodromy  1647879408  1647879934  7.547265e+01  7.192981e+00  7.756248e+01  False
6    loxodromy  1647879887  1647880052  7.314855e+00  5.785740e+00  8.530783e+00  False
7    loxodromy  1647880269  1647880791  7.148535e+01  1.586464e+01  6.491823e+01  False
8    loxodromy  1647880708  1647881099  3.851243e+01  1.045363e+01  4.098257e+01  False
9    loxodromy  1647881558  1647882034  8.071446e+01  1.995143e+01  7.296240e+01  False
10   loxodromy  1647882027  1647882383  4.434739e+01  1.056861e+01  4.647918e+01  False
11   loxodromy  1647882381  1647882741  4.471434e+01  1.179279e+01  3.837158e+01  False
12   loxodromy  1647882812  1647882989  1.173725e+01  2.256231e+01  3.588679e+00  False
13   loxodromy  1647883006  1647883181  1.055387e+01  1.507304e+01  1.166948e+01  False
14   loxodromy  1647883165  1647883354  1.126291e+01  2.044374e+01  7.527972e+00  False
15   loxodromy  1647883434  1647883565  3.969228e+00  1.274712e+01  5.418004e+00  False
16   loxodromy  1647883585  1647883667  1.500197e+00  6.004350e+00  7.155667e+00  False
17   loxodromy  1647884440  1647884459  4.385401e-08  4.385401e-08  5.234188e-08  False
18  orthodromy  1647874029  1647874280  2.801182e+01  6.877272e+00  2.932055e+01  False
19  orthodromy  1647874369  1647875122  2.015530e+02  3.346121e+01  1.828253e+02   True
20  orthodromy  1647875192  1647875931  1.415236e+02  3.076907e+01  1.237799e+02   True
21  orthodromy  1647876314  1647876807  9.826881e+01  1.187715e+01  1.046233e+02   True
22  orthodromy  1647876864  1647877459  1.302921e+02  3.307359e+01  1.588335e+02   True
23  orthodromy  1647877542  1647878399  1.918017e+02  3.823146e+01  2.199159e+02   True
24  orthodromy  1647878460  1647878763  2.525637e+01  2.709579e+01  1.119593e+01  False
25  orthodromy  1647878933  1647879136  1.154050e+01  1.032526e+01  7.558357e+00  False
26  orthodromy  1647879029  1647880052  2.872141e+02  3.222593e+01  2.741458e+02   True
27  orthodromy  1647880205  1647881210  2.602575e+02  2.084766e+01  2.529046e+02   True
28  orthodromy  1647881326  1647881495  9.352496e+00  7.900666e+00  1.723846e+01  False
29  orthodromy  1647881495  1647882965  7.665329e+02  1.489871e+01  7.764106e+02   True
30  orthodromy  1647883006  1647883181  1.055387e+01  1.507304e+01  1.166948e+01  False
31  orthodromy  1647883161  1647883300  6.029689e+00  6.987414e+00  1.014055e+01  False
32  orthodromy  1647883585  1647883667  1.500197e+00  6.004350e+00  7.155667e+00  False
33  orthodromy  1647884440  1647884459  4.385401e-08  4.385401e-08  5.234188e-08  False
```
`detected` is a dataframe containing statistics on the, where:
- `iswhat` which type of curve was identified
- `start` unix timestamp of the start of the curve
- `stop` unix timestamp of the end of the curve
- `dolmax` maximum distance between the orthodromy and loxodromy connecting starting and ending points, in meters
- `dolmax` maximum distance between the orthodromy and loxodromy connecting starting and ending points, in meters
- `domax` maximum distance between the trajectory points and the orthodromy connecting starting and ending points, in meters
- `dlmax` maximum distance between the trajectory points and the loxodromy connecting starting and ending points, in meters
- `pure` whether the `iswhat` curve confidently fits the trajectory points and is very different from the other type of curve
```
import matplotlib.pyplot as plt
df = df.assign(tunix = detect.compute_tunix(df.timestamp))
dcolor = {"orthodromy":"blue","loxodromy":"red"}
plt.scatter(df.longitude,df.latitude,c="black")
for _,line in detected.query('pure').iterrows():
    seg = df.query("@line.start<=tunix<=@line.stop")
    plt.scatter(seg.longitude,seg.latitude,c=dcolor[line.iswhat],s=5)

plt.gca().set_aspect('equal')
plt.show()
```
