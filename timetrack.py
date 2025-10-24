# ruff: noqa: E402
# %%
from datetime import datetime
from pathlib import Path

import pandas as pd  # type: ignore
from traffic.core import Flight, Traffic

from detect_longest import DetectLongestOrthodromyLoxodromy

icao24 = "a1f1a0"
start = 1659854804
stop = 1659855207
selecteddate = datetime.fromtimestamp(start).strftime("%Y-%m-%d")
p = Path(f"./outfiles/trajs/{selecteddate}.parquet")
assert p.exists()
t = Traffic.from_file(p).between(start - 1200, stop + 1200)
assert t is not None
flight = t[icao24]
flight = flight.assign(date=flight.data.timestamp.dt.date)
flight

# %%
res = DetectLongestOrthodromyLoxodromy(
    timesplit=3600,
    smooth=0.01,
    thresh_iou=0.1,
    track_tolerance_degrees=0.5,
    thresh_slope=0.001,
    thresh_border=0.1,
    r=0.5,
    dolmax=30,
).apply(flight.data)
res = res.assign(dangle=res["maxangle"] - res["minangle"])
res = (
    res.reset_index()
    .assign(
        start=pd.to_datetime(res.start.values, unit="s", utc=True),
        stop=pd.to_datetime(res.stop.values, unit="s", utc=True),
    )
    .drop(columns="date")
)
res
# %%
import altair as alt
import csaps
import numpy as np
from pyproj import Proj


def compute_angle(t, xr, yr, params=dict(smooth=0.01)):
    # n = xr.shape[0]
    xy = [xr, yr]
    sxy = csaps.csaps(t, xy, smooth=params["smooth"])
    # pxy = sxy(t, nu=0)
    dx, dy = sxy(t, nu=1)
    angle = np.degrees((np.arctan2(dy, dx)))
    return angle


def compute_angle_flight(flight: Flight) -> Flight:
    df = flight.data
    t = (df.timestamp - df.timestamp.min()).dt.total_seconds().to_numpy()
    xr = df.x.to_numpy()
    yr = df.y.to_numpy()
    angle = np.unwrap(90 - compute_angle(t, xr, yr, params=dict(smooth=0.01)))
    return flight.assign(compute_track=angle)


# this helps to align the x axis
flight = flight.between(res.start.min(), res.stop.max())  # type: ignore
assert flight is not None

chart = (
    alt.vconcat(
        (
            flight.compute_xy(Proj(proj="merc"))
            .pipe(compute_angle_flight)
            .rename(columns={"compute_track": "loxodromy"})  # type: ignore
            .compute_xy(
                Proj(
                    proj="gnom",
                    lat_0=flight.latitude_mean,
                    lon_0=flight.longitude_mean,
                )
            )
            .pipe(compute_angle_flight)
            .rename(columns={"compute_track": "orthodromy"})
            .drop(columns=["date", "serials", "spi", "squawk", "hour"])
            .chart()
            .transform_fold(["loxodromy", "orthodromy"], as_=["type", "value"])
            .mark_line()
            .encode(
                alt.X("utchoursminutesseconds(timestamp):T")
                .title(None)
                .axis(format="%H:%M"),
                alt.Y("value:Q")
                .scale(domain=[75, 130])
                .title("Angle (in degrees)")
                .axis(
                    titleAngle=0,
                    titleAnchor="end",
                    titleAlign="left",
                    titleY=-18,
                ),
                alt.Color("type:N").legend(None),
            )
            .properties(height=200, width=500)
        ),
        alt.Chart(res)
        .mark_bar(height=10)
        .encode(
            alt.X("utchoursminutesseconds(start):T").title(None).axis(format="%H:%M"),
            alt.X2("utchoursminutesseconds(stop):T"),
            alt.Y("iswhat:N").title(None),
            color="iswhat:N",
        )
        .properties(width=500),
    )
    .resolve_axis(x="shared")
    .configure_axis(
        labelFontSize=16,
        titleFontSize=18,
        labelFont="Roboto Condensed",
        titleFont="Roboto Condensed",
    )
)
chart.save("ortho_loxo.pdf")
chart
