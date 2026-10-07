"""Download helpers for the city bike project.

Raw files are cached in data/raw, so each source is downloaded only once.

Sources
-------
- HSL city bike origin-destination trips (CC BY 4.0): https://dev.hsl.fi/citybikes/
- HSL city bike stations via Helsinki Region Infoshare (CC BY 4.0):
  https://hri.fi/data/en_GB/dataset/hsl-n-kaupunkipyoraasemat
- FMI hourly weather observations, Helsinki Kaisaniemi (CC BY 4.0): https://opendata.fmi.fi/
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen, urlretrieve

import pandas as pd

TRIP_URL = "https://dev.hsl.fi/citybikes/od-trips-{year}/{year}-{month:02d}.csv"
STATION_URL = "https://opendata.arcgis.com/datasets/726277c507ef4914b0aec3cbcfcbfafc_0.csv"
FMI_URL = "https://opendata.fmi.fi/wfs"
KAISANIEMI_FMISID = 100971
FMI_PARAMETERS = {
    "TA_PT1H_AVG": "temperature_c",
    "PRA_PT1H_ACC": "precipitation_mm",
    "RH_PT1H_AVG": "humidity_pct",
    "WS_PT1H_AVG": "wind_ms",
}
SEASON_MONTHS = range(4, 11)  # the bikes are in use from April to October


def download_trips(years, raw_dir):
    paths = []
    for year in years:
        for month in SEASON_MONTHS:
            path = Path(raw_dir) / f"od-{year}-{month:02d}.csv"
            if not path.exists():
                print(f"Downloading {path.name}")
                urlretrieve(TRIP_URL.format(year=year, month=month), path)
            paths.append(path)
    return paths


def download_stations(raw_dir):
    path = Path(raw_dir) / "stations.csv"
    if not path.exists():
        urlretrieve(STATION_URL, path)
    return path


def _fetch_fmi_month(start, end):
    query = urlencode({
        "service": "WFS",
        "version": "2.0.0",
        "request": "getFeature",
        "storedquery_id": "fmi::observations::weather::hourly::simple",
        "fmisid": KAISANIEMI_FMISID,
        "parameters": ",".join(FMI_PARAMETERS),
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    with urlopen(f"{FMI_URL}?{query}", timeout=120) as response:
        root = ET.fromstring(response.read())
    ns = {"BsWfs": "http://xml.fmi.fi/schema/wfs/2.0"}
    rows = []
    for element in root.iter("{http://xml.fmi.fi/schema/wfs/2.0}BsWfsElement"):
        rows.append({
            "time_utc": element.find("BsWfs:Time", ns).text,
            "parameter": element.find("BsWfs:ParameterName", ns).text,
            "value": float(element.find("BsWfs:ParameterValue", ns).text),
        })
    return pd.DataFrame(rows)


def download_weather(years, raw_dir):
    """Hourly observations; FMI allows at most 744 hours per request, so fetch monthly."""
    path = Path(raw_dir) / "weather_kaisaniemi.csv"
    if path.exists():
        return path
    frames = []
    for year in years:
        for month_start in pd.date_range(f"{year}-03-01", f"{year}-11-01", freq="MS", tz="UTC"):
            frames.append(_fetch_fmi_month(month_start, month_start + pd.offsets.MonthBegin(1)))
    long = pd.concat(frames, ignore_index=True).drop_duplicates(["time_utc", "parameter"])
    wide = (
        long.pivot(index="time_utc", columns="parameter", values="value")
        .rename(columns=FMI_PARAMETERS)
        .reset_index()
    )
    wide.to_csv(path, index=False)
    return path
