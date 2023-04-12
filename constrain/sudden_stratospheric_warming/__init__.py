"""
The module uses code adapted from the repository
https://github.com/BrisClimate/Persistent_model_biases_in_the_CMIP6_representation_of_stratospheric_polar_vortex_variability
which in turn uses code from
https://github.com/wseviour/vortex-moments/

The vortex moments code is downloaded into this module when it is first imported

"""

import pathlib

import iris
import requests
import datetime

import numpy as np
import more_itertools as mit
from pandas import DataFrame

from irise.grid import get_datetime


# Download vortex moments code and add as importable modules here when the module
# is first loaded.
def _download_vortex_moments_dependency():
    here = pathlib.Path(__file__).parent

    repository = "https://raw.githubusercontent.com/wseviour/vortex-moments/"
    revision = "323abf19b015b19a1177d238ec32be35ebcf08c5"
    for filename in ["vor_fast.py", "vor_fast_setup.py"]:
        file_path = repository + revision + "/" + filename
        r = requests.get(file_path)
        with open(here / filename, "w") as f:
            f.write(r.text)


try:
    from constrain.sudden_stratospheric_warming import vor_fast, vor_fast_setup
except ImportError:
    print("Downloading Vortex Moments Dependency")
    _download_vortex_moments_dependency()
    from constrain.sudden_stratospheric_warming import vor_fast, vor_fast_setup


def find_ssws(u10):
    """Breakdown of the stratospheric polar vortex defined by a reversal of the
    zonal-mean zonal wind at 60N, 10hPa

    Adapted from
    https://github.com/BrisClimate/Persistent_model_biases_in_the_CMIP6_representation_of_stratospheric_polar_vortex_variability

    Args:
        u10 (iris.cube.Cube): Zonal-mean zonal wind as a function of time. Should be at
            10hPa and 60N but can be at other locations.

    Returns:
        pandas.DataFrame:
            A DataFrame with columns of "Start" and "End" and each row being an event
    """
    # find indices of dates of wind reversal
    idx = np.where(u10.data < 0)
    times = get_datetime(u10)

    # group the indices into events
    events = [list(group) for group in mit.consecutive_groups(idx[0])]

    ssws = DataFrame()
    ssws["Start"] = [times[event[0]] for event in events]
    ssws["End"] = [times[event[-1]] for event in events]

    if len(ssws) == 0:
        print("No events")
        return ssws
    else:
        print(ssws, "\n")

    # check last event is not a final warming. If it is, remove
    if ssws.iloc[-1]["End"] > (times[-1] - datetime.timedelta(days=10)):
        print("Removing final warming\n")
        ssws = ssws.drop(index=ssws.index[-1])

    # delete any events starting in April or November
    remove = []
    for idx, ssw in ssws.iterrows():
        if ssw["Start"].month in (4, 11):
            remove.append(idx)

    if len(remove) > 0:
        print("events starting in April/November\n", ssws.iloc[remove], "\n")
        ssws = ssws.drop(index=remove)

    # for  events, check if 20 consecutive days of westerlies before events
    remove = []
    for idx, ssw in ssws.iterrows():
        t_start = ssw["Start"] - datetime.timedelta(days=20)
        u10_before = u10.extract(iris.Constraint(
            time=lambda cell:  t_start <= cell.point < ssw["Start"])
        )
        if np.any(u10_before.data < 0):
            remove.append(idx)

    if len(remove) > 0:
        print("Events without 20 consecutive days of westerlies prior\n", ssws.iloc[remove], "\n")
        ssws = ssws.drop(index=remove)

    # check that there are at least 10 consecutive days of westerlies at some point
    # after last remaining event
    remove = []
    for idx, ssw in ssws.iterrows():
        u10_after = u10.extract(iris.Constraint(
            time=lambda cell: cell.point > ssw["End"])
        )
        idx_end = np.where(u10_after.data > 0)
        times_end = get_datetime(u10)

        # group the indices into events
        events_end = [list(group) for group in mit.consecutive_groups(idx_end[0])]

        if all(times_end[event[-1]] - times_end[event[0]] < datetime.timedelta(days=10) for event in events_end):
            remove.append(idx)

    if len(remove) > 0:
        print("Events without 10 consecutive days of westerlies after\n", ssws.iloc[remove], "\n")
        ssws = ssws.drop(index=remove)

    if len(ssws) == 0:
        print("No events remaining")
    else:
        print(ssws, "\n")

    return ssws


def classify_ssws(ssws, gph):
    """Classify the sudden-stratospheric warming vortex type using vor_fast

    Adds a column of "Vortex Type" to the input ssws dataframe with either "split",
    "displaced", or unclassified

    Args:
        ssws (pandas.DataFrame): A dataframe containing SSW events with "Start" times.
            Produced by find_ssws().
        gph (iris.cube.Cube): Geopotential height as a function of time, latitude, and
            longitude.
    """
    # using vortex moments classify vortex as split, displacement or unclassifiable
    # NB calculate vorTex edge for each model separately and add below
    # Set up moment diagnostics
    aspect = np.empty(0)
    latcent = np.empty(0)

    # Set up cartesian mapping xypoints and restrict to NH
    lons = gph.coord("longitude").points
    lats = gph.coord("latitude").points
    gph_nh, lats_nh, xypoints = vor_fast_setup.setup(gph.data, lats, lons, "NH")

    # Use times to index events because geopotential height is a numpy array after using
    # vor_fast.setup
    times = list(get_datetime(gph))

    vortex_id = []
    for idx, ssw in ssws.iterrows():
        print("Calculating moments for event around " + str(ssw["Start"]))

        idx_start = times.index(ssw["Start"] - datetime.timedelta(days=10))
        idx_end = times.index(ssw["Start"] + datetime.timedelta(days=10))
        for iday in range(idx_start, idx_end + 1):
            moments = vor_fast.calc_moments(
                gph_nh[iday, :, :],
                lats_nh,
                lons,
                xypoints,
                hemisphere="NH",
                field_type="GPH",
                edge=3.019e4,
                resolution="low",
            )
            aspect = np.append(aspect, moments["aspect_ratio"])
            latcent = np.append(latcent, moments["centroid_latitude"])

        aspect_days = np.count_nonzero(aspect > 2.4)
        latcent_days = np.count_nonzero(latcent < 66)
        print("days above aspect threshold=", aspect_days)
        print("days below latcent threshold=", latcent_days)

        if aspect_days > latcent_days:
            vortex_type = "split"
        elif aspect_days < latcent_days:
            vortex_type = "displaced"
        else:
            vortex_type = "unclassified"
        vortex_id.append(vortex_type)

    ssws["Vortex Type"] = vortex_id
    print(ssws)
