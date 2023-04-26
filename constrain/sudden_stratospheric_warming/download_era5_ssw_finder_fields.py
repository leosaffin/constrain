import pathlib

import cdsapi
import iris
from iris.analysis import MEAN, Linear
from iris.coord_categorisation import add_day_of_year, add_season_year
from iris.util import equalise_attributes

from irise import constants
import constrain

c = cdsapi.Client()

for year in range(1940, 2020+1):
    print(year)

    fname = 'era5_ssw_finder_fields_{}.nc'.format(year)

    c.retrieve(
       'reanalysis-era5-pressure-levels',
       {
           'product_type': 'reanalysis',
           'format': 'netcdf',
           'variable': [
               'geopotential', 'u_component_of_wind',
           ],
           'pressure_level': '10',
           'year': year,
           'month': ['01', '02', '03', '04', '11', '12'],
           'day': list(range(1, 31+1)),
           'time': ['00:00', '06:00', '12:00', '18:00'],
           'area': [90, -180, 0, 180],
       },
       fname)

    cubes = iris.load(fname)

    # Geopotential Height
    gph = cubes.extract_cube("geopotential") / constants.g
    add_day_of_year(gph, "time")
    gph = gph.aggregated_by("day_of_year", MEAN)
    gph.coord("longitude").circular = True
    gph = constrain.regrid_to_coarsest(gph)
    iris.save(gph, "era5_daily-mean-z-10hPa_{}.nc".format(year))

    # Zonal-mean zonal wind at 60N
    u = cubes.extract_cube("eastward_wind")
    add_day_of_year(u, "time")
    u = u.aggregated_by("day_of_year", MEAN)
    u = u.interpolate([("latitude", 60)], Linear())
    u = u.collapsed(["longitude"], MEAN)
    iris.save(u, "era5_daily-mean-zonal-mean-u-10hPa_{}.nc".format(year))

    pathlib.Path(fname).unlink()

for filename in [
    "era5_daily-mean-zonal-mean-u-10hPa_*.nc",
    "era5_daily-mean-z-10hPa_1940-2020.nc",
]:
    cubes = iris.load(filename)
    equalise_attributes(u)
    cube = cubes.concatenate_cube()
    add_season_year(cube, "time", seasons=["ndjfma", "mjjaso"])
    iris.save(cube, filename.replace("*", "1940-2020"))
