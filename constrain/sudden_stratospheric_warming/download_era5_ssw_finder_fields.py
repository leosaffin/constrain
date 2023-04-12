import pathlib

import cdsapi
import iris
from iris.analysis import MEAN, Linear
from iris.coord_categorisation import add_day_of_year, add_season_year

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

u = iris.load("era5_daily-mean-zonal-mean-u-10hPa_*.nc")
for cube in u:
    del cube.attributes["history"]
u = u.concatenate_cube()
add_season_year(u, "time", seasons=["ndjfma", "mjjaso"])
iris.save(u, "era5_daily-mean-zonal-mean-u-10hPa_1940-2020.nc")

gph = iris.load("era5_daily-mean-z-10hPa_*.nc")
for cube in gph:
    del cube.attributes["history"]
gph = gph.concatenate_cube()
add_season_year(gph, "time", seasons=["ndjfma", "mjjaso"])
iris.save(gph, "era5_daily-mean-z-10hPa_1940-2020.nc")

        
        
        
        
