import cdsapi
import iris
from iris.analysis import MEAN
from iris.coord_categorisation import add_day_of_year
from iris.util import equalise_attributes

import constrain

c = cdsapi.Client()

year_start, year_end = 1979, 2023
for variable in ["u_component_of_wind", "v_component_of_wind"]:
    for year in range(year_start, year_end):
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": [variable],
                "pressure_level": list(range(200, 600+1, 50)),
                "year": year,
                "month": ["11", "12", "1", "2", "3"],
                "day": list(range(1, 31 + 1)),
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": [90, -180, 0, 180],
            },
            "download.nc".format(year),
        )

        # Calculate daily mean and coarse grain
        cube = iris.load_cube("download.nc")
        cube.coord("longitude").circular = True
        cube_coarse = constrain.regrid_to_coarsest(cube)

        add_day_of_year(cube_coarse, "time")
        cube_coarse_daily_mean = cube_coarse.aggregated_by("day_of_year", MEAN)

        iris.save(cube_coarse_daily_mean, "download_{}.nc".format(year))

    # Collapse the downloaded files into a single file
    cubes = iris.load("download_*.nc")
    equalise_attributes(cubes)
    cubes = cubes.concatenate_cube()
    iris.save(cubes, "{}_daily-mean_era5_CanESM5-grid-0-90N-600-200hPa_NDJFM-1979-2022.nc".format(
        variable[0], year_start, year_end
    ))
