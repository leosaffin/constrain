import pathlib

import cdsapi
import iris
from iris.analysis import MEAN
from iris.coord_categorisation import add_day_of_year

import constrain

c = cdsapi.Client()

output_filename = "era5-CanESM5-grid_daily-mean-uv-250hPa_North-Atlantic_NDJFMA_{}.nc"
for year in range(1940, 2022 + 1):
    print(year)

    fname_temp = "download_{}.nc".format(year)
    fname = output_filename.format(year)

    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            "pressure_level": 250,
            "year": year,
            "month": ["01", "02", "03", "04", "11", "12"],
            "day": list(range(1, 31 + 1)),
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": [90, -90, 20, 60],
        },
        fname_temp,
    )

    cubes = iris.load(fname_temp)

    # Regrid data to coarse grid and delete downloaded file
    newcubes = iris.cube.CubeList()
    for cube in cubes:
        add_day_of_year(cube, "time")
        cube_daily_mean = cube.aggregated_by("day_of_year", MEAN)
        cube_daily_mean_coarse_grid = constrain.regrid_to_coarsest(cube_daily_mean)
        newcubes.append(cube_daily_mean_coarse_grid)

    iris.save(newcubes, fname)
    pathlib.Path(fname_temp).unlink()

# Merge all files to single file
cubes = iris.load(output_filename.format("*"))
for cube in cubes:
    del cube.attributes["history"]

cubes = cubes.concatenate()
iris.save(cubes, output_filename.format("1940-2022"))
