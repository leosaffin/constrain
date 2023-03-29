import cdsapi
import iris

from constrain import regrid_to_coarsest

c = cdsapi.Client()

for year in range(1940, 2023):
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            "pressure_level": "500",
            "year": year,
            "month": ["01", "02", "03", "12"],
            "day": list(range(1, 31 + 1)),
            "time": ["00:00", "12:00"],
            "area": [90, -180, 0, 180],
        },
        "download_{}.nc".format(year),
    )

# Collapse the downloaded files into a single file
cubes = iris.load("download_*.nc")
newcubes = iris.cube.CubeList()
for variable in ["eastward_wind", "northward_wind"]:
    cubes_by_year = cubes.extract(variable)
    for cube in cubes_by_year:
        del cube.attributes["history"]

    cube = cubes_by_year.concatenate_cube()
    newcubes.append(cube)
iris.save(newcubes, "era5_snapshot-u-v-500hpa_0N-90N_DJFM_1940-2022.nc")

# Save a second file with data on the coarsest grid
newcubes_coarse = iris.cube.CubeList()
for cube in newcubes:
    newcubes_coarse.append(regrid_to_coarsest(cube))
iris.save(
    newcubes_coarse, "era5-CanESM5-grid_snapshot-u-v-500hpa_0N-90N_DJFM_1940-2022.nc"
)
