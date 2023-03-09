import pathlib

import iris
from iris.analysis import AreaWeighted

here = pathlib.Path(__file__).resolve().parent
coarse_grid = iris.load_cube(here / "grid.nc")


def regrid_to_coarsest(cube):
    return cube.regrid(coarse_grid, AreaWeighted())
