import pathlib

import iris
from iris.analysis import AreaWeighted

here = pathlib.Path(__file__).resolve().parent
coarse_grid = iris.load_cube(here / "grid.nc")


def regrid_to_coarsest(cube):
    boundaries = dict()
    for axis in ["x", "y"]:
        coord = cube.coord(axis=axis, dim_coords=True)
        if not coord.has_bounds():
            coord.guess_bounds()

        boundaries[coord.name()] = (coord.bounds.min(), coord.bounds.max())

    coarse_cube = cube.regrid(coarse_grid, AreaWeighted())

    # Regridding extrapolates to full globe so reselect area
    coarse_cube = coarse_cube.intersection(**boundaries)

    return coarse_cube
