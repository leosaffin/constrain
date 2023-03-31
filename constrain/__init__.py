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

        boundaries[coord.name()] = (coord.points.min(), coord.points.max())

    # Only regrid to the region in the input (don't extrapolate to the full globe)
    coarse_grid_subset = coarse_grid.intersection(**boundaries)

    coarse_cube = cube.regrid(coarse_grid_subset, AreaWeighted())

    return coarse_cube
