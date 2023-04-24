import pathlib

import numpy as np
import iris
from iris.coords import DimCoord
from iris.cube import Cube
from iris.analysis import AreaWeighted

here = pathlib.Path(__file__).resolve().parent
coarse_grid = iris.load_cube(here / "grid.nc")


def regrid_to_coarsest(cube):
    """Regrid to the coarsest grid in the CMIP6 models (CanESM5)

    Args:
        cube (iris.cube.Cube): The cube to be regridded

    Returns:
        iris.cube.Cube: The regridded cube
    """
    # Only regrid to the region in the input (don't extrapolate to the full globe)
    coarse_grid_subset = _subset_cube(coarse_grid, cube)

    return cube.regrid(coarse_grid_subset, AreaWeighted())


def regrid_to_degrees(cube, spacing):
    """Regrid to a longitude-latitude grid with the specified spacing

    Args:
        cube (iris.cube.Cube): The cube to be regridded
        spacing (scalar): The grid spacing in degress

    Returns:
        iris.cube.Cube: The regridded cube
    """
    new_grid = _n_degree_grid(spacing)

    new_grid_subset = _subset_cube(new_grid, cube)

    return cube.regrid(new_grid_subset, AreaWeighted())


def _subset_cube(cube, target):
    boundaries = dict()
    for axis in ["x", "y"]:
        coord = target.coord(axis=axis, dim_coords=True)
        if not coord.has_bounds():
            coord.guess_bounds()

        boundaries[coord.name()] = (coord.points.min(), coord.points.max())

    return cube.intersection(**boundaries)


def _n_degree_grid(spacing):
    longitude = DimCoord(
        points=np.arange(0, 360, spacing),
        standard_name="longitude",
        units="degrees",
        circular=True,
    )
    longitude.guess_bounds()

    latitude = DimCoord(
        points=np.arange(-90, 90, spacing),
        standard_name="latitude",
        units="degrees",
    )
    latitude.guess_bounds()

    cube = Cube(
        data=np.zeros([len(latitude.points), len(longitude.points)]),
        long_name="nothing",
        dim_coords_and_dims=[(latitude, 0), (longitude, 1)]
    )

    return cube
