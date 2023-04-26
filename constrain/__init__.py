import pathlib

import numpy as np
import iris
from iris.coords import DimCoord
from iris.cube import Cube
from iris.analysis import AreaWeighted

here = pathlib.Path(__file__).resolve().parent
coarse_grid = iris.load_cube(here / "grid.nc")


def regrid_to_coarsest(cube, allow_partial_areas=True):
    """Regrid to the coarsest grid in the CMIP6 models (CanESM5)

    Args:
        cube (iris.cube.Cube): The cube to be regridded
        allow_partial_areas (bool):

    Returns:
        iris.cube.Cube: The regridded cube
    """
    # Only regrid to the region in the input (don't extrapolate to the full globe)
    coarse_grid_subset = _subset_cube(coarse_grid, cube)

    if allow_partial_areas:
        _match_bounds(cube, coarse_grid_subset)

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


def _match_bounds(cube_to_regrid, coarse_cube):
    # Match the edges of the bounds in the coarse cube to the bounds in the cube to be
    # regridded so an area-weighted regridding still works when the higher-resolution
    # cube only partially fills the outer grid boxes
    # e.g. I have two cubes with latitudes from 0 to 90. One is a 2-degree grid and the
    # other is a 1-degree grid. The bounds at 0 will be -0.5, 0.5 for the one-degree
    # grid and -1, 1 for the 2-degree grid. If I want to regrid the 1-degree grid to the
    # 2-degree grid, I will get masked data at 0 because it is missing data for the
    # region from -1,-0.5. A way around this is to shorten the bounds in the 2-degree
    # grid
    for axis in ("x", "y"):
        coord_fine = cube_to_regrid.coord(axis=axis, dim_coords=True)

        if not coord_fine.circular:
            coord_coarse = coarse_cube.coord(axis=axis, dim_coords=True)

            # Can't write directly to the bounds array so create a copy and later
            # overwrite the bounds attribute with the copied and modified array
            new_bounds = coord_coarse.bounds.copy()
            bounds_to_match = coord_fine.bounds

            # Flip bounds if the directions are different
            flipped = (new_bounds[0, 0] < new_bounds[-1, 0]) != (bounds_to_match[0, 0] < bounds_to_match[-1, 0])
            if flipped:
                new_bounds = new_bounds[::-1, ::-1]

            # Match the edges of the bounds
            new_bounds[0, 0] = bounds_to_match[0, 0]
            new_bounds[-1, -1] = bounds_to_match[-1, -1]

            # Flip the bounds back
            if flipped:
                new_bounds = new_bounds[::-1, ::-1]

            coord_coarse.bounds = new_bounds


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
