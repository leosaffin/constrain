import pytest
import numpy as np
from iris.analysis import AreaWeighted
from iris.cube import Cube
from iris.coords import DimCoord

import constrain


def test_regrid_to_coarsest(cube_mslp_single_time):
    cube_mslp_coarse = constrain.regrid_to_coarsest(cube_mslp_single_time)

    lons = cube_mslp_coarse.coord("longitude").points
    lats = cube_mslp_coarse.coord("latitude").points

    assert (lons + 180 == constrain.coarse_grid.coord("longitude").points).all()
    assert (lats == constrain.coarse_grid.coord("latitude").points[-len(lats):]).all()


@pytest.mark.parametrize("allow_partial_areas", [True, False])
def test_regrid_to_coarsest_partial_areas(cube_mslp, allow_partial_areas):
    cube_mslp_coarse = constrain.regrid_to_coarsest(
        cube_mslp, allow_partial_areas=allow_partial_areas
    )

    # allow_partial_areas=True should result in no masked regions and
    # allow_partial_areas=False should result in some masked regions
    assert cube_mslp_coarse.data.mask.any() != allow_partial_areas
    assert not cube_mslp_coarse.data.mask.all()


def test_regrid_to_coarsest_from_coarsest():
    cube_coarse = constrain.coarse_grid.intersection(latitude=(0, 90), ignore_bounds=True)
    cube_coarse_2 = constrain.regrid_to_coarsest(cube_coarse)

    assert (cube_coarse.data == cube_coarse_2.data).all()
    for coord in ["longitude", "latitude"]:
        assert (cube_coarse.coord(coord).points ==
                cube_coarse_2.coord(coord).points).all()


@pytest.mark.parametrize("spacing", [1, 2, 3])
def test_regrid_to_degrees(cube_mslp, spacing):
    coarse_cube = constrain.regrid_to_degrees(cube_mslp, spacing)

    for coord in ["longitude", "latitude"]:
        points = coarse_cube.coord(coord).points
        assert (points[1:] - points[:-1]) == pytest.approx(spacing)


def test_area_weighted_regrid_masks():
    lon = DimCoord(
        points=np.arange(0, 90, 1),
        standard_name="longitude",
    )

    lat = DimCoord(
        points=np.arange(0, 90, 1),
        standard_name="latitude",
    )

    lon.guess_bounds()
    lat.guess_bounds()

    cube = Cube(
        data=np.zeros([len(lat.points), len(lon.points)]),
        long_name="blank",
        dim_coords_and_dims=[(lat, 0), (lon, 1)],
    )

    lon_hi_res = DimCoord(
        points=np.arange(0, 90, 0.1),
        standard_name="longitude",
    )

    lat_hi_res = DimCoord(
        points=np.arange(0, 90, 0.1),
        standard_name="latitude",
    )

    lon_hi_res.guess_bounds()
    lat_hi_res.guess_bounds()

    cube_hi_res = Cube(
        data=np.zeros([len(lat_hi_res.points), len(lon_hi_res.points)]),
        long_name="blank",
        dim_coords_and_dims=[(lat_hi_res, 0), (lon_hi_res, 1)],
    )

    cube_regridded = cube_hi_res.regrid(cube, AreaWeighted())

    assert cube_regridded.data.mask[0, :].all()
    assert cube_regridded.data.mask[:, 0].all()

