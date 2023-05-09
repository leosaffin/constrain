import pytest
import numpy as np
from iris.analysis import AreaWeighted

import constrain


def test_regrid_to_coarsest():
    cube = constrain._n_degree_grid(1.0)
    cube_coarse = constrain.regrid_to_coarsest(cube)

    lons = cube_coarse.coord("longitude").points
    lats = cube_coarse.coord("latitude").points

    assert (lons == constrain.coarse_grid.coord("longitude").points).all()
    assert (lats == constrain.coarse_grid.coord("latitude").points).all()


@pytest.mark.parametrize("allow_partial_areas", [True, False])
def test_regrid_to_coarsest_partial_areas(allow_partial_areas):
    cube = constrain._n_degree_grid(1.0).intersection(
        latitude=(1, 90), ignore_bounds=True
    )

    # Give the array a mask so testing the value of masked elements later does not fail
    # in the case of zero masked data
    cube.data = np.ma.masked_where(np.zeros_like(cube.data), cube.data)

    cube_coarse = constrain.regrid_to_coarsest(
        cube, allow_partial_areas=allow_partial_areas
    )

    # allow_partial_areas=True should result in no masked regions and
    # allow_partial_areas=False should result in some masked regions
    assert cube_coarse.data.mask.any() != allow_partial_areas
    assert not cube_coarse.data.mask.all()


def test_regrid_to_coarsest_from_coarsest():
    cube_coarse = constrain.coarse_grid.intersection(latitude=(0, 90), ignore_bounds=True)
    cube_coarse_2 = constrain.regrid_to_coarsest(cube_coarse)

    assert (cube_coarse.data == cube_coarse_2.data).all()
    for coord in ["longitude", "latitude"]:
        assert (cube_coarse.coord(coord).points ==
                cube_coarse_2.coord(coord).points).all()


@pytest.mark.parametrize("spacing", [1, 2, 3])
def test_regrid_to_degrees(spacing):
    cube = constrain._n_degree_grid(0.5)
    coarse_cube = constrain.regrid_to_degrees(cube, spacing)

    for coord in ["longitude", "latitude"]:
        points = coarse_cube.coord(coord).points
        assert (points[1:] - points[:-1]) == pytest.approx(spacing)


def test_area_weighted_regrid_masks():
    cube = constrain._n_degree_grid(1.0).intersection(
        longitude=(0.5, 89.5),
        latitude=(0.5, 89.5),
        ignore_bounds=True,
    )
    cube_hi_res = constrain._n_degree_grid(0.1).intersection(
        longitude=(0.5, 89.5),
        latitude=(0.5, 89.5),
        ignore_bounds=True,
    )

    cube_regridded = cube_hi_res.regrid(cube, AreaWeighted())

    print(cube_regridded.data.mask)
    assert cube_regridded.data.mask[0, :].all()
    assert cube_regridded.data.mask[-1, :].all()
    assert cube_regridded.data.mask[:, 0].all()
    assert cube_regridded.data.mask[:, -1].all()
