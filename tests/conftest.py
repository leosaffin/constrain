import pathlib

import iris
import pytest

here = pathlib.Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def cube_mslp():
    cube = iris.load_cube(here / "testdata/era5_monthly_mean_mslp_20N-90N_NDJFMA.nc")
    cube.coord("longitude").circular = True
    yield cube


@pytest.fixture(scope="session")
def cube_mslp_single_time(cube_mslp):
    yield cube_mslp[0]
