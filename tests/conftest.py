import pytest
from iris.tests import stock

from iris.cube import CubeList


def _cube3d_lat_lon():
    cube = stock.realistic_3d()
    cube.coord("grid_longitude").rename("longitude")
    cube.coord("grid_latitude").rename("latitude")

    return cube


@pytest.fixture(scope="session")
def cube3d_lat_lon():
    yield _cube3d_lat_lon()


@pytest.fixture(scope="session")
def cube4d_lat_lon():
    cubes = CubeList()
    for p in range(500, 1000+1, 50):
        cube = _cube3d_lat_lon()
        cube.coord("air_pressure").points = p
        cubes.append(cube)

    return cubes.merge_cube()
