import pytest

from numpy.testing import assert_allclose

from constrain.eddy_feedback_parameter import horiz_EPflux_div


@pytest.mark.parametrize("cube", ["cube3d_lat_lon", "cube4d_lat_lon"])
def test_horiz_EPflux_div_4d(request, cube):
    u = request.getfixturevalue(cube)
    v = request.getfixturevalue(cube)

    divf_h = horiz_EPflux_div(u, v)

    results = [
        -2.1971021947636834e-07, -1.6457757021158595e-07, -1.0966263923971633e-07,
        -5.4814613652224744e-08, -0.0, 5.4814613652224744e-08, 1.0966263923971633e-07,
        1.6457757021158595e-07, 2.1971021947636834e-07
    ]

    for subcube in divf_h.slices_over(["time", "air_pressure"]):
        assert_allclose(subcube.data, results)

