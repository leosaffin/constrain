from iris.analysis import MEAN
from iris.analysis.cartography import area_weights


def from_boxes(mslp, south_box=(-90, 60, 20, 55), north_box=(-90, 60, 55, 90)):
    """NAO index calculated as the difference in mean-sea-level pressure between two boxes

    Following Stephenson et al. (2006) the NAO index is calculated as the difference in
    mean-sea-level pressure between a Southern and a Northern box covering the North
    Atlantic.

    Args:
        mslp:
        south_box (iterable, optional): The coordinates of the Southern box to average
            over, in the order West, East, South, North. Default is (-90, 60, 20, 55)
        north_box (iterable, optional): The coordinates of the Northern box to average
            over, in the order West, East, South, North. Default is (-90, 60, 55, 90)

    Returns:
        iris.cube.Cube: NAO index. Same properties as the input mslp, but with the
            longitude/latitude dimensions collapsed
    """
    mslp_south = box_average(mslp, south_box)
    mslp_north = box_average(mslp, north_box)

    nao = mslp_south - mslp_north

    nao.convert_units("hPa")
    nao.rename("nao_index")

    return nao


def box_average(cube, box):
    xc = cube.coord(axis="x", dim_coords=True).name()
    yc = cube.coord(axis="y", dim_coords=True).name()

    cube_in_box = cube.intersection(
        **{xc: (box[0], box[1]), yc: (box[2], box[3])}, ignore_bounds=True
    )

    weights = area_weights(cube_in_box)

    return cube_in_box.collapsed([xc, yc], MEAN, weights=weights)
