from iris.analysis import MEAN
from iris.coord_categorisation import add_season_year
from iris.util import squeeze
from eofs.iris import Eof


def from_eofs(mslp_anomaly):
    """

    Calculate the Arctic Oscillation index and pattern using the EOF definition from
    Thompson and Wallace (2000) (https://doi.org/10.1029/98GL00950): the "leading
    principal component of the wintertime (November-April) monthly mean
    sea-level-pressure anomaly field over the domain poleward of 20N"

    Also suggested is to remove the linear trend from each gridpoint before performing
    the EOF analysis

    Args:
        mslp_anomaly (iris.cube.Cube):

    Returns:

    """
    solver = Eof(mslp_anomaly, weights="coslat")

    eof_pattern = squeeze(solver.eofsAsCorrelation(neofs=1))
    eof_pattern.rename("arctic_oscillation_eof1")

    pc1 = squeeze(solver.pcs(pcscaling=1, npcs=1))

    add_season_year(pc1, "time", seasons=["ndjfma", "mjjaso"])
    ao_index = pc1.aggregated_by("season_year", MEAN)
    ao_index.rename("arctic_oscillation_index")

    return eof_pattern, ao_index
