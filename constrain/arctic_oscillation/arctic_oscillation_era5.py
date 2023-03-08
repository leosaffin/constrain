import iris
from iris.analysis import MEAN
from iris.coord_categorisation import add_month
import iris.quickplot as qplt
import matplotlib.pyplot as plt

from irise import plot

from constrain import arctic_oscillation


def main():
    mslp = iris.load_cube("era5_monthly_mean_CanESM5_grid_mslp_20N-90N_NDJFMA.nc")

    mslp_anomaly = as_anomaly_by_month(mslp)

    eof_pattern, ao_index = arctic_oscillation.from_eofs(mslp_anomaly)

    iris.save(eof_pattern, "AO_pattern_TW2000_ERA5.nc")
    iris.save(ao_index, "AO_index_TW2000_ERA5.nc")

    plot_ao(eof_pattern, ao_index)


def as_anomaly_by_month(cube):
    # Calculate mean for each month individually
    add_month(cube, "time")
    cube_mean_by_month = cube.aggregated_by("month", MEAN)

    cube_anomaly = iris.cube.CubeList()
    for subcube in cube.slices_over("time"):
        month = subcube.coord("month").points
        cs = iris.Constraint(month=month)

        cube_anomaly.append(subcube - cube_mean_by_month.extract(cs).data)

    return cube_anomaly.merge_cube()


def plot_ao(eof_pattern, ao_index):
    plot.pcolormesh(eof_pattern, vmin=-1, vmax=1, cmap="seismic", projection="NorthPolarStereo")
    plt.savefig("AO_pattern_TW2000_ERA5.png")

    plt.figure()
    qplt.plot(ao_index)
    plt.savefig("AO_index_TW2000_ERA5.png")

    plt.show()


if __name__ == '__main__':
    main()
