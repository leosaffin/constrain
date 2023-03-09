from math import ceil

import iris
from iris.coord_categorisation import add_month
import iris.quickplot as qplt
import matplotlib.pyplot as plt

from constrain import north_atlantic_oscillation


def main():
    mslp = iris.load_cube("era5_monthly_mean_CanESM5_grid_mslp_20N-90N_NDJFMA.nc")

    add_month(mslp, "time")
    mslp = mslp.extract(iris.Constraint(month=lambda x: str(x).lower() in ["dec", "jan", "feb", "mar"]))

    nao_index = north_atlantic_oscillation.from_boxes(mslp)

    iris.save(nao_index, "NAOI_monthly_ERA5.nc")

    fig, axes = plot_nao(nao_index)
    fig.savefig("NAOI_monthly_S2006_ERA5.png")
    plt.show()


def plot_nao(nao_index):
    months = ["Dec", "Jan", "Feb", "Mar"]

    nrows = ceil(len(months) / 2)

    fig, axes = plt.subplots(nrows, 2, figsize=(8, nrows * 5), sharex="all", sharey="all")
    for n, month in enumerate(months):
        nao_month = nao_index.extract(iris.Constraint(month=month))

        ax = plt.axes(axes[n // 2, n % 2])
        qplt.plot(nao_month)
        ax.set_title(month)

    fig.autofmt_xdate()

    return fig, axes


if __name__ == '__main__':
    main()
