"""
Calculate the eddy-feedback parameter from downloaded ERA5 data following
Hardiman et al. (2022)

Usage:
  efp_era5 <filenames>... [--output_path=<str>]

Options:
    -h --help
        Show this screen.
    <filenames>
        The files containing velocity data
    --output_path=<str>
        Where to save the results [default: ./]
"""
import datetime
from dateutil.parser import parse as dateparse
from dateutil.relativedelta import relativedelta

from docopt import docopt
import iris
from iris.coord_categorisation import add_month

from constrain import eddy_feedback_parameter


def main(
    filenames,
    output_path="./",
    months=("Dec", "Jan", "Feb"),
):
    ua = iris.load_cube(filenames, "eastward_wind")
    va = iris.load_cube(filenames, "northward_wind")

    # Extract requested months
    for cube in ua, va:
        add_month(cube, "time")

    months_cs = iris.Constraint(month=lambda x: x in months)
    ua = ua.extract(months_cs)
    va = va.extract(months_cs)

    # Calculate zonal acceleration due to horizontal EP flux divergence
    print("Calculating EP-flux divergence")
    div_f = eddy_feedback_parameter.horiz_EPflux_div(ua, va)

    # Calculate zonal mean zonal wind
    print("Calculating zonal-mean zonal wind")
    ua_zm = ua.collapsed("longitude", iris.analysis.MEAN)

    # Calculate seasonal means ensuring only complete seasons
    span = calculate_season_span(months)
    divf_season = eddy_feedback_parameter.seasonal_mean(div_f, span=span)
    ua_zm_season = eddy_feedback_parameter.seasonal_mean(ua_zm, span=span)

    months_str = "".join([m[0] for m in months])
    iris.save(
        divf_season, output_path + "era5_EP-flux-divergence_{}.nc".format(months_str)
    )
    iris.save(
        ua_zm_season,
        output_path + "era5_zonal-mean-zonal-wind_{}.nc".format(months_str),
    )

    # Calculate EFP. Correlation between EP flux divergence and zonal-mean zonal wind
    print("Calculating eddy-feedback parameter")
    efp = eddy_feedback_parameter.eddy_feedback_parameter(divf_season, ua_zm_season)

    print("Eddy feedback parameter = {:.2f}".format(efp.data))
    iris.save(
        efp, output_path + "era5_eddy-feedback-parameter-{}.nc".format(months_str)
    )

    # Calculate EFP for 1993-2016 to compare with Hardiman et al. (2022)
    cs = iris.Constraint(season_year=lambda x: 1993 <= x < 2016)
    divf_sub = divf_season.extract(cs)
    ua_zm_sub = ua_zm_season.extract(cs)

    efp_sub = eddy_feedback_parameter.eddy_feedback_parameter(divf_sub, ua_zm_sub)
    print("Eddy feedback parameter (1993-2016) = {:.2f}".format(efp_sub.data))
    iris.save(
        efp_sub,
        output_path + "era5_eddy-feedback-parameter-{}-1993-2016.nc".format(months_str),
    )


def calculate_season_span(months):
    start = dateparse(months[0] + "01")
    end = dateparse(months[-1] + " 01") + relativedelta(months=1, days=-1)
    dt = end - start

    if dt < datetime.timedelta(0):
        dt = end - (start - relativedelta(years=1))

    return dt


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    args = docopt(__doc__)
    print(args)
    main(args["<filenames>"], output_path=args["--output_path"])
