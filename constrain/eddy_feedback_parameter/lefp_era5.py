"""
Calculate local-eddy feedback parameter for ERA5 data

Usage:
  lefp_era5 <filenames>... [--definition=<str>] [--output_path=<str>]

Options:
    -h --help
        Show this screen.
    <filenames>
        The files containing velocity data
    --definition=<str>
        Definition of the barotropic energy generation rate to use [default: FY02]
    --output_path=<str>
        Where to save the results [default: ./]
"""
from docopt import docopt
import iris
from iris.coord_categorisation import add_month, add_season_year

from constrain import eddy_feedback_parameter


def main(
    filenames,
    output_path="./",
    months=("Dec", "Jan", "Feb"),
    definition="FY02",
):
    ua = iris.load_cube(filenames, "eastward_wind")
    va = iris.load_cube(filenames, "northward_wind")

    for cube in ua, va:
        add_month(cube, "time")
        add_season_year(cube, "time", seasons=("ndjfma", "mjjaso"))

    lefp_all_levels = iris.cube.CubeList()
    pressure_levels = ua.coord("pressure_level").points
    for pressure_level in pressure_levels:
        print(pressure_level)
        cs = iris.Constraint(pressure_level=pressure_level)
        ua_s = ua.extract(cs)
        va_s = va.extract(cs)

        lefp = eddy_feedback_parameter.local_eddy_feedback(ua_s, va_s, months=months)
        lefp_all_levels.append(lefp)

    # Save merged eddy feedback parameter 2d maps
    lefp_all_levels = lefp_all_levels.merge()
    lefp_all_levels = lefp_all_levels.concatenate_cube()
    iris.save(
        lefp_all_levels, output_path + "G_mean_{}_lat_lon_DJF_ERA5.nc".format(definition)
    )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    args = docopt(__doc__)
    print(args)
    main(
        args["<filenames>"],
        output_path=args["--output_path"],
        definition=args["--definition"],
    )
