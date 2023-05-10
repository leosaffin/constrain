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
        add_season_year(cube, "time")

    # Calculate by year to not use too much memory
    # Ignore first and last year as they are not full seasons
    lefp_all_years = iris.cube.CubeList()
    season_years = list(set(ua.coord("season_year").points))[1:-1]
    for year in season_years:
        print(year)
        cs = iris.Constraint(season_year=year)
        ua_s = ua.extract(cs)
        va_s = va.extract(cs)

        # Calculate E-vectors
        print('Calculating E-vectors')
        e_lon, e_lat = eddy_feedback_parameter.Evectors(
            ua_s, va_s, months=months, window=61, f1=2, f2=6, definition=definition,
        )

        # Calculate background deformation flow, D
        print('Calculating background deformation flow')
        d_lon, d_lat = eddy_feedback_parameter.background_deformation_flow(
            ua_s, va_s, months=months, window=61, f=10, definition=definition,
        )

        # Calculate barotropic energy generation rate, G=E.D, and
        # correct metadata
        print('Calculating G = E.D')
        lefp = e_lon * d_lon + e_lat * d_lat
        lefp.rename('barotropic_energy_generation_rate')
        lefp.units = 'm2 s-3'

        # Calculate G climatology
        lefp_b = lefp.collapsed("season_year", iris.analysis.MEAN)
        lefp_all_years.append(lefp_b)

    # Save merged eddy feedback parameter 2d maps
    lefp_all_years = lefp_all_years.merge_cube()
    iris.save(
        lefp_all_years, output_path + "G_mean_{}_lat_lon_DJF_ERA5.nc".format(definition)
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
