"""
Generate a CSV file of sudden stratospheric warmings in ERA5 data with events
classified as "split" or "displaced" (or "unclassified)

This script has been written by modifying the code in
https://github.com/BrisClimate/Persistent_model_biases_in_the_CMIP6_representation_of_stratospheric_polar_vortex_variability
which in turn uses code from https://github.com/wseviour/vortex-moments/ for classifying
events

Usage:
  ssws_era5 <u_filename> <z_filename> [--output_file=<str>]

Options:
    -h --help
        Show this screen.
    <u_filename>
        File(s) containing zonal mean zonal wind data as a function of time
    <z_filename>
        File(s) containing geopotential height data as a function of time, latitude, and
        longitude
    --output_file=<str>
        Where to save the results [default: era5_ssws.csv]

"""

import iris
from iris.coord_categorisation import add_season_year
import pandas

from constrain.sudden_stratospheric_warming import find_ssws, classify_ssws


def main(u_filename, z_filename, output_file="era5_ssws.csv"):
    # Load ERA5 data for all years
    zonal_wind = iris.load_cube(u_filename)
    geopotential_height = iris.load_cube(z_filename)

    for cube in [zonal_wind, geopotential_height]:
        if "season_year" not in [c.name() for c in cube.coords()]:
            add_season_year(cube, "time", seasons=["ndjfma", "mjjaso"])

    years = set(zonal_wind.coord("season_year").points)

    # Find SSWs for each winter season
    ssws = []
    for year in years:
        print(year)
        cs = iris.Constraint(season_year=year)
        zonal_wind_season = zonal_wind.extract(cs)
        gph_season = geopotential_height.extract(cs)

        # Find SSW onset dates
        ssws_season = find_ssws(zonal_wind_season)

        # Classify SSW events
        classify_ssws(ssws_season, gph_season)

        ssws.append(ssws_season)

    ssws = pandas.concat(ssws)
    print(ssws)
    ssws.to_csv(output_file)


if __name__ == '__main__':
    import warnings
    from docopt import docopt

    warnings.filterwarnings("ignore")

    args = docopt(__doc__)
    print(args)
    main(args["<u_filename>"], args["<z_filename>"], output_file=args["--output_file"])
