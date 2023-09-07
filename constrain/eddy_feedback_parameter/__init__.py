from __future__ import division
import numpy as np
import iris
import iris.analysis.cartography
from iris.analysis import Linear
from iris.analysis.calculus import differentiate
from iris.analysis.stats import pearsonr

from constrain.filters import lowpass_lanczos


def horiz_EPflux_div(U, V):
    """
    Created on Thurs Nov 17 13:48 2022

    @author: Christine McKenna

    =======================================================================
    Purpose: Calculates the zonal acceleration due to the QG component of
             the horizontal EP-flux divergence at a single pressure
             level, as defined in:

             Hardimann et al. 2022
             https://doi.org/10.1038/s41612-022-00280-4

             This code requires input in the form of iris cubes.

    =======================================================================

    Category: Diagnostics

    Input: U(..., lat, lon) - iris cube of zonal wind
           V(..., lat, lon) - iris cube of meridional wind

    Output: divF_h(..., lat) - iris cube of horizontal EP flux
                              divergence
    """
    # Define radius of Earth in metres
    a = iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS

    # ---------------------------------------------------------------
    # Calculate momentum flux, UsVs_z
    # (s = zonal deviation, z = zonal mean)
    # ---------------------------------------------------------------
    # NB: use fact that UsVs_z = UV_z - UzVz

    # Calculate zonal mean of individual fields
    Uz = U.collapsed(['longitude'], iris.analysis.MEAN)
    Vz = V.collapsed(['longitude'], iris.analysis.MEAN)

    # Calculate zonal mean of products of fields
    UV_z = (U * V).collapsed(['longitude'], iris.analysis.MEAN)

    # Calculate momentum flux
    UsVs_z = UV_z - Uz * Vz

    # Convert latitudes to radians and calculate cos(latitude)
    lats = UsVs_z.coord('latitude').copy()
    lats.convert_units('radians')
    lats = lats.points
    cos2lat = np.cos(lats) ** 2

    # Match cos2lat shape to cube
    coords = [c.name() for c in UsVs_z.dim_coords]
    idx_lat = coords.index("latitude")
    idx = [np.newaxis] * len(UsVs_z.shape)
    idx[idx_lat] = Ellipsis
    cos2lat = cos2lat[tuple(idx)]

    # -----------------------------------------------------------------
    # Calculate zonal accel due to divergence of horiz EP flux, Flat
    # -----------------------------------------------------------------
    # Calculation is -(1/a/cos2lat) * d(UsVs_z*cos2lat)/dlat

    # Calculate UsVs_z*cos2lat (Flat*coslat)
    Flatcoslat = UsVs_z * cos2lat

    # Calculate d(Flat*coslat)/dlat
    dFlatcoslat_dlat = differentiate_horizontal(Flatcoslat, "latitude")

    # Calculate horizontal EP flux divergence
    divF_h = dFlatcoslat_dlat * (-1 / a / cos2lat)

    # Update metadata following calculations
    divF_h.rename('tendency_of_eastward_wind_due_to_horizontal_eliassen_' + \
                  'palm_flux_divergence')
    divF_h.units = 'm s-2'

    return divF_h


def seasonal_mean(cube, span):
    """
    Created on Wed Nov 23 10:27 2022

    @author: Christine McKenna

    =======================================================================
    Purpose: Calculates the seasonal mean of a cube(day, ...) in each year.

             The argument 'span' specifies the minimum time-period to be
             included in each seasonal mean, in units of days.

             The input cube should already be subset into the desired
             season.

    =======================================================================

    Category: Diagnostics

    Input: cube(day, ...) - iris cube of any variable, already subset
                            into the required season
           span - specifies the minimum time-period to be included in
                  the seasonal means in days

    Output: cube_b(year, ...) - sesonal mean of cube in each year

    """

    # Add required time coordinates
    if "season_year" not in [c.name() for c in cube.coords()]:
        iris.coord_categorisation.add_season_year(cube, "time")

    # Calculate seasonal mean
    cube_b = cube.aggregated_by("season_year", iris.analysis.MEAN)
    cube_b = cube_b.extract(iris.Constraint(time=lambda t: (t.bound[1] - \
                                                            t.bound[0]) >= span))

    # Return seasonal mean
    return cube_b


def eddy_feedback_parameter(divF_h, Uz):
    """
    Created on Wed Nov 23 10:10 2022

    @author: Christine McKenna

    =======================================================================
    Purpose: Calculates the eddy feedback parameter as defined in:

             Hardimann et al. 2022
             https://doi.org/10.1038/s41612-022-00280-4

             This code requires input in the form of iris cubes.

    =======================================================================

    Category: Diagnostics

    Input: divF_h(year,lat) - iris cube of seasonal mean zonal acceleration
                              due to the QG component of the horizontal EP
                              flux divergence
           Uz(year,lat) - iris cube of seasonal mean zonal mean zonal wind

    Output: EFP(scalar) - iris cube of the eddy feedback parameter, or
                          the area-weighted average over 25N-72N of the
                          local correlation squared between divF_h and Uz

    """

    # Correlate zonal mean zonal wind with horizontal EP flux divergence
    r = pearsonr(divF_h, Uz, corr_coords="season_year")

    # Calculate percentage variance explained
    r2 = r ** 2
    r2.rename("Pearson's r squared")

    # Average r2 over latitudes 25N-72N to get eddy feedback parameter
    r2_box = r2.intersection(latitude=(25, 72), ignore_bounds=True)
    weights = iris.analysis.cartography.cosine_latitude_weights(r2_box)
    EFP = r2_box.collapsed('latitude', iris.analysis.MEAN, weights=weights)

    # Return eddy feedback parameter
    return EFP


def local_eddy_feedback(ua, va, months=("Dec", "Jan", "Feb"), f1=2, f2=6, f3=10, window=61, definition="MC89"):
    """Calculates the eddy feedback G=E.D as defined in Mak and Cai 1989

    https://doi.org/10.1175/1520-0469(1989)046<3289:LBI>2.0.CO;2

    G=E.D is the barotropic energy generation rate. It diagnoses the kinetic energy
    exchange between transient eddies (described by E) and the background flow
    (described by D). Positive G means eddies grow and extract energy from the
    background flow, thereby decelerating it; negative G means eddies decay and lose
    energy to the background flow, thereby accelarating it.

    The equation for G is given in Cartesian coordinates in Mak and Cai 1989, but we use
    the version in spherical geometry. For details see the functions used in this
    script.

    Args:
        ua (iris.cube.Cube): Zonal wind
        va (iris.cube.Cube): Meridional wind
        months: The months to output the local eddy feedback over
            (default is ("Dec", "Jan", "Feb"))
        f1 (int): Short timescale limit for eddies when filtering with Lanczos filter.
            Default is 2
        f2 (int): Long timescale limit for eddies when filtering with Lanczos filter.
            Default is 6
        f3 (int): Short timescale limit for low-pass winds. Default is 10
        window (int): The window to calculate Lanczos filters over. Note that the input
            ua and va must have enough padding to calculate the filters over. For
            example, using the default parameters with daily data means the winds need
            an extra month (30 days) at either end. So NDJFM for outputting DJF eddy
            feedback. Default is 61
        definition (str): Either "MC89" to us the Mak and Cai (1989) definition
            (default), or "FY02" to use the Fukutomi and Yasunari (2002) definition

    Returns:
        iris.cube.Cube
            The local eddy feedback on the same grid as the input winds
    """
    # Calculate by year to not use too much memory
    # Ignore first and last year as they are not full seasons
    lefp_all_years = iris.cube.CubeList()
    season_years = list(set(ua.coord("season_year").points))[1:-1]
    for year in season_years:
        cs = iris.Constraint(season_year=year)
        ua_s = ua.extract(cs)
        va_s = va.extract(cs)

        # Calculate E-vectors
        e_lon, e_lat = Evectors(
            ua_s, va_s, months=months, window=window, f1=f1, f2=f2, definition=definition
        )

        # Calculate background deformation flow, D
        d_lon, d_lat = background_deformation_flow(
            ua_s, va_s, months=months, window=window, f=f3, definition=definition
        )

        # Calculate barotropic energy generation rate, G=E.D, and
        # correct metadata
        lefp = e_lon * d_lon + e_lat * d_lat
        lefp.rename('barotropic_energy_generation_rate')
        lefp.units = 'm2 s-3'

        lefp_all_years.append(lefp)

    return lefp_all_years.concatenate_cube()


def Evectors(u, v, months, window, f1, f2, definition="MC89"):
    """
    Created on Thurs Feb 02 18:35 2023

    @author: Christine McKenna

    =====================================================================
    Purpose: Calculates the horizontal E-vectors for a 2D flow (u,v)
             for the specified months and timescale of variability
             (defined through window, f1, f2).

             The E-vectors are defined in Mak and Cai 1989, where the
             x-component differs from the Hoskins et al. 1983 E-vector
             by a factor of a half:

             doi:10.1175/1520-0469(1989)046<3289:LBI>2.0.CO;2

             Alternatively the E-vectors can be calculated using thespherical
             coordinate definition in Fukutomi and Yasunari 2002, or
             Kawamura and Murakami 1995, where the x-component also differs from
             the Hoskins et al. 1983 E-vector by a factor of a half:

             https://doi.org/10.2151/jmsj.80.311
             https://doi.org/10.2151/jmsj1965.73.6_1087

             Before calculating the E-vectors, a Lanczos filter is
             applied to the data to extract synoptic scale variability
             only. The user must specify the timescale of variability
             to extract (f1-f2 days) and the window size for the filter.

    =====================================================================

    Category: Diagnostics

    Input: u(day,lat,lon) - iris cube of zonal wind. Timeseries must
                            contain months calculating E-vectors for,
                            plus an extra buffer that is at least equal
                            to the Lanczos filter window size minus 1.
                            For example, if calculating E-vectors for
                            DJFM using a filter window size of 61 days,
                            need to at least include NDJFMA in your
                            timeseries so there is a buffer of 60 days.
                            Limiting the timeseries length to just what
                            you need is helpful as it reduces the size
                            of the data that needs to be stored and the
                            computational time, which can be large when
                            dealing with daily data.
           v(day,lat,lon) - iris cube of meridional wind. Details are
                            same as for ua.
           months         - months to calculate E-vectors for in form
                            of, e.g., ['Dec','Jan','Feb','Mar']
           window         - window size used for Lanczos filter
           f1             - cut-in frequency of Lanczos filter in days
           f2             - cut-out frequency of Lanczos filter in days
           definition     - Either "MC89" to us the Mak and Cai (1989)
                            definition (default), or "FY02" to use the
                            Fukutomi and Yasunari (2002) definition

    Output: E_lon(day,lat,lon) - iris cube of E-vector in longitude
                                 direction
            E_lat(day,lat,lon) - iris cube of E-vector in latitude
                                 direction

    """
    if definition not in ("MC89", "FY02"):
        raise ValueError(
            "Can only use the E-Vector definitions of Mak and Cai (1989) ('MC89') or"
            "Fukutomi and Yasunari (2002) ('FY02')"
        )

    ### ---------------------------------------------------------------
    ### Sort cube coordinates and define some required parameters
    ### ---------------------------------------------------------------

    ## Convert latitudes to radians and calculate cos(latitude)
    if definition == "FY02":
        lats = u.coord('latitude').copy()
        lats.convert_units('radians')
        coslat = np.cos(lats.points)[np.newaxis, :, np.newaxis]

    ### ---------------------------------------------------------------
    ### Extract synoptic (f1-f2 day) component of flow and
    ### specified months
    ### ---------------------------------------------------------------

    ## Extract timescales > f1 days
    u_f1 = lowpass_lanczos(u, window, f1)
    v_f1 = lowpass_lanczos(v, window, f1)

    ## Extract timescales > f2 days
    u_f2 = lowpass_lanczos(u, window, f2)
    v_f2 = lowpass_lanczos(v, window, f2)

    ## Now get synoptic scale (f1-f2 days) ua/va
    u_hp = u_f1 - u_f2
    v_hp = v_f1 - v_f2

    ## Fix month coordinate in u_hp/v_hp since applying
    ## Lanczos filter mucks this coordinate up
    u_hp.remove_coord('month')
    v_hp.remove_coord('month')
    iris.coord_categorisation.add_month(u_hp, 'time')
    iris.coord_categorisation.add_month(v_hp, 'time')

    ## Extract specified months
    u_hp = u_hp.extract(iris.Constraint(month=months))
    v_hp = v_hp.extract(iris.Constraint(month=months))

    ### ---------------------------------------------------------------
    ### Calculate E-vectors
    ### ---------------------------------------------------------------
    if definition == "MC89":
        E_lon = 1 / 2 * (v_hp ** 2 - u_hp ** 2)
        E_lat = (-1) * u_hp * v_hp

    elif definition == "FY02":
        E_lon = 1 / 2 * (v_hp ** 2 - u_hp ** 2) * coslat
        E_lat = (-1) * u_hp * v_hp * coslat

    ## Updata metadata following calculations
    E_lon.rename('Evector_in_longitude_direction')
    E_lat.rename('Evector_in_latitude_direction')
    E_lon.units = 'm2 s-2'
    E_lat.units = 'm2 s-2'

    ## Return E-vectors
    return E_lon, E_lat


def background_deformation_flow(u, v, months, window, f, definition="MC89"):
    """
    Created on Thurs Feb 02 18:46 2023

    @author: Christine McKenna

    =======================================================================
    Purpose: Calculates the deformation components of a 2D background flow
             (u,v) for the specified months and timescale of variability
             (defined through window, f).

             The deformation components are defined in Cartesian
             coordinates in Mak and Cai 1989:

             doi:10.1175/1520-0469(1989)046<3289:LBI>2.0.CO;2

             This code calculates the deformation components in spherical
             geometry; see Deng and Mak (2006).

             Alternatively, the deformation components defined in spherical
             coordinates in Fukutomi and Yasunari 2002, based on
             Kawamura and Murakami 1995 can be used:

             https://doi.org/10.2151/jmsj.80.311
             https://doi.org/10.2151/jmsj1965.73.6_1087

             You can deduce the deformation components, D, from the
             equation for CK in Fukutomi and Yasunari (2002), since
             they also define the E-vectors and CK=E.D.

             Before calculating the deformation components, a lowpass
             Lanczos filter is applied to the data to extract the low
             frequency or background flow only. The user must specify
             the timescale of variability to extract (> f days) and the
             window size for the filter.

    =======================================================================

    Category: Diagnostics

    Input: u(day,lat,lon) - iris cube of zonal wind. Timeseries must
                            contain months calculating deformation
                            components for, plus an extra buffer that is
                            at least equal to the Lanczos filter window
                            size minus 1. For example, if calculating
                            for DJFM using a filter window size of 61 days,
                            need to at least include NDJFMA in your
                            timeseries so there is a buffer of 60 days.
                            Limiting the timeseries length to just what
                            you need is helpful as it reduces the size
                            of the data that needs to be stored and the
                            computational time, which can be large when
                            dealing with daily data.
           v(day,lat,lon) - iris cube of meridional wind. Details are
                            same as for ua.
           months         - months to calculate components for in form
                            of, e.g., ['Dec','Jan','Feb','Mar']
           window         - window size used for Lanczos filter
           f              - frequency of lowpass Lanczos filter in days
           definition     - Either "MC89" to us the Mak and Cai (1989)
                            definition (default), or "FY02" to use the
                            Fukutomi and Yasunari (2002) definition

    Output: D_lon(day,lat,lon) - iris cube of stretching deformation
                                 component
            D_lat(day,lat,lon) - iris cube of shearing deformation
                                 component

    """
    if definition not in ("MC89", "FY02"):
        raise ValueError(
            "Can only use the E-Vector definitions of Mak and Cai (1989) ('MC89') or"
            "Fukutomi and Yasunari (2002) ('FY02')"
        )

    ### ---------------------------------------------------------------
    ### Sort cube coordinates and define some required parameters
    ### ---------------------------------------------------------------

    ## Convert latitudes/longitudes to radians and
    ## calculate cos(latitude)
    lats = u.coord('latitude').copy()
    lats.convert_units('radians')
    coslat = np.cos(lats.points)[np.newaxis, :, np.newaxis]

    if definition == "MC89":
        tanlat = np.tan(lats.points)[np.newaxis, :, np.newaxis]

    ## Define radius of Earth in metres
    a = iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS

    ### ---------------------------------------------------------------
    ### Extract low frequency background (> f day) component of flow
    ### and specified months
    ### ---------------------------------------------------------------

    U = lowpass_lanczos(u, window, f)
    V = lowpass_lanczos(v, window, f)

    ## Fix month coordinate in U/V since applying
    ## Lanczos filter mucks this coordinate up
    U.remove_coord('month')
    V.remove_coord('month')
    iris.coord_categorisation.add_month(U, 'time')
    iris.coord_categorisation.add_month(V, 'time')

    ## Extract specified months
    U = U.extract(iris.Constraint(month=months))
    V = V.extract(iris.Constraint(month=months))

    ### ---------------------------------------------------------------
    ### Calculate deformation flow
    ### ---------------------------------------------------------------

    ## Calculate required derivatives
    dU_dlon = differentiate_horizontal(U, "longitude")
    dV_dlon = differentiate_horizontal(V, "latitude")

    if definition == "MC89":
        dU_dlat = differentiate_horizontal(U, "latitude")
        dV_dlat = differentiate_horizontal(V, "latitude")

    elif definition == "FY02":
        dVcoslat_dlat = differentiate_horizontal(V * coslat, "latitude")
        dUcoslat_dlat = differentiate_horizontal(U * coslat, "latitude")

    ## Calculate D_lon and D_lat
    if definition == "MC89":
        D_lon = dU_dlon / a / coslat - dV_dlat / a - V * tanlat / a
        D_lat = dV_dlon / a / coslat + dU_dlat / a + U * tanlat / a

    elif definition == "FY02":
        D_lon = (dU_dlon - dVcoslat_dlat) / a / coslat
        D_lat = (dV_dlon + dUcoslat_dlat) / a / coslat

    ## Updata metadata following calculations
    D_lon.rename('horizontal_stretching_deformation_component_of_flow')
    D_lat.rename('horizontal_shearing_deformation_component_of_flow')
    D_lon.units = 's-1'
    D_lat.units = 's-1'

    ## Return D components
    return D_lon, D_lat


def differentiate_horizontal(cube, coord):

    """Calculates the derivative of the input cube with respect to the horizontal
    coordinate and returns the derivative on the grid of the input field


    Args:
        cube (iris.cube.Cube): Field to calculate the derivative for
        coord (str): The coordinate to calculate the derivative over

    Returns:
        iris.cube.Cube:
            Derivative of the input cube with respect to the specified coord in units
            per radian

    """
    # Calculate the derivative and interpolate to the original grid
    d_dx = differentiate(cube, coord)
    d_dx = d_dx.interpolate([(coord, cube.coord(coord).points)], Linear())

    # Convert to original to units which makes sure the differentiation was done in
    # radians
    d_dx.convert_units(cube.units)

    # Re-add the original coordinate to fix annoying floating points errors and losing
    # coordinate bounds
    d_dx.replace_coord(cube.coord(coord))

    return d_dx
