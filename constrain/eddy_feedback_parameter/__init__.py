from __future__ import division
import numpy as np
import iris
import iris.analysis.cartography
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

    Input: U(day,lat,lon) - iris cube of zonal wind
           V(day,lat,lon) - iris cube of meridional wind

    Output: divF_h(day,lat) - iris cube of horizontal EP flux
                              divergence
    """

    ### ---------------------------------------------------------------
    ### Sort cube coordinates and define some required parameters
    ### ---------------------------------------------------------------

    ## Convert latitudes to radians and calculate cos(latitude)
    lats = U.coord('latitude').copy()
    lats.convert_units('radians')
    lats = lats.points
    coslat = np.cos(lats)[np.newaxis]
    cos2lat = coslat ** 2

    ## Define radius of Earth in metres
    a = iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS

    ### ---------------------------------------------------------------
    ### Calculate momentum flux, UsVs_z
    ### (s = zonal deviation, z = zonal mean)
    ### ---------------------------------------------------------------

    ## NB: use fact that UsVs_z = UV_z - UzVz

    ## Calculate zonal mean of individual fields
    Uz = U.collapsed(['longitude'], iris.analysis.MEAN)
    Vz = V.collapsed(['longitude'], iris.analysis.MEAN)

    ## Calculate zonal mean of products of fields
    UV_z = (U * V).collapsed(['longitude'], iris.analysis.MEAN)

    ## Now calculate momentum flux
    UsVs_z = UV_z - Uz * Vz

    ### -----------------------------------------------------------------
    ### Calculate zonal accel due to divergence of horiz EP flux, Flat
    ### -----------------------------------------------------------------

    ## Calculation is -(1/a/cos2lat) * d(UsVs_z*cos2lat)/dlat

    ## Calculate UsVs_z*cos2lat (Flat*coslat)
    Flatcoslat = UsVs_z * cos2lat

    ## Calculate d(Flat*coslat)/dlat
    dFlatcoslat_dlat = Flatcoslat.copy()
    Flatcoslat_arr = Flatcoslat.data
    # One-sided finite difference for first latitude
    dFlatcoslat_dlat.data[:, 0] = (Flatcoslat_arr[:, 1] - \
                                   Flatcoslat_arr[:, 0]) / \
                                  (lats[1] - lats[0])
    # Centered finite difference for latitudes in between
    dFlatcoslat_dlat.data[:, 1:-1] = (Flatcoslat_arr[:, 2:] - \
                                      Flatcoslat_arr[:, :-2]) / \
                                     (lats[2:] - lats[:-2])
    # One-sided finite difference for last latitude
    dFlatcoslat_dlat.data[:, -1] = (Flatcoslat_arr[:, -1] - \
                                    Flatcoslat_arr[:, -2]) / \
                                   (lats[-1] - lats[-2])

    ## Calculate horizontal EP flux divergence
    divF_h = dFlatcoslat_dlat * (-1 / a / cos2lat)

    ## Update metadata following calculations
    divF_h.rename('tendency_of_eastward_wind_due_to_horizontal_eliassen_' + \
                  'palm_flux_divergence')
    divF_h.units = 'm s-2'
    divF_h.coord('latitude').convert_units('degrees')

    ## Return horizontal EP flux divergence
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
    iris.coord_categorisation.add_season_year(cube, 'time')

    # Calculate seasonal mean
    cube_b = cube.aggregated_by('season_year', iris.analysis.MEAN)
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
    r = pearsonr(divF_h, Uz, corr_coords='time')

    # Calculate percentage variance explained
    r2 = r ** 2
    r2.rename("Pearson's r squared")

    # Average r2 over latitudes 25N-72N to get eddy feedback parameter
    r2_box = r2.intersection(latitude=(25, 72))
    weights = iris.analysis.cartography.cosine_latitude_weights(r2_box)
    EFP = r2_box.collapsed('latitude', iris.analysis.MEAN, weights=weights)

    # Return eddy feedback parameter
    return EFP


def Evectors(u, v, months, window, f1, f2):
    """
    Created on Thurs Feb 02 18:35 2023

    @author: Christine McKenna

    =====================================================================
    Purpose: Calculates the horizontal E-vectors for a 2D flow (u,v)
             for the specified months and timescale of variability
             (defined through window, f1, f2).

             The E-vectors are defined in Cartesian coordinates in
             Mak and Cai 1989, where the x-component differs from the
             Hoskins et al. 1983 E-vector by a factor of a half:

             doi:10.1175/1520-0469(1989)046<3289:LBI>2.0.CO;2

             This code calculates the E-vectors in spherical geometry
             by multiplying the E-vectors by cos(lat).

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

    Output: E_lon(day,lat,lon) - iris cube of E-vector in longitude
                                 direction
            E_lat(day,lat,lon) - iris cube of E-vector in latitude
                                 direction

    """

    ### ---------------------------------------------------------------
    ### Sort cube coordinates and define some required parameters
    ### ---------------------------------------------------------------

    ## Convert latitudes to radians and calculate cos(latitude)
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

    E_lon = 1 / 2 * (v_hp ** 2 - u_hp ** 2) * coslat
    E_lat = (-1) * u_hp * v_hp * coslat

    ## Updata metadata following calculations
    E_lon.rename('Evector_in_longitude_direction')
    E_lat.rename('Evector_in_latitude_direction')
    E_lon.units = 'm2 s-2'
    E_lat.units = 'm2 s-2'

    ## Return E-vectors
    return E_lon, E_lat


def background_deformation_flow(u, v, months, window, f):
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

    Output: D_lon(day,lat,lon) - iris cube of stretching deformation
                                 component
            D_lat(day,lat,lon) - iris cube of shearing deformation
                                 component

    """

    ### ---------------------------------------------------------------
    ### Sort cube coordinates and define some required parameters
    ### ---------------------------------------------------------------

    ## Convert latitudes/longitudes to radians and
    ## calculate cos(latitude)
    lats = u.coord('latitude').copy()
    lats.convert_units('radians')
    coslat = np.cos(lats.points)[np.newaxis, :, np.newaxis]

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
    dU_dlon = dX_dlon(U)
    dV_dlon = dX_dlon(V)
    dVcoslat_dlat = dX_dlat(V * coslat)
    dUcoslat_dlat = dX_dlat(U * coslat)

    ## Calculate D_lon and D_lat
    D_lon = (dU_dlon - dVcoslat_dlat) / a / coslat
    D_lat = (dV_dlon + dUcoslat_dlat) / a / coslat

    ## Updata metadata following calculations
    D_lon.rename('horizontal_stretching_deformation_component_of_flow')
    D_lat.rename('horizontal_shearing_deformation_component_of_flow')
    D_lon.units = 's-1'
    D_lat.units = 's-1'

    ## Return D components
    return D_lon, D_lat


def dX_dlon(X):
    """
    Created on Mon Feb 06 16:45 2023

    @author: Christine McKenna

    ============================================================
    Purpose: Calculates the derivative in longitude of a field X
             with dimensions (time,lat,lon), where longitude does
             not wrap around.

    Input: X(time,lat,lon) - field X to calculate derivative
                             for, must be an iris cube

    Output: dX_dlon(time,lat,lon) - derivative of X in longitude

    """

    # Extract longitude coordinates and convert to radians
    lons = X.coord('longitude').copy()
    lons.convert_units('radians')
    lons = lons.points[np.newaxis, np.newaxis, :]

    # Create cube to save result in
    dX_dlon = X.copy()

    # One-sided finite difference for first longitude
    dX_dlon.data[:, :, 0] = (X.data[:, :, 1] - X.data[:, :, 0]) / \
                            (lons[:, :, 1] - lons[:, :, 0])
    # Centered finite difference for longitudes in between
    dX_dlon.data[:, :, 1:-1] = (X.data[:, :, 2:] - X.data[:, :, :-2]) / \
                               (lons[:, :, 2:] - lons[:, :, :-2])
    # One-sided finite difference for last longitude
    dX_dlon.data[:, :, -1] = (X.data[:, :, -1] - X.data[:, :, -2]) / \
                             (lons[:, :, -1] - lons[:, :, -2])

    return dX_dlon


def dX_dlat(X):
    """
    Created on Mon Feb 06 16:53 2023

    @author: Christine McKenna

    ============================================================
    Purpose: Calculates the derivative in latitude of a field X
             with dimensions (time,lat,lon).

    Input: X(time,lat,lon) - field X to calculate derivative
                             for, must be an iris cube

    Output: dX_dlat(time,lat,lon) - derivative of X in latitude

    """

    # Extract latitude coordinates and convert to radians
    lats = X.coord('latitude').copy()
    lats.convert_units('radians')
    lats = lats.points[np.newaxis, :, np.newaxis]

    # Create cube to save result in
    dX_dlat = X.copy()

    # One-sided finite difference for first latitude
    dX_dlat.data[:, 0, :] = (X.data[:, 1, :] - X.data[:, 0, :]) / \
                            (lats[:, 1, :] - lats[:, 0, :])
    # Centered finite difference for latitudes in between
    dX_dlat.data[:, 1:-1, :] = (X.data[:, 2:, :] - X.data[:, :-2, :]) / \
                               (lats[:, 2:, :] - lats[:, :-2, :])
    # One-sided finite difference for last latitude
    dX_dlat.data[:, -1, :] = (X.data[:, -1, :] - X.data[:, -2, :]) / \
                             (lats[:, -1, :] - lats[:, -2, :])

    return dX_dlat
