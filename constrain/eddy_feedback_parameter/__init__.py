from __future__ import division
import numpy as np
import iris
import iris.analysis.cartography
from iris.analysis.stats import pearsonr


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

