from __future__ import division
import numpy as np
import iris
import iris.analysis.cartography


def lowpass_weights(window, cutoff):
    """
    Created on Jan 6 16:48 2017

    @author: iris website

    =========================================================
    Purpose: Calculates weights for a low pass Lanczos filter
    =========================================================

    Input: window - integer. The length of the filter window.
           cutoff - float. The cutoff frequency in inverse time steps.

    Output: w - weights

    """

    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma

    return w[1:-1]


def lowpass_lanczos(cube, window, cutoff):
    """
    Created on Jan 6 16:30 2017

    @author: iris website

    ================================================================
    Purpose: Low pass filters a time-series by applying a weighted
             running mean over the time dimension. Filter used is
             a Lanczos filter. Filters out timescales less than that
             defined by cutoff. Time coordinate for filtering is the
             same as the coordinate used for the time dimension of
             the cube (i.e. whether it's in days, months, years,
             etc). The number of weights used is specified by the
             window size.

             For more information, in particular regarding choosing
             the window size, see Duchon 1979:

             doi:10.1175/1520-0450(1979)018<1016:LFIOAT>2.0.CO;2

    ================================================================

    Input: cube - cube of the time-series you want to lowpass filter
           window - integer. The length of the filter window. Must
                    be an odd number, otherwise the function will
                    force it to be odd.
           cutoff - float. The cutoff frequency is in time steps.

    Output: cube_lp - lowpass filtered cube

    """

    wgts = lowpass_weights(window, 1/cutoff) # Find weights
    cube_lp = cube.rolling_window('time',iris.analysis.MEAN,\
                                  len(wgts),weights=wgts)

    return cube_lp
