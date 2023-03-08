#!/bin/bash

# Regrid ERA5 onto coarsest model grid (CanESM5) subset region north of 20N
GRID=../grid.nc
file_in=era5_monthly_mean_mslp_20N-90N_NDJFMA.nc
file_out=era5_monthly_mean_CanESM5_grid_mslp_20N-90N_NDJFMA.nc

cdo -sellonlatbox,-180,180,20,90 -remapbil,$GRID $file_in $file_out
