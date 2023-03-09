# Constrain

Calculate climate indices from CMIP6 and reanalysis data

```python
"""
Calculate Arctic Oscillation from monthly mean-sea-level pressure anomalies
"""

import iris
from constrain import arctic_oscillation

# Load monthly MSLP anomalies
mslp = iris.load_cube("mslp_filename.nc")

# Only use MSLP north of 20N
mslp = mslp.extract(iris.Constraint(latitude=lambda y: y >= 20))

# Calculate AO pattern and index
ao_pattern, ao_index = arctic_oscillation.from_eofs(mslp)
```
