# earthquake_tidal_seasonal_loading

Python code to calculate solid-earth tidal stresses and water loading stresses on faults, and analyze their correlation with seismicity.

## Requirements

- Python 3.7+
- Required packages: `numpy`, `pandas`, `matplotlib`, `scipy`, `xarray`
- SPOTL (Solid-Earth Tide Program): https://igppweb.ucsd.edu/~agnew/Spotl/spotlmain.html
- GRACE water loading data (https://podaac.github.io/tutorials/quarto_text/DataSubscriberDownloader.html and then downloaded with this: 
podaac-data-downloader -c TELLUS_GRAC-GRFO_MASCON_GRID_RL06.3_V4 -d ./data --start-date 2000-01-01T00:00:00Z --end-date 2022-01-01T00:00:00Z -b="-95,29,-87,37")

Install Python dependencies:
```bash
pip install numpy pandas matplotlib scipy xarray
```

## Workflow

### 1. Preprocess Earthquake Catalog

Convert earthquake catalog to standardized format with required columns: `time`, `longitude`, `latitude`, `magnitude`, `depth`.

```bash
python preprocess_catalog.py input_catalog.csv output_processed.csv
```

Options:
- `--min-mag`: Minimum magnitude filter (default: 0.0)
- `--max-depth`: Maximum depth in km (default: 20.0)
- `--plot`: Create a quick plot of the catalog
- `--interactive`: Use Matplotlib interactive selector for spatial filtering
- `--interactive-web`: Use browser-based interactive selector

Example:
```bash
python preprocess_catalog.py nm_usgs.csv nm_usgs_processed.csv --min-mag 1.0 --max-depth 15.0
```

### 2. Calculate Tidal Strain

Use SPOTL to calculate tidal strain at earthquake locations. You need to provide:
- Processed earthquake catalog
- Date range for strain calculation
- Path to SPOTL executable (typically `ertid`)

```bash
python spotl_helper.py --catalog nm_usgs_processed.csv \
    --start-date 2000-01-01 \
    --end-date 2022-01-01 \
    --spotl-executable /path/to/spotl/bin/ertid \
    --output strain_output.csv
```

The script automatically:
- Determines the centroid location from the catalog
- Generates SPOTL input files
- Runs the strain calculation
- Converts output to CSV format with columns: `time`, `strain_N_nanostrain`, `strain_E_nanostrain`, `strain_NE_nanostrain`

### 3. Tidal Sensitivity Analysis

Analyze tidal stress modulation of seismicity, optionally including GRACE water loading.

```bash
python tidal_seasonal.py \
    --catalog nm_usgs_processed.csv \
    --spotl-output strain_output.csv \
    --projection fault_plane \
    --strike 173 \
    --dip 73 \
    --rake 90 \
    --grace-file GRCTellus.JPL.200204_202506.GLO.RL06.3M.MSCNv04.nc \
    --grace-lon-min -92 --grace-lon-max -88 \
    --grace-lat-min 34 --grace-lat-max 38 \
    --output-prefix results \
    --n-bootstrap 1000 \
    --output-alpha-timeseries alpha_timeseries.csv
```

Parameters:
- `--projection`: `fault_plane` (requires strike/dip/rake) or `optimally_oriented`
- `--strike`, `--dip`, `--rake`: Fault orientation in degrees
- `--grace-file`: Optional GRACE NetCDF file for water loading analysis
- `--grace-lon-min/max`, `--grace-lat-min/max`: Region for GRACE data averaging
- `--n-bootstrap`: Number of bootstrap iterations for significance testing
- `--output-alpha-timeseries`: Save time-varying phase modulation parameter

Outputs:
- Phase-amplitude plots
- Sliding window analysis
- Bootstrap significance tests
- Alpha (phase modulation) vs time plots

### 4. Coherence Analysis (Optional)

Analyze coherence between tidal stress, GRACE water loading stress, and seismicity.

```bash
python coherence.py \
    --catalog nm_usgs_processed.csv \
    --spotl-output strain_output.csv \
    --grace-file GRCTellus.JPL.200204_202506.GLO.RL06.3M.MSCNv04.nc \
    --projection fault_plane \
    --strike 173 --dip 73 --rake 90 \
    --output-prefix coherence_results
```

Outputs:
- Time series plots of tidal and GRACE stresses
- Sliding window seismicity counts
- Coherence spectra
- Cross-correlation analysis

## Data Files

Example data files included:
- `nm_usgs.csv`, `nm_usgs_longer.csv`: New Madrid earthquake catalogs
- `GRCTellus.JPL.200204_202506.GLO.RL06.3M.MSCNv04.nc`: GRACE water loading data

## Adapting to Other Datasets

The scripts are designed to be generalizable:

1. **Earthquake Catalog**: Any CSV with columns for time, location (lon/lat), magnitude, and depth
2. **SPOTL Calculation**: Works for any geographic location; the script auto-detects catalog centroid
3. **Fault Parameters**: Specify strike/dip/rake for your fault of interest, or use `optimally_oriented` mode
4. **GRACE Data**: Adjust `--grace-lon/lat-min/max` to match your study region


## Citations
This code reproduces Beauce et al. (2023) and it applies it to a new region. Citing is appropriate if you use the code or results/interpretations in the following papers.

Method:
```
Beaucé, E., Poli, P., Waldhauser, F., Holtzman, B., and Scholz, C. (2023). Enhanced tidal sensitivity of seismicity before the 2019 magnitude 7.1 Ridgecrest, California earthquake. Geophysical Research Letters, 50, e2023GL104375, https://doi.org/10.1029/2023GL104375
```

This code:
```
Walter, J. I., H. DeShon, and P. Neupane (in review), Solid earth tides modulate earthquake activity in the New Madrid Seismic Zone, Seismological Research 
Letters
```

## Notes

- All times should be in UTC
- Depths should be in kilometers
- Longitudes can be -180 to 180 or 0 to 360 (automatically handled)
- GRACE data download instructions (not included): https://podaac.github.io/tutorials/
- SPOTL executable must be compiled separately from source
