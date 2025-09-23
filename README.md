# earthquake_tidal_seasonal_loading
Python code to calculate solid-earth stresses on faults from tidal loading and water loading


python spotl_helper.py --catalog nm_filtered_3.csv --start-date 2012-01-01 --end-date 2022-01-01 --spotl-executable /Users/jwalter/seis/spotl/bin/ertid --output strain_nm.csv
python spotl_helper.py --catalog nm_usgs_processed_longer.csv --start-date 2000-01-01 --end-date 2022-01-01 --spotl-executable /Users/jwalter/seis/spotl/bin/ertid --output strain_nm_longer.csv

https://podaac.github.io/tutorials/quarto_text/DataSubscriberDownloader.html

podaac-data-downloader -c TELLUS_GRAC-GRFO_MASCON_GRID_RL06.3_V4 -d ./data --start-date 2000-01-01T00:00:00Z --end-date 2022-01-01T00:00:00Z -b="-95,29,-87,37"



python nm_tidal_seasonal.py --projection fault_plane --strike 173 --dip 73 --rake 90 --catalog  nm_usgs_processed_longer.csv  --spotl-output strain_nm_longer.csv --grace-file GRCTellus.JPL.200204_202506.GLO.RL06.3M.MSCNv04.nc --output-prefix reelfoot_longer --n-bootstrap 5000 --output-alpha-timeseries ceri_longer_alpha.csv

python plot_alpha_comparison.py ceri_alpha.csv phasenet_alpha.csv 

python test_coherence_enhanced.py --catalog nm_usgs_processed_longer.csv --spotl-output strain_nm_longer.csv --grace-file GRCTellus.JPL.200204_202506.GLO.RL06.3M.MSCNv04.nc --projection fault_plane --strike 173 --dip 73 --rake 90 --output-prefix reelfoot_longer



python seasonal_nm.py --catalog nm_usgs_processed_longer.csv --completeness-mag 1.4 --mag-min 1.4 --mag-max 3.0 --use-declustered --output-prefix usgs

python seasonal_alpha_analysis.py --alpha-csv ceri_longer_alpha.csv --output-prefix ceri_alpha
