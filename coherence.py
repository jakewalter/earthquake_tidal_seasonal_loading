#!/usr/bin/env python3
"""
Enhanced Coherence Analysis: Tidal vs GRACE Coulomb Stress with Seismicity
Plots tidal Coulomb stress, GRACE Coulomb stress, and their coherence with
90-day sliding windows of cumulative seismicity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore")

# Import functions from the main analysis script
import sys
sys.path.append('.')
from tidal_seasonal import (
    load_catalog_for_paper, load_spotl_strain_csv,
    build_tidal_interpolator, load_grace_data, to_epoch_seconds
)

def compute_sliding_seismicity(catalog, window_days=90, time_grid=None):
    """Compute sliding window cumulative seismicity count (consistent with nm_tidal_seasonal).
    
    Parameters:
    -----------
    catalog : DataFrame
        Earthquake catalog with 'time' column
    window_days : int
        Window size in days
    time_grid : array-like
        Time points for evaluation
        
    Returns:
    --------
    seismicity_count : ndarray
        Number of events in sliding window at each time point
    """
    if time_grid is None:
        start_time = catalog['time'].min()
        end_time = catalog['time'].max()
        time_grid = pd.date_range(start=start_time, end=end_time, freq='D')
    
    # Use pandas time series operations for efficiency
    catalog_times = pd.to_datetime(catalog['time'])
    
    seismicity_count = np.zeros(len(time_grid))
    window_timedelta = pd.Timedelta(days=window_days)
    
    # Use pandas rolling window for efficiency, which is much faster
    s = pd.Series(np.ones(len(catalog_times)), index=catalog_times)
    daily_counts = s.resample('D').sum()
    # Use a centered rolling window and align with the original time_grid
    rolling_sum = daily_counts.rolling(window=f'{window_days}D', center=True).sum()
    seismicity_count = rolling_sum.reindex(time_grid, method='nearest').fillna(0).values
            
    return seismicity_count

def compute_coherence(x, y, fs=1.0, nperseg=256):
    """
    Compute coherence between two time series.
    """
    # Remove NaN values
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) < 10:
        return np.array([]), np.array([])

    x_clean = x[valid]
    y_clean = y[valid]

    # Compute coherence
    f, Cxy = signal.coherence(x_clean, y_clean, fs=fs, nperseg=min(nperseg, len(x_clean)//2))

    return f, Cxy

def bandpass_filter(series, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to a pandas Series.
    
    :param series: pandas Series with a DatetimeIndex.
    :param lowcut: Low frequency cutoff (in Hz).
    :param highcut: High frequency cutoff (in Hz).
    :param fs: Sampling frequency (in Hz).
    :param order: Filter order.
    :return: Filtered pandas Series.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Ensure the series is sorted by time
    series = series.sort_index()
    
    # Interpolate to fill any gaps and ensure a constant sampling rate
    interpolated_series = series.interpolate(method='time')
    
    # Design the filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply the filter
    filtered_data = filtfilt(b, a, interpolated_series.values)
    
    return pd.Series(filtered_data, index=interpolated_series.index)

def analyze_signal_coherence(catalog_path, spotl_path, grace_file,
                           strike=210, dip=30, rake=-90,
                           grace_lon_min=-92, grace_lon_max=-88,
                           grace_lat_min=34, grace_lat_max=38,
                           output_prefix='coherence_test',
                           projection='fault_plane',
                           n_boot=1000,
                           output_alpha_timeseries=None):
    """
    Enhanced coherence analysis with comprehensive plotting.
    """
    print("=== ENHANCED COHERENCE ANALYSIS ===")
    print(f"Catalog: {catalog_path}")
    print(f"SPOTL: {spotl_path}")
    print(f"GRACE: {grace_file}")
    print(f"Projection: {projection}")
    print(f"Fault parameters: strike={strike}, dip={dip}, rake={rake}")
    print(f"GRACE region: lon=[{grace_lon_min}, {grace_lon_max}], lat=[{grace_lat_min}, {grace_lat_max}]")

    # Load data
    print("\nLoading earthquake catalog...")
    catalog = load_catalog_for_paper(catalog_path)
    print(f"Loaded {len(catalog)} earthquakes")

    print("\nLoading SPOTL strain data...")
    strain_df = load_spotl_strain_csv(spotl_path)
    print(f"Loaded strain data: {len(strain_df)} time points")

    # Build tidal stress interpolator
    print("\nBuilding tidal stress interpolator...")
    interp_kwargs = {'projection_mode': projection}
    if projection == 'fault_plane':
        if None in [strike, dip, rake]:
            raise ValueError("strike, dip, rake must be provided for fault_plane projection")
        interp_kwargs.update({'strike': strike, 'dip': dip, 'rake': rake})
    
    tidal_interp = build_tidal_interpolator(strain_df, **interp_kwargs)

    # Load GRACE data
    print("\nLoading GRACE data...")
    try:
        grace_interp, grace_df = load_grace_data(
            grace_file, grace_lon_min, grace_lon_max,
            grace_lat_min, grace_lat_max,
            strike=strike, dip=dip, rake=rake
        )
        print(f"Loaded GRACE data: {len(grace_df)} time points")
    except Exception as e:
        print(f"Error loading GRACE data: {e}")
        return None

    # Create common time grid (daily sampling for seismicity analysis)
    start_time = max(catalog['time'].min(), strain_df['time'].min(), grace_df['time'].min())
    end_time = min(catalog['time'].max(), strain_df['time'].max(), grace_df['time'].max())
    
    print(f"\nOverlap period: {start_time.date()} to {end_time.date()}")

    # Hourly time grid for stress calculations (same as original test_coherence.py)
    time_grid_hourly = pd.date_range(start=start_time, end=end_time, freq='H')
    time_secs_hourly = to_epoch_seconds(time_grid_hourly)

    # Daily time grid for seismicity analysis
    time_grid_daily = pd.date_range(start=start_time, end=end_time, freq='D')

    print(f"Hourly grid: {len(time_grid_hourly)} points")
    print(f"Daily grid: {len(time_grid_daily)} points")

    # Compute stress time series at hourly resolution
    print("\nComputing stress time series...")
    try:
        tidal_stress_result = tidal_interp(time_secs_hourly)
        if isinstance(tidal_stress_result, dict):
            tidal_coulomb_hourly = tidal_stress_result['coulomb_stress']
        else:
            tidal_coulomb_hourly = tidal_stress_result
        print(f"Tidal Coulomb stress computed: {len(tidal_coulomb_hourly)} points")
    except Exception as e:
        print(f"Error computing tidal stress: {e}")
        return None

    try:
        grace_coulomb_hourly = grace_interp(time_secs_hourly)
        print(f"GRACE Coulomb stress computed: {len(grace_coulomb_hourly)} points")
    except Exception as e:
        print(f"Error computing GRACE stress: {e}")
        return None

    # Attempt to construct GRACE load (Pa) time series on the hourly grid and
    # compute Coulomb-from-load using helper from nm_tidal_seasonal if available.
    grace_load_hourly = None
    grace_dCFS_from_load = None
    try:
        # If grace_df returned by load_grace_data contains 'load_pa', use it
        if 'load_pa' in grace_df.columns:
            # Interpolate grace_df load_pa onto hourly times
            gf = grace_df.copy()
            gf['time'] = pd.to_datetime(gf['time'])
            gf = gf.sort_values('time')
            interp_load = pd.Series(gf['load_pa'].values, index=gf['time'])
            # Use pandas reindex/interpolate to hourly grid
            interp_hourly = interp_load.reindex(pd.DatetimeIndex(time_grid_hourly)).interpolate(method='time')
            grace_load_hourly = interp_hourly.values
        else:
            # If load_pa not present, try if grace_interp can provide load (some implementations)
            # Otherwise leave as None
            try:
                # If grace_interp returns dict-like with load_pa, attempt to call
                test = grace_interp(time_secs_hourly)
                if isinstance(test, dict) and 'load_pa' in test:
                    grace_load_hourly = test['load_pa']
            except Exception:
                grace_load_hourly = None
    except Exception:
        grace_load_hourly = None

    # If we have a load series, compute Coulomb change from that load using helper
    try:
        from tidal_seasonal import coulomb_stress_change
        if grace_load_hourly is not None:
            dCFS_list = []
            for val in grace_load_hourly:
                if np.isfinite(val):
                    dCFS_val, _, _ = coulomb_stress_change(val, strike, dip, rake, 0.4)
                else:
                    dCFS_val = np.nan
                dCFS_list.append(dCFS_val)
            grace_dCFS_from_load = np.array(dCFS_list)
    except Exception as e:
        print('Could not compute Coulomb-from-load:', e)
        grace_dCFS_from_load = None

    # Save CSV with time, load_pa (if available), and dCFS_from_load (if available)
    out_df = pd.DataFrame({'time': pd.to_datetime(time_grid_hourly)})
    if grace_load_hourly is not None:
        out_df['load_pa'] = grace_load_hourly
    if grace_dCFS_from_load is not None:
        out_df['dCFS_from_load'] = grace_dCFS_from_load
    
    if 'load_pa' in out_df.columns or 'dCFS_from_load' in out_df.columns:
        try:
            out_csv = f"{output_prefix}_grace_load_timeseries.csv"
            out_df.to_csv(out_csv, index=False)
            print(f"Saved GRACE load timeseries to: {out_csv}")
        except Exception as e:
            print('Failed to save GRACE load CSV:', e)

    # Use hourly data directly (no downsampling needed)
    tidal_coulomb_data = tidal_coulomb_hourly
    grace_coulomb_data = grace_coulomb_hourly
    time_grid_data = time_grid_hourly

    print(f"Hourly tidal stress: {len(tidal_coulomb_data)} points")
    print(f"Hourly GRACE stress: {len(grace_coulomb_data)} points")

    # Compute 90-day sliding seismicity
    print("\nComputing 90-day sliding seismicity...")
    seismicity_count = compute_sliding_seismicity(catalog, window_days=90, time_grid=time_grid_daily)
    print(f"Seismicity count computed: {len(seismicity_count)} points")

    # Compute tidal stress variability (peak-to-trough over 90-day windows)
    print("Computing tidal stress variability...")
    def compute_tidal_variability(stress_data, stress_times, window_days=90, time_grid=None):
        """
        Compute running peak-to-trough variability of tidal stress
        """
        if time_grid is None:
            time_grid = pd.date_range(start=stress_times.min(), end=stress_times.max(), freq='D')
        
        # Create stress DataFrame for efficient windowing
        stress_df = pd.DataFrame({'stress': stress_data}, index=stress_times)
        
        # Use rolling window to calculate max and min, which is much faster
        rolling_max = stress_df['stress'].rolling(window=f'{window_days}D', center=True).max()
        rolling_min = stress_df['stress'].rolling(window=f'{window_days}D', center=True).min()
        
        variability_series = rolling_max - rolling_min
        
        # Align with the output time_grid
        variability = variability_series.reindex(time_grid, method='nearest').fillna(np.nan).values
        
        return variability
    
    tidal_variability = compute_tidal_variability(tidal_coulomb_data, time_grid_data, window_days=90, time_grid=time_grid_daily)
    print(f"Tidal variability computed: {len(tidal_variability)} points")

    # Create comprehensive plots
    print("\nCreating plots...")
    
    # Figure 1: Time series overview (now 6 panels: combine normalized plots)
    fig, axes = plt.subplots(6, 1, figsize=(15, 18), sharex=True)
    fig.suptitle(f'Coulomb Stress and Seismicity Analysis\nFault: strike={strike}°, dip={dip}°, rake={rake}°', 
                 fontsize=14, fontweight='bold')

    # Plot 1: Tidal Coulomb stress (hourly data - subsample for plotting efficiency)
    # Subsample hourly data for plotting (every 6 hours to reduce plot complexity)
    step = 6  # Plot every 6th hour
    time_sub = time_grid_data[::step]
    tidal_sub = tidal_coulomb_data[::step]
    valid_tidal = ~np.isnan(tidal_sub)
    
    axes[0].plot(time_sub[valid_tidal], tidal_sub[valid_tidal], 
                'b-', linewidth=0.3, alpha=0.8, label='Tidal Coulomb Stress')
    axes[0].set_ylabel('Tidal Coulomb\nStress (Pa)', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    
    # Statistics
    tidal_stats = f'σ={np.nanstd(tidal_coulomb_data):.2e} Pa'
    axes[0].text(0.02, 0.95, tidal_stats, transform=axes[0].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Plot 2: GRACE Coulomb stress (subsample for consistency)
    grace_sub = grace_coulomb_data[::step]
    valid_grace = ~np.isnan(grace_sub)
    axes[1].plot(time_sub[valid_grace], grace_sub[valid_grace], 
                'r-', linewidth=0.5, label='GRACE Coulomb Stress')
    axes[1].set_ylabel('GRACE Coulomb\nStress (Pa)', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    # Statistics
    grace_stats = f'σ={np.nanstd(grace_coulomb_data):.2e} Pa'
    axes[1].text(0.02, 0.95, grace_stats, transform=axes[1].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    # Plot 3: 90-day seismicity count
    axes[2].plot(time_grid_daily, seismicity_count, 'g-', linewidth=1.0, label='90-day Seismicity Count')
    axes[2].set_ylabel('Seismicity Count\n(90-day window)', fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')
    
    # Statistics
    seis_stats = f'μ={np.mean(seismicity_count):.1f}, σ={np.std(seismicity_count):.1f} events'
    axes[2].text(0.02, 0.95, seis_stats, transform=axes[2].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Plot 4: Tidal stress variability (peak-to-trough)
    valid_variability = ~np.isnan(tidal_variability)
    axes[3].plot(time_grid_daily[valid_variability], tidal_variability[valid_variability], 
                'orange', linewidth=1.0, label='Tidal Stress Variability (90-day P2T)')
    axes[3].set_ylabel('Tidal Variability\n(Peak-to-Trough, Pa)', fontsize=10)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')
    
    # Statistics
    var_stats = f'μ={np.nanmean(tidal_variability):.0f}, σ={np.nanstd(tidal_variability):.0f} Pa'
    axes[3].text(0.02, 0.95, var_stats, transform=axes[3].transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.5))

    # Plot 5: Combined normalized comparison with full resolution (including alpha)
    # Create daily averages of hourly stress for normalized comparison
    def downsample_for_comparison(hourly_data, hourly_times, daily_times):
        df = pd.DataFrame({'time': hourly_times, 'value': hourly_data})
        df['date'] = df['time'].dt.date
        daily_means = df.groupby('date')['value'].mean()
        daily_data = np.full(len(daily_times), np.nan)
        for i, daily_time in enumerate(daily_times):
            date_key = daily_time.date()
            if date_key in daily_means.index:
                daily_data[i] = daily_means[date_key]
        return daily_data
    
    # Normalize all signals for comparison
    def normalize_signal(x):
        x_clean = x[~np.isnan(x)]
        if len(x_clean) > 0:
            return (x - np.nanmean(x)) / np.nanstd(x)
        return x

    # Use FULL RESOLUTION for normalized plot
    # Tidal: use hourly data (subsampled for plotting efficiency)
    tidal_norm_full = normalize_signal(tidal_coulomb_data)
    grace_norm_full = normalize_signal(grace_coulomb_data)
    
    # For seismicity: interpolate daily values to hourly grid for comparison
    seis_interp = np.interp(
        range(len(time_grid_data)), 
        np.linspace(0, len(time_grid_data)-1, len(time_grid_daily)),
        seismicity_count
    )
    seis_norm_full = normalize_signal(seis_interp)
    
    # For tidal variability: interpolate daily values to hourly grid
    var_interp = np.interp(
        range(len(time_grid_data)), 
        np.linspace(0, len(time_grid_data)-1, len(time_grid_daily)),
        tidal_variability
    )
    var_norm_full = normalize_signal(var_interp)

    # Plot with subsampling for efficiency but preserving full variability
    step = 6  # Same subsampling as other plots
    axes[4].plot(time_sub, tidal_norm_full[::step], 'b-', linewidth=0.3, alpha=0.8, label='Tidal Stress (normalized)')
    axes[4].plot(time_sub, grace_norm_full[::step], 'r-', linewidth=0.5, alpha=0.9, label='GRACE Stress (normalized)')
    axes[4].plot(time_sub, seis_norm_full[::step], 'g-', linewidth=0.8, label='Seismicity (normalized)')
    axes[4].plot(time_sub, var_norm_full[::step], 'orange', linewidth=0.8, label='Tidal Variability (normalized)')
    
    # Add normalized alpha to the same plot
    alpha_df = None
    try:
        from tidal_seasonal import compute_alpha_timeseries
        print('Computing alpha time series for plotting...')
        alpha_df = compute_alpha_timeseries(catalog, tidal_interp, window_days=90, overlap_percent=0.9, min_events=10)
    except Exception as e:
        print('Could not compute alpha time series:', e)

    # Save alpha time series if requested
    if alpha_df is not None and not alpha_df.empty and output_alpha_timeseries is not None:
        try:
            alpha_df.to_csv(output_alpha_timeseries, index=False)
            print(f"Saved alpha time series to: {output_alpha_timeseries}")
        except Exception as e:
            print(f"Could not save alpha time series: {e}")

    # Compute filtered alpha time series for use throughout the analysis
    alpha_filtered_series = None
    alpha_filtered_daily = None
    
    if alpha_df is not None and not alpha_df.empty:
        alpha_vals = alpha_df['alpha_coulomb'].values if 'alpha_coulomb' in alpha_df.columns else (alpha_df['alpha'].values if 'alpha' in alpha_df.columns else None)
        if alpha_vals is not None:
            try:
                # Create a pandas Series for alpha
                alpha_series = pd.Series(alpha_vals, index=pd.to_datetime(alpha_df['time']))
                
                # Resample to regular daily interval before filtering
                alpha_daily_resampled = alpha_series.resample('D').mean().interpolate(method='time')
                
                # Define filter parameters for daily data - use periods in days, not frequencies
                # For daily sampling: Nyquist frequency is 0.5 cycles/day
                # 2-24 month periods = 60-720 days
                
                # Convert periods to normalized frequencies
                fs = 1.0  # 1 sample per day
                nyquist = 0.5 * fs
                
                # Periods in days
                low_period_days = 6 * 30  # 6 months = 180 days
                high_period_days = 24 * 30  # 24 months = 720 days
                
                # Convert to frequencies (cycles per day)
                high_freq = 1.0 / low_period_days  # Higher frequency (shorter period)
                low_freq = 1.0 / high_period_days  # Lower frequency (longer period)
                
                # Normalize by Nyquist frequency
                low_norm = low_freq / nyquist
                high_norm = high_freq / nyquist
                
                print(f"Filter parameters: {low_period_days}-{high_period_days} day periods")
                print(f"Normalized frequencies: [{low_norm:.6f}, {high_norm:.6f}]")
                
                # Check if frequencies are valid
                if low_norm >= 1.0 or high_norm >= 1.0:
                    print("Warning: Filter frequencies exceed Nyquist limit - skipping filtering")
                    alpha_filtered_series = None
                else:
                    # Apply bandpass filter using scipy directly
                    from scipy.signal import butter, filtfilt
                    b, a = butter(3, [low_norm, high_norm], btype='band')
                    filtered_values = filtfilt(b, a, alpha_daily_resampled.values)
                    alpha_filtered_series = pd.Series(filtered_values, index=alpha_daily_resampled.index)
                
                # Interpolate filtered alpha to the daily grid for hypothesis testing
                alpha_filtered_daily = alpha_filtered_series.reindex(time_grid_daily, method='nearest').values
                
                print(f"Alpha filtering successful. Original range: [{np.nanmin(alpha_vals):.4f}, {np.nanmax(alpha_vals):.4f}]")
                print(f"Filtered range: [{np.nanmin(alpha_filtered_series.values):.4f}, {np.nanmax(alpha_filtered_series.values):.4f}]")
                
            except Exception as e:
                print(f"Error in alpha filtering: {e}")
                alpha_filtered_series = None
                alpha_filtered_daily = None
            
            # Plot the original normalized alpha
            mask = ~np.isnan(alpha_vals)
            norm_alpha = np.full_like(alpha_vals, np.nan)
            if mask.any():
                norm_alpha[mask] = (alpha_vals[mask] - np.nanmean(alpha_vals[mask])) / np.nanstd(alpha_vals[mask])
            axes[4].plot(alpha_df['time'], norm_alpha, color='m', linewidth=1.2, label='Alpha (normalized)')

    axes[4].set_ylabel('Normalized\nAmplitude', fontsize=10)
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc='upper right')

    # Plot 6: GRACE Load vs Coulomb-from-load comparison (moved from separate figure)
    try:
        if ('load_pa' in out_df.columns) or ('dCFS_from_load' in out_df.columns):
            times_plot = pd.to_datetime(out_df['time'])
            # Normalize helper
            def norm(x):
                x = np.array(x, dtype=float)
                mask = np.isfinite(x)
                res = np.full_like(x, np.nan)
                if mask.any():
                    res[mask] = (x[mask] - np.nanmean(x[mask])) / np.nanstd(x[mask])
                return res

            if 'load_pa' in out_df.columns:
                axes[5].plot(times_plot, norm(out_df['load_pa'].values), label='GRACE load (normalized)', color='tab:blue')
            if 'dCFS_from_load' in out_df.columns:
                axes[5].plot(times_plot, norm(out_df['dCFS_from_load'].values), label='Coulomb-from-load (normalized)', color='tab:red')

            axes[5].set_ylabel('GRACE Load\nComparison', fontsize=10)
            axes[5].set_xlabel('Time', fontsize=10)
            axes[5].legend(loc='upper right')
            axes[5].grid(True, alpha=0.3)
        else:
            axes[5].text(0.5, 0.5, 'GRACE load data unavailable', ha='center', va='center')
            axes[5].set_ylabel('GRACE Load\nComparison', fontsize=10)
            axes[5].set_xlabel('Time', fontsize=10)
    except Exception as e:
        print(f"Error plotting GRACE load comparison: {e}")
        axes[5].text(0.5, 0.5, 'GRACE load comparison unavailable', ha='center', va='center')
        axes[5].set_ylabel('GRACE Load\nComparison', fontsize=10)
        axes[5].set_xlabel('Time', fontsize=10)

    # Format x-axis - use time range from daily grid for all plots
    for ax in axes:
        ax.set_xlim(time_grid_daily.min(), time_grid_daily.max())
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        # Rotate labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_time_series.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Figure 1.5: Alpha vs GRACE Coulomb Stress Comparison
    if alpha_df is not None and not alpha_df.empty:
        try:
            fig_alpha, ax1 = plt.subplots(1, 1, figsize=(15, 6))
            fig_alpha.suptitle('Alpha Time Series vs GRACE Coulomb Stress', fontsize=14, fontweight='bold')
            
            # Plot alpha on primary y-axis
            alpha_vals = alpha_df['alpha_coulomb'].values if 'alpha_coulomb' in alpha_df.columns else (alpha_df['alpha'].values if 'alpha' in alpha_df.columns else None)
            if alpha_vals is not None:
                line1 = ax1.plot(alpha_df['time'], alpha_vals, color='purple', linewidth=1.5, label='Alpha (Coulomb)')
                ax1.set_xlabel('Time', fontsize=12)
                ax1.set_ylabel('Alpha (Coulomb)', fontsize=12, color='purple')
                ax1.tick_params(axis='y', labelcolor='purple')
                ax1.grid(True, alpha=0.3)
                
                # Create secondary y-axis for GRACE Coulomb stress
                ax2 = ax1.twinx()
                
                # Downsample GRACE data to daily for cleaner plotting
                grace_daily_for_corr = downsample_for_comparison(grace_coulomb_data, time_grid_data, time_grid_daily)
                grace_daily_plot = grace_daily_for_corr
                valid_grace_plot = ~np.isnan(grace_daily_plot)
                
                line2 = ax2.plot(time_grid_daily[valid_grace_plot], grace_daily_plot[valid_grace_plot], 
                               color='red', linewidth=1.0, alpha=0.7, label='GRACE Coulomb Stress')
                ax2.set_ylabel('GRACE Coulomb Stress (Pa)', fontsize=12, color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # Format x-axis
                ax1.set_xlim(time_grid_daily.min(), time_grid_daily.max())
                ax1.xaxis.set_major_locator(mdates.YearLocator())
                ax1.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                
                # Add combined legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper left')
                
                # Add statistics text box
                alpha_mean = np.nanmean(alpha_vals)
                alpha_std = np.nanstd(alpha_vals)
                grace_mean = np.nanmean(grace_coulomb_data)
                grace_std = np.nanstd(grace_coulomb_data)
                
                stats_text = f'Alpha: μ={alpha_mean:.4f}, σ={alpha_std:.4f}\nGRACE: μ={grace_mean:.2e}, σ={grace_std:.2e} Pa'
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                        fontsize=10)
                
                plt.tight_layout()
                plt.savefig(f'{output_prefix}_alpha_grace_comparison.png', dpi=300, bbox_inches='tight')
                plt.close(fig_alpha)
                
                print(f"Saved alpha vs GRACE comparison plot: {output_prefix}_alpha_grace_comparison.png")
            else:
                print("No alpha values available for comparison plot")
        except Exception as e:
            print(f"Error creating alpha vs GRACE comparison plot: {e}")
    else:
        print("Alpha time series unavailable - skipping alpha vs GRACE comparison plot")

    # Figure 3: Hypothesis testing plot
    if alpha_df is not None and not alpha_df.empty:
        try:
            # Create hypothesis figure: share the date x-axis only between the
            # first two panels (date-based plots). The bottom panel shows
            # month-of-year and must NOT share the date x-axis or it will
            # inherit date limits (causing large empty white space).
            fig_hyp = plt.figure(figsize=(12, 10))
            gs = fig_hyp.add_gridspec(3, 1, height_ratios=[1, 1, 0.9])
            ax1 = fig_hyp.add_subplot(gs[0, 0])
            ax2 = fig_hyp.add_subplot(gs[1, 0], sharex=ax1)
            ax3 = fig_hyp.add_subplot(gs[2, 0])
            fig_hyp.suptitle('Hypothesis: Alpha as the Controlling Factor for Seismicity Modulation', fontsize=14, fontweight='bold')

            # --- Panel 1: Forcing Terms ---
            ax1.set_title('Forcing Terms: Tidal Stress and Alpha (Sensitivity)', fontsize=12)
            
            # Normalize signals
            def normalize(x):
                x_clean = x[~np.isnan(x)]
                if len(x_clean) > 0:
                    return (x - np.nanmean(x_clean)) / np.nanstd(x_clean)
                return x

            # Use the filtered alpha for the hypothesis test if available, otherwise use unfiltered
            if alpha_filtered_daily is not None:
                alpha_daily = alpha_filtered_daily.copy()
                alpha_label = 'Filtered Alpha (6-24mo, norm)'
                print(f"Using filtered alpha: {np.sum(~np.isnan(alpha_daily))} valid points out of {len(alpha_daily)}")
            else:
                # Fallback to unfiltered alpha
                alpha_vals = alpha_df['alpha_coulomb'].values if 'alpha_coulomb' in alpha_df.columns else (alpha_df['alpha'].values if 'alpha' in alpha_df.columns else None)
                alpha_times = pd.to_datetime(alpha_df['time'].values)
                x_interp = (time_grid_daily - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
                xp_interp = (alpha_times - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
                alpha_daily = np.interp(x_interp, xp_interp, alpha_vals)
                alpha_label = 'Alpha (unfiltered, norm)'
                print(f"Using unfiltered alpha: {np.sum(~np.isnan(alpha_daily))} valid points out of {len(alpha_daily)}")
            
            tidal_daily_for_plot = downsample_for_comparison(tidal_coulomb_data, time_grid_data, time_grid_daily)

            # Clean data before normalization
            valid_alpha = ~np.isnan(alpha_daily)
            valid_tidal = ~np.isnan(tidal_daily_for_plot)
            valid_seis = ~np.isnan(seismicity_count)
            
            print(f"Data validity: Alpha {np.sum(valid_alpha)}/{len(alpha_daily)}, Tidal {np.sum(valid_tidal)}/{len(tidal_daily_for_plot)}, Seismicity {np.sum(valid_seis)}/{len(seismicity_count)}")
            
            # Only normalize and plot data where we have valid values
            norm_alpha = np.full_like(alpha_daily, np.nan)
            norm_tidal = np.full_like(tidal_daily_for_plot, np.nan)
            
            if np.sum(valid_alpha) > 10:
                alpha_clean = alpha_daily[valid_alpha]
                norm_alpha[valid_alpha] = (alpha_clean - np.nanmean(alpha_clean)) / np.nanstd(alpha_clean)
            
            if np.sum(valid_tidal) > 10:
                tidal_clean = tidal_daily_for_plot[valid_tidal]
                norm_tidal[valid_tidal] = (tidal_clean - np.nanmean(tidal_clean)) / np.nanstd(tidal_clean)

            # Plot only valid data
            valid_tidal_plot = ~np.isnan(norm_tidal)
            valid_alpha_plot = ~np.isnan(norm_alpha)
            
            if np.sum(valid_tidal_plot) > 0:
                ax1.plot(time_grid_daily[valid_tidal_plot], norm_tidal[valid_tidal_plot], 'b-', linewidth=0.5, alpha=0.7, label='Tidal Coulomb Stress (norm)')
            if np.sum(valid_alpha_plot) > 0:
                ax1.plot(time_grid_daily[valid_alpha_plot], norm_alpha[valid_alpha_plot], 'm-', linewidth=1.2, label=alpha_label)
            ax1.set_ylabel('Normalized Amplitude')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)

            # --- Panel 2: Predicted vs. Observed Seismicity ---
            # ax2 already created above and shares x-axis with ax1
            ax2.set_title('Hypothesis Test: Predicted vs. Observed Seismicity Rate', fontsize=12)

            # Predicted seismicity = alpha * tidal_stress (only where both are valid)
            predicted_seismicity = np.full_like(alpha_daily, np.nan)
            valid_both = valid_alpha & valid_tidal
            if np.sum(valid_both) > 10:
                predicted_seismicity[valid_both] = alpha_daily[valid_both] * tidal_daily_for_plot[valid_both]
            
            # Normalize predicted and observed seismicity
            norm_predicted = np.full_like(predicted_seismicity, np.nan)
            norm_observed = np.full_like(seismicity_count, np.nan)
            
            if np.sum(~np.isnan(predicted_seismicity)) > 10:
                pred_clean = predicted_seismicity[~np.isnan(predicted_seismicity)]
                norm_predicted[~np.isnan(predicted_seismicity)] = (pred_clean - np.nanmean(pred_clean)) / np.nanstd(pred_clean)
            
            if np.sum(valid_seis) > 10:
                seis_clean = seismicity_count[valid_seis]
                norm_observed[valid_seis] = (seis_clean - np.nanmean(seis_clean)) / np.nanstd(seis_clean)

            # Plot only valid data
            valid_obs_plot = ~np.isnan(norm_observed)
            valid_pred_plot = ~np.isnan(norm_predicted)
            
            if np.sum(valid_obs_plot) > 0:
                ax2.plot(time_grid_daily[valid_obs_plot], norm_observed[valid_obs_plot], 'g-', linewidth=1.2, label='Observed Seismicity Rate (norm)')
            if np.sum(valid_pred_plot) > 0:
                ax2.plot(time_grid_daily[valid_pred_plot], norm_predicted[valid_pred_plot], 'k--', linewidth=1.0, label='Predicted Rate (Alpha * Tidal Stress, norm)')
            
            # Calculate correlation
            valid_comparison = ~np.isnan(norm_observed) & ~np.isnan(norm_predicted)
            corr_pred_obs, p_pred_obs = pearsonr(norm_predicted[valid_comparison], norm_observed[valid_comparison])
            
            ax2.set_ylabel('Normalized Amplitude')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            ax2.text(0.02, 0.95, f'Correlation(Observed, Predicted): r = {corr_pred_obs:.3f}', 
                     transform=ax2.transAxes, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            print(f"Valid data for correlation: {len(tidal_clean)} points ({np.sum(valid_all)/len(time_grid_daily)*100:.1f}%)")

            plt.tight_layout()
            plt.savefig(f'{output_prefix}_hypothesis.png', dpi=300, bbox_inches='tight')
            plt.close(fig_hyp)
            print(f"Saved hypothesis testing plot: {output_prefix}_hypothesis.png")
        except Exception as e:
            print(f"Error creating hypothesis testing plot: {e}")

        # Correlations figure
        try:
            # Create a new figure for correlations with 2x3 grid
            fig_corr, axes_corr = plt.subplots(2, 3, figsize=(15, 10))
            fig_corr.suptitle('Signal Correlations and Coherence Analysis', fontsize=14, fontweight='bold')

            # Prepare data for correlations (similar to hypothesis testing)
            tidal_daily_for_plot = downsample_for_comparison(tidal_coulomb_data, time_grid_data, time_grid_daily)
            grace_daily_for_corr = downsample_for_comparison(grace_coulomb_data, time_grid_data, time_grid_daily)

            # Clean data
            valid_tidal = ~np.isnan(tidal_daily_for_plot)
            valid_grace = ~np.isnan(grace_daily_for_corr)
            valid_seis = ~np.isnan(seismicity_count)
            valid_var = ~np.isnan(tidal_variability)

            # Get clean data arrays
            tidal_clean = tidal_daily_for_plot[valid_tidal] if np.sum(valid_tidal) > 10 else np.array([])
            grace_clean = grace_daily_for_corr[valid_grace] if np.sum(valid_grace) > 10 else np.array([])
            seis_clean = seismicity_count[valid_seis] if np.sum(valid_seis) > 10 else np.array([])
            var_clean = tidal_variability[valid_var] if np.sum(valid_var) > 10 else np.array([])

            if len(tidal_clean) > 10 and len(seis_clean) > 10 and len(var_clean) > 10:
                # Top row: Alpha-GRACE correlation scatter plot (density)
                if alpha_df is not None and not alpha_df.empty:
                    alpha_vals = alpha_df['alpha_coulomb'].values if 'alpha_coulomb' in alpha_df.columns else (alpha_df['alpha'].values if 'alpha' in alpha_df.columns else None)
                    if alpha_vals is not None:
                        # Interpolate GRACE to alpha time points for scatter plot
                        alpha_times = alpha_df['time'].values
                        grace_daily_for_corr = downsample_for_comparison(grace_coulomb_data, time_grid_data, time_grid_daily)
                        grace_at_alpha = np.interp(
                            (alpha_times - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'D'),
                            (time_grid_daily - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'D'),
                            grace_daily_for_corr
                        )

                        # Clean data for scatter plot
                        valid_scatter = ~(np.isnan(alpha_vals) | np.isnan(grace_at_alpha))
                        if np.sum(valid_scatter) > 10:
                            alpha_scatter = alpha_vals[valid_scatter]
                            grace_scatter = grace_at_alpha[valid_scatter]

                            # Create density scatter plot
                            try:
                                from scipy.stats import gaussian_kde
                                xy = np.vstack([alpha_scatter, grace_scatter])
                                kde = gaussian_kde(xy)(xy)
                                idx = kde.argsort()
                                x, y, z = alpha_scatter[idx], grace_scatter[idx], kde[idx]
                                sc = axes_corr[0,0].scatter(x, y, c=z, s=20, cmap='coolwarm', alpha=0.6)
                                # Add colorbar
                                cbar = plt.colorbar(sc, ax=axes_corr[0,0], shrink=0.8)
                                cbar.set_label('Density')
                            except:
                                # Fallback to regular scatter
                                axes_corr[0,0].scatter(alpha_scatter, grace_scatter, alpha=0.3, s=10, color='blue')

                            corr_ag_scatter, p_ag_scatter = pearsonr(alpha_scatter, grace_scatter)
                            axes_corr[0,0].set_xlabel('Alpha (Coulomb)')
                            axes_corr[0,0].set_ylabel('GRACE Coulomb Stress (Pa)')
                            axes_corr[0,0].set_title(f'Alpha vs GRACE\nr = {corr_ag_scatter:.3f} (p = {p_ag_scatter:.3e})')
                            axes_corr[0,0].grid(True, alpha=0.3)
                        else:
                            axes_corr[0,0].text(0.5, 0.5, 'Insufficient data\nfor Alpha-GRACE scatter', ha='center', va='center')
                    else:
                        axes_corr[0,0].text(0.5, 0.5, 'Alpha data\nunavailable', ha='center', va='center')
                else:
                    axes_corr[0,0].text(0.5, 0.5, 'Alpha data\nunavailable', ha='center', va='center')

                # Top-middle: Alpha vs Seismicity correlation (density plot)
                if alpha_df is not None and not alpha_df.empty and len(seis_clean) > 10:
                    alpha_vals = alpha_df['alpha_coulomb'].values if 'alpha_coulomb' in alpha_df.columns else (alpha_df['alpha'].values if 'alpha' in alpha_df.columns else None)
                    if alpha_vals is not None:
                        # Interpolate alpha to daily grid for correlation with seismicity
                        alpha_times = pd.to_datetime(alpha_df['time'])
                        alpha_daily_interp = np.interp(
                            time_grid_daily.astype('int64') // 10**9,
                            alpha_times.astype('int64') // 10**9,
                            alpha_vals
                        )
                        
                        # Get valid data where both alpha and seismicity exist
                        valid_alpha_seis = ~np.isnan(alpha_daily_interp) & valid_seis
                        if np.sum(valid_alpha_seis) > 10:
                            alpha_seis = alpha_daily_interp[valid_alpha_seis]
                            seis_for_alpha = seis_clean
                            
                            # Create density scatter plot
                            try:
                                from scipy.stats import gaussian_kde
                                xy = np.vstack([alpha_seis, seis_for_alpha])
                                kde = gaussian_kde(xy)(xy)
                                idx = kde.argsort()
                                x, y, z = alpha_seis[idx], seis_for_alpha[idx], kde[idx]
                                sc = axes_corr[0,1].scatter(x, y, c=z, s=20, cmap='cividis', alpha=0.6)
                                # Add colorbar
                                cbar = plt.colorbar(sc, ax=axes_corr[0,1], shrink=0.8)
                                cbar.set_label('Density')
                            except:
                                # Fallback to regular scatter
                                axes_corr[0,1].scatter(alpha_seis, seis_for_alpha, alpha=0.3, s=10, color='green')
                            
                            corr_as, p_as = pearsonr(alpha_seis, seis_for_alpha)
                            axes_corr[0,1].set_xlabel('Alpha (Coulomb)')
                            axes_corr[0,1].set_ylabel('Seismicity Count (90-day)')
                            axes_corr[0,1].set_title(f'Alpha vs Seismicity\\nr = {corr_as:.3f} (p = {p_as:.3e})')
                            axes_corr[0,1].grid(True, alpha=0.3)
                        else:
                            axes_corr[0,1].text(0.5, 0.5, 'Insufficient data\\nfor Alpha-Seismicity', ha='center', va='center')
                    else:
                        axes_corr[0,1].text(0.5, 0.5, 'Alpha data\\nunavailable', ha='center', va='center')
                else:
                    axes_corr[0,1].text(0.5, 0.5, 'Data unavailable\\nfor Alpha-Seismicity', ha='center', va='center')

                # Scatter plot: Tidal Variability vs Seismicity (replace with density)
                # axes_corr[1,0].scatter(var_clean, seis_clean, alpha=0.6, s=20, color='orange')
                # Use seaborn-style density plot if available, otherwise use matplotlib hist2d
                try:
                    from scipy.stats import gaussian_kde
                    # Create kernel density estimate
                    xy = np.vstack([var_clean, seis_clean])
                    kde = gaussian_kde(xy)(xy)
                    # Sort by density for better visualization
                    idx = kde.argsort()
                    x, y, z = var_clean[idx], seis_clean[idx], kde[idx]
                    axes_corr[1,0].scatter(x, y, c=z, s=20, cmap='viridis', alpha=0.6)
                except:
                    # Fallback to regular scatter with transparency
                    axes_corr[1,0].scatter(var_clean, seis_clean, alpha=0.3, s=10, color='orange')

                corr_vs, p_vs = pearsonr(var_clean, seis_clean)
                axes_corr[1,0].set_xlabel('Tidal Variability (Pa)')
                axes_corr[1,0].set_ylabel('Seismicity Count (90-day)')
                axes_corr[1,0].set_title(f'Tidal Variability vs Seismicity\\nr = {corr_vs:.3f} (p = {p_vs:.3e})')
                axes_corr[1,0].grid(True, alpha=0.3)

                # Scatter plot: Tidal vs Tidal Variability (replace with density)
                # axes_corr[1,1].scatter(tidal_clean, var_clean, alpha=0.6, s=20, color='purple')
                try:
                    from scipy.stats import gaussian_kde
                    # Create kernel density estimate
                    xy = np.vstack([tidal_clean, var_clean])
                    kde = gaussian_kde(xy)(xy)
                    # Sort by density for better visualization
                    idx = kde.argsort()
                    x, y, z = tidal_clean[idx], var_clean[idx], kde[idx]
                    axes_corr[1,1].scatter(x, y, c=z, s=20, cmap='plasma', alpha=0.6)
                except:
                    # Fallback to regular scatter with transparency
                    axes_corr[1,1].scatter(tidal_clean, var_clean, alpha=0.3, s=10, color='purple')

                corr_tv, p_tv = pearsonr(tidal_clean, var_clean)
                axes_corr[1,1].set_xlabel('Tidal Coulomb Stress (Pa)')
                axes_corr[1,1].set_ylabel('Tidal Variability (Pa)')
                axes_corr[1,1].set_title(f'Tidal Stress vs Variability\\nr = {corr_tv:.3f} (p = {p_tv:.3e})')
                axes_corr[1,1].grid(True, alpha=0.3)
                axes_corr[1,1].set_xlim(-200, 200)

                # Bottom-right: GRACE vs Seismicity correlation (density plot)
                if len(grace_clean) > 10 and len(seis_clean) > 10:
                    # Create density scatter plot
                    try:
                        from scipy.stats import gaussian_kde
                        xy = np.vstack([grace_clean, seis_clean])
                        kde = gaussian_kde(xy)(xy)
                        idx = kde.argsort()
                        x, y, z = grace_clean[idx], seis_clean[idx], kde[idx]
                        sc = axes_corr[1,2].scatter(x, y, c=z, s=20, cmap='inferno', alpha=0.6)
                        # Add colorbar
                        cbar = plt.colorbar(sc, ax=axes_corr[1,2], shrink=0.8)
                        cbar.set_label('Density')
                    except:
                        # Fallback to regular scatter
                        axes_corr[1,2].scatter(grace_clean, seis_clean, alpha=0.3, s=10, color='red')
                    
                    corr_gs, p_gs = pearsonr(grace_clean, seis_clean)
                    axes_corr[1,2].set_xlabel('GRACE Coulomb Stress (Pa)')
                    axes_corr[1,2].set_ylabel('Seismicity Count (90-day)')
                    axes_corr[1,2].set_title(f'GRACE vs Seismicity\\nr = {corr_gs:.3f} (p = {p_gs:.3e})')
                    axes_corr[1,2].grid(True, alpha=0.3)
                else:
                    axes_corr[1,2].text(0.5, 0.5, 'Insufficient data\\nfor GRACE-Seismicity', ha='center', va='center')

                # Coherence analysis (on axes_corr[0,2])
                f_tg, coh_tg = compute_coherence(tidal_clean, grace_clean, fs=1.0)  # daily sampling
                f_ts, coh_ts = compute_coherence(tidal_clean, seis_clean, fs=1.0)
                f_gs, coh_gs = compute_coherence(grace_clean, seis_clean, fs=1.0)
                f_vs, coh_vs = compute_coherence(var_clean, seis_clean, fs=1.0)

                if len(f_tg) > 0:
                    axes_corr[0,2].plot(f_tg, coh_tg, 'k-', label='Tidal-GRACE', linewidth=2)
                if len(f_ts) > 0:
                    axes_corr[0,2].plot(f_ts, coh_ts, 'b-', label='Tidal-Seismicity', linewidth=2)
                if len(f_gs) > 0:
                    axes_corr[0,2].plot(f_gs, coh_gs, 'r-', label='GRACE-Seismicity', linewidth=2)
                if len(f_vs) > 0:
                    axes_corr[0,2].plot(f_vs, coh_vs, 'orange', linestyle='-', label='TidalVar-Seismicity', linewidth=2)

                axes_corr[0,2].set_xlabel('Frequency (cycles/day)')
                axes_corr[0,2].set_ylabel('Coherence')
                axes_corr[0,2].set_title('Signal Coherence')
                axes_corr[0,2].set_xlim([0, 0.5])  # Focus on low frequencies
                axes_corr[0,2].grid(True, alpha=0.3)
                axes_corr[0,2].legend()

                # Add frequency band indicators
                axes_corr[0,2].axvspan(1/365, 1/30, alpha=0.2, color='yellow', label='Seasonal')  # Seasonal band
                axes_corr[0,2].axvspan(0.5, 2.0, alpha=0.2, color='cyan', label='Tidal')  # Tidal band (but mostly outside our freq range)

                # Add colorbars for density plots
                try:
                    # Colorbar for variability-seismicity density plot
                    sm1 = plt.cm.ScalarMappable(cmap='viridis')
                    sm1.set_array([])
                    cbar1 = plt.colorbar(sm1, ax=axes_corr[1,0], shrink=0.8)
                    cbar1.set_label('Density')

                    # Colorbar for tidal-variability density plot
                    sm2 = plt.cm.ScalarMappable(cmap='plasma')
                    sm2.set_array([])
                    cbar2 = plt.colorbar(sm2, ax=axes_corr[1,1], shrink=0.8)
                    cbar2.set_label('Density')

                    # Colorbar for GRACE-seismicity density plot
                    sm3 = plt.cm.ScalarMappable(cmap='inferno')
                    sm3.set_array([])
                    cbar3 = plt.colorbar(sm3, ax=axes_corr[1,2], shrink=0.8)
                    cbar3.set_label('Density')
                except:
                    pass  # Skip colorbars if they fail

            plt.tight_layout()
            plt.savefig(f'{output_prefix}_correlations.png', dpi=300, bbox_inches='tight')
            plt.close(fig_corr)
        except Exception as e:
            print(f"Error creating correlations plot: {e}")    # Alpha-GRACE correlation analysis with bandpass filtering
    corr_ag, p_ag = None, None
    alpha_grace_description = "Alpha-GRACE correlation: not available"
    
    if alpha_df is not None and not alpha_df.empty:
        try:
            # Get alpha values
            alpha_vals = alpha_df['alpha_coulomb'].values if 'alpha_coulomb' in alpha_df.columns else (alpha_df['alpha'].values if 'alpha' in alpha_df.columns else None)
            
            if alpha_vals is not None:
                # Create a pandas Series for alpha for easier filtering
                alpha_series = pd.Series(alpha_vals, index=pd.to_datetime(alpha_df['time']))

                # Use the globally computed filtered alpha
                if alpha_filtered_series is not None:
                    # Get the filtered values back onto the original, irregular alpha timestamps
                    alpha_filtered_reindexed = alpha_filtered_series.reindex(alpha_series.index, method='nearest')
                    alpha_filtered = alpha_filtered_reindexed.values
                else:
                    alpha_filtered_reindexed = None
                    alpha_filtered = None

                # Interpolate GRACE Coulomb stress to alpha time points
                alpha_times = alpha_df['time'].values
                grace_daily_for_corr = downsample_for_comparison(grace_coulomb_data, time_grid_data, time_grid_daily)
                grace_at_alpha = np.interp(
                    (alpha_times - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'D'),
                    (time_grid_daily - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'D'),
                    grace_daily_for_corr
                )
                
                # Clean original data for correlation
                valid_alpha_grace = ~(np.isnan(alpha_vals) | np.isnan(grace_at_alpha))
                if np.sum(valid_alpha_grace) > 10:
                    alpha_clean = alpha_vals[valid_alpha_grace]
                    grace_at_alpha_clean = grace_at_alpha[valid_alpha_grace]
                    corr_ag, p_ag = pearsonr(alpha_clean, grace_at_alpha_clean)
                    alpha_grace_description = f"Alpha-GRACE correlation (unfiltered): r = {corr_ag:.3f} (p = {p_ag:.3e}), n = {len(alpha_clean)} points"
                    print(f"\n=== ALPHA-GRACE CORRELATION (UNFILTERED) ===")
                    print(alpha_grace_description)

                # Clean filtered data for correlation
                if alpha_filtered_reindexed is not None:
                    valid_filtered = ~(np.isnan(alpha_filtered_reindexed.values) | np.isnan(grace_at_alpha))
                    if np.sum(valid_filtered) > 10:
                        alpha_filtered_clean = alpha_filtered_reindexed.values[valid_filtered]
                        grace_at_alpha_filtered_clean = grace_at_alpha[valid_filtered]
                        corr_ag_filtered, p_ag_filtered = pearsonr(alpha_filtered_clean, grace_at_alpha_filtered_clean)
                        filtered_description = f"Alpha-GRACE correlation (6-24 month bandpass): r = {corr_ag_filtered:.3f} (p = {p_ag_filtered:.3e}), n = {len(alpha_filtered_clean)} points"
                        print(f"\n=== ALPHA-GRACE CORRELATION (FILTERED) ===")
                        print(filtered_description)

                        # Create a new plot for the filtered comparison
                        fig_filtered, ax1_f = plt.subplots(1, 1, figsize=(15, 6))
                        fig_filtered.suptitle('Filtered Alpha (6-24mo) vs GRACE Coulomb Stress', fontsize=14, fontweight='bold')
                        
                        # Plot filtered alpha
                        line1_f = ax1_f.plot(alpha_filtered_series.index, alpha_filtered_series.values, color='c', linewidth=1.5, label='Filtered Alpha (6-24mo)')
                        ax1_f.set_xlabel('Time', fontsize=12)
                        ax1_f.set_ylabel('Filtered Alpha', fontsize=12, color='c')
                        ax1_f.tick_params(axis='y', labelcolor='c')
                        ax1_f.grid(True, alpha=0.3)

                        # Create secondary y-axis for GRACE Coulomb stress
                        ax2_f = ax1_f.twinx()
                        
                        # Plot GRACE data on the secondary axis
                        grace_daily_for_plot = downsample_for_comparison(grace_coulomb_data, time_grid_data, time_grid_daily)
                        valid_grace_plot = ~np.isnan(grace_daily_for_plot)
                        
                        line2_f = ax2_f.plot(time_grid_daily[valid_grace_plot], grace_daily_for_plot[valid_grace_plot], 
                                       color='red', linewidth=1.0, alpha=0.7, label='GRACE Coulomb Stress')
                        ax2_f.set_ylabel('GRACE Coulomb Stress (Pa)', fontsize=12, color='red')
                        ax2_f.tick_params(axis='y', labelcolor='red')
                        
                        # Add combined legend
                        lines_f = line1_f + line2_f
                        labels_f = [l.get_label() for l in lines_f]
                        ax1_f.legend(lines_f, labels_f, loc='upper left')

                        plt.tight_layout()
                        plt.savefig(f'{output_prefix}_filtered_alpha_comparison.png', dpi=300, bbox_inches='tight')
                        plt.close(fig_filtered)
                        print(f"Saved filtered alpha comparison plot: {output_prefix}_filtered_alpha_comparison.png")
                else:
                    print("Filtered alpha not available for plotting")

        except Exception as e:
            print(f"Error during alpha-GRACE correlation or filtering: {e}")

    # Final summary printout
    print("\n\n" + "="*30)
    print("      FINAL CORRELATION SUMMARY")
    print("="*30)
    if 'corr_gs' in locals():
        print(f"GRACE vs Seismicity: r = {corr_gs:.3f} (p = {p_gs:.3e})")
    if 'corr_vs' in locals():
        print(f"Tidal Variability vs Seismicity: r = {corr_vs:.3f} (p = {p_vs:.3e})")
    print(alpha_grace_description)
    if 'filtered_description' in locals():
        print(filtered_description)
    print("="*30)

    # --- NEW FIGURE: Monthly normalized comparison of Seismicity, Alpha, GRACE ---
    try:
        print('\nCreating monthly normalized comparison figure...')

        # Create a monthly datetime index spanning the overlap period.
        # Use to_period('M').to_timestamp() to get month-start at midnight
        start_month = pd.to_datetime(time_grid_daily.min()).to_period('M').to_timestamp()
        end_month = pd.to_datetime(time_grid_daily.max()).to_period('M').to_timestamp()
        monthly_index = pd.date_range(start=start_month, end=end_month, freq='MS')

        # Monthly observed seismicity: count events per-month
        cat_times = pd.to_datetime(catalog['time'])
        cat_df = pd.DataFrame({'time': cat_times})
        cat_df['month_start'] = pd.to_datetime(cat_df['time'].dt.to_period('M').dt.to_timestamp())
        monthly_counts = cat_df.groupby('month_start').size().reindex(monthly_index, fill_value=0)

        # Monthly alpha: compute monthly mean from alpha_df if available
        monthly_alpha = pd.Series(index=monthly_index, dtype=float)
        if alpha_df is not None and not alpha_df.empty:
            alpha_series = pd.Series(alpha_df['alpha_coulomb'].values if 'alpha_coulomb' in alpha_df.columns else (alpha_df['alpha'].values if 'alpha' in alpha_df.columns else None),
                                      index=pd.to_datetime(alpha_df['time']))
            alpha_monthly = alpha_series.resample('MS').mean()
            monthly_alpha = alpha_monthly.reindex(monthly_index)

        # Monthly GRACE coulomb: use daily downsampled grace_daily and aggregate to month
        grace_daily = downsample_for_comparison(grace_coulomb_data, time_grid_data, time_grid_daily)
        grace_daily_dates = pd.DatetimeIndex(time_grid_daily)
        grace_df_monthly = pd.DataFrame({'time': grace_daily_dates, 'grace': grace_daily}).set_index('time')
        grace_monthly_mean = grace_df_monthly['grace'].resample('MS').mean().reindex(monthly_index)

        # Combine into a DataFrame
        monthly_df = pd.DataFrame({
            'time': monthly_index,
            'seismicity_count': monthly_counts.values,
            'alpha_monthly': monthly_alpha.values,
            'grace_monthly': grace_monthly_mean.values
        })

        # Normalize each series (z-score) where possible
        def zscore_safe(x):
            x = np.array(x, dtype=float)
            mask = np.isfinite(x)
            res = np.full_like(x, np.nan)
            if mask.sum() > 1 and np.nanstd(x[mask]) > 0:
                res[mask] = (x[mask] - np.nanmean(x[mask])) / np.nanstd(x[mask])
            elif mask.sum() > 0:
                res[mask] = x[mask] - np.nanmean(x[mask])
            return res

        monthly_df['seis_norm'] = zscore_safe(monthly_df['seismicity_count'].values)
        monthly_df['alpha_norm'] = zscore_safe(monthly_df['alpha_monthly'].values)
        monthly_df['grace_norm'] = zscore_safe(monthly_df['grace_monthly'].values)

        # Save CSV for external use
        out_csv = f"{output_prefix}_monthly_comparison.csv"
        monthly_df.to_csv(out_csv, index=False)
        print(f"Saved monthly comparison CSV: {out_csv}")

        # Plot the monthly normalized time series
        fig_m, axm = plt.subplots(1, 1, figsize=(14, 5))
        axm.plot(monthly_df['time'], monthly_df['seis_norm'], '-o', color='tab:green', label='Seismicity (monthly, norm)')
        axm.plot(monthly_df['time'], monthly_df['alpha_norm'], '-s', color='magenta', label='Alpha (monthly, norm)')
        axm.plot(monthly_df['time'], monthly_df['grace_norm'], '-^', color='red', label='GRACE Coulomb (monthly, norm)')

        axm.set_xlabel('Time')
        axm.set_ylabel('Normalized (z-score)')
        axm.set_title('Monthly Normalized Comparison: Seismicity, Alpha, GRACE')
        axm.grid(True, alpha=0.3)
        axm.legend(loc='upper right')
        axm.xaxis.set_major_locator(mdates.YearLocator())
        axm.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.tight_layout()
        out_png = f"{output_prefix}_monthly_comparison.png"
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close(fig_m)
        print(f"Saved monthly comparison figure: {out_png}")

    except Exception as e:
        print('Could not create monthly comparison figure:', e)

    # --- NEW FIGURE: Weekly normalized comparison of Seismicity, Alpha, GRACE ---
    try:
        print('\nCreating weekly normalized comparison figure...')

        # Create a weekly datetime index spanning the overlap period.
        # Use to_period('W').to_timestamp() to get week-start (Monday)
        start_week = pd.to_datetime(time_grid_daily.min()).to_period('W').to_timestamp()
        end_week = pd.to_datetime(time_grid_daily.max()).to_period('W').to_timestamp()
        weekly_index = pd.date_range(start=start_week, end=end_week, freq='W-MON')

        # Weekly observed seismicity: count events per-week
        cat_times = pd.to_datetime(catalog['time'])
        cat_df = pd.DataFrame({'time': cat_times})
        cat_df['week_start'] = pd.to_datetime(cat_df['time'].dt.to_period('W').dt.to_timestamp())
        weekly_counts = cat_df.groupby('week_start').size().reindex(weekly_index, fill_value=0)

        # Weekly alpha: compute weekly mean from alpha_df if available
        weekly_alpha = pd.Series(index=weekly_index, dtype=float)
        if alpha_df is not None and not alpha_df.empty:
            alpha_series = pd.Series(alpha_df['alpha_coulomb'].values if 'alpha_coulomb' in alpha_df.columns else (alpha_df['alpha'].values if 'alpha' in alpha_df.columns else None),
                                      index=pd.to_datetime(alpha_df['time']))
            alpha_weekly = alpha_series.resample('W-MON').mean()
            weekly_alpha = alpha_weekly.reindex(weekly_index)

        # Weekly GRACE coulomb: use daily downsampled grace_daily and aggregate to week
        grace_daily = downsample_for_comparison(grace_coulomb_data, time_grid_data, time_grid_daily)
        grace_daily_dates = pd.DatetimeIndex(time_grid_daily)
        grace_df_weekly = pd.DataFrame({'time': grace_daily_dates, 'grace': grace_daily}).set_index('time')
        grace_weekly_mean = grace_df_weekly['grace'].resample('W-MON').mean().reindex(weekly_index)

        # Combine into a DataFrame
        weekly_df = pd.DataFrame({
            'time': weekly_index,
            'seismicity_count': weekly_counts.values,
            'alpha_weekly': weekly_alpha.values,
            'grace_weekly': grace_weekly_mean.values
        })

        # Normalize each series (z-score) where possible
        def zscore_safe(x):
            x = np.array(x, dtype=float)
            mask = np.isfinite(x)
            res = np.full_like(x, np.nan)
            if mask.sum() > 1 and np.nanstd(x[mask]) > 0:
                res[mask] = (x[mask] - np.nanmean(x[mask])) / np.nanstd(x[mask])
            elif mask.sum() > 0:
                res[mask] = x[mask] - np.nanmean(x[mask])
            return res

        weekly_df['seis_norm'] = zscore_safe(weekly_df['seismicity_count'].values)
        weekly_df['alpha_norm'] = zscore_safe(weekly_df['alpha_weekly'].values)
        weekly_df['grace_norm'] = zscore_safe(weekly_df['grace_weekly'].values)

        # Save CSV for external use
        out_csv = f"{output_prefix}_weekly_comparison.csv"
        weekly_df.to_csv(out_csv, index=False)
        print(f"Saved weekly comparison CSV: {out_csv}")

        # Plot the weekly normalized time series
        fig_w, axw = plt.subplots(1, 1, figsize=(14, 5))
        axw.plot(weekly_df['time'], weekly_df['seis_norm'], '-', color='tab:green', linewidth=0.8, alpha=0.7, label='Seismicity (weekly, norm)')
        axw.plot(weekly_df['time'], weekly_df['alpha_norm'], '-', color='magenta', linewidth=1.2, label='Alpha (weekly, norm)')
        axw.plot(weekly_df['time'], weekly_df['grace_norm'], '-^', color='red', markersize=3, linewidth=1.0, alpha=0.9, label='GRACE Coulomb (weekly, norm)')

        axw.set_xlabel('Time')
        axw.set_ylabel('Normalized (z-score)')
        axw.set_title('Weekly Normalized Comparison: Seismicity, Alpha, GRACE')
        axw.grid(True, alpha=0.3)
        axw.legend(loc='upper right')
        axw.xaxis.set_major_locator(mdates.YearLocator())
        axw.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.tight_layout()
        out_png = f"{output_prefix}_weekly_comparison.png"
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close(fig_w)
        print(f"Saved weekly comparison figure: {out_png}")

    except Exception as e:
        print('Could not create weekly comparison figure:', e)

    # --- NEW: Month-of-year (seasonal cycle) with bootstrap CIs and absolute amplitudes ---
    try:
        print('\nCreating month-of-year average (seasonal cycle) figure with bootstrap CIs...')

        # Reconstruct monthly_df if needed
        if 'monthly_df' not in locals():
            start_month = pd.to_datetime(time_grid_daily.min()).to_period('M').to_timestamp()
            end_month = pd.to_datetime(time_grid_daily.max()).to_period('M').to_timestamp()
            monthly_index = pd.date_range(start=start_month, end=end_month, freq='MS')
            cat_times = pd.to_datetime(catalog['time'])
            cat_df = pd.DataFrame({'time': cat_times})
            cat_df['month_start'] = pd.to_datetime(cat_df['time'].dt.to_period('M').dt.to_timestamp())
            monthly_counts = cat_df.groupby('month_start').size().reindex(monthly_index, fill_value=0)
        monthly_alpha = pd.Series(index=monthly_index, dtype=float)
        if alpha_df is not None and not alpha_df.empty:
            alpha_series = pd.Series(alpha_df['alpha_coulomb'].values if 'alpha_coulomb' in alpha_df.columns else (alpha_df['alpha'].values if 'alpha' in alpha_df.columns else None), index=pd.to_datetime(alpha_df['time']))
            monthly_alpha = alpha_series.resample('MS').mean().reindex(monthly_index)
            grace_daily = downsample_for_comparison(grace_coulomb_data, time_grid_data, time_grid_daily)
            grace_dates = pd.DatetimeIndex(time_grid_daily)
            grace_df_monthly = pd.DataFrame({'time': grace_dates, 'grace': grace_daily}).set_index('time')
            grace_monthly_mean = grace_df_monthly['grace'].resample('MS').mean().reindex(monthly_index)
            monthly_df = pd.DataFrame({'time': monthly_index, 'seismicity_count': monthly_counts.values, 'alpha_monthly': monthly_alpha.values, 'grace_monthly': grace_monthly_mean.values})

        monthly_df['time'] = pd.to_datetime(monthly_df['time'])
        monthly_df['month'] = monthly_df['time'].dt.month

        months = np.arange(1,13)

        # helper to get per-month arrays across years
        def month_arrays(col):
            arrs = []
            for m in months:
                vals = monthly_df.loc[monthly_df['month'] == m, col].dropna().values.astype(float)
                arrs.append(vals)
            return arrs

        seiz_arrs = month_arrays('seismicity_count')
        alpha_arrs = month_arrays('alpha_monthly')
        grace_arrs = month_arrays('grace_monthly')

        # Compute observed monthly means (raw)
        seiz_mean = np.array([np.nan if len(a)==0 else np.nanmean(a) for a in seiz_arrs])
        alpha_mean = np.array([np.nan if len(a)==0 else np.nanmean(a) for a in alpha_arrs])
        grace_mean = np.array([np.nan if len(a)==0 else np.nanmean(a) for a in grace_arrs])

        # Bootstrap raw means per month and normalized means across months
        def bootstrap_monthly(arrs, B=2000, seed=0):
            rng = np.random.default_rng(seed)
            raw_boot = np.full((B, 12), np.nan)
            for b in range(B):
                for i, vals in enumerate(arrs):
                    if len(vals) > 0:
                        res = rng.choice(vals, size=len(vals), replace=True)
                        raw_boot[b, i] = np.nanmean(res)
            # normalize each bootstrap sample across months
            norm_boot = np.full_like(raw_boot, np.nan)
            for b in range(B):
                arr = raw_boot[b, :]
                mask = np.isfinite(arr)
                if mask.sum() > 1 and np.nanstd(arr[mask]) > 0:
                    norm_boot[b, mask] = (arr[mask] - np.nanmean(arr[mask])) / np.nanstd(arr[mask])
                elif mask.sum() > 0:
                    norm_boot[b, mask] = arr[mask] - np.nanmean(arr[mask])
            return raw_boot, norm_boot

        B = 2000
        seiz_raw_boot, seiz_norm_boot = bootstrap_monthly(seiz_arrs, B=B, seed=1)
        alpha_raw_boot, alpha_norm_boot = bootstrap_monthly(alpha_arrs, B=B, seed=2)
        grace_raw_boot, grace_norm_boot = bootstrap_monthly(grace_arrs, B=B, seed=3)

        # Observed normalized means (z-score across months)
        def znorm(v):
            mask = np.isfinite(v)
            res = np.full_like(v, np.nan)
            if mask.sum() > 1 and np.nanstd(v[mask]) > 0:
                res[mask] = (v[mask] - np.nanmean(v[mask])) / np.nanstd(v[mask])
            elif mask.sum() > 0:
                res[mask] = v[mask] - np.nanmean(v[mask])
            return res

        seiz_mean_norm = znorm(seiz_mean)
        alpha_mean_norm = znorm(alpha_mean)
        grace_mean_norm = znorm(grace_mean)

        # Compute bootstrap CIs (2.5/97.5) for normalized and raw means
        def ci_from_boot(boot, q=(2.5,97.5)):
            lower = np.nanpercentile(boot, q[0], axis=0)
            upper = np.nanpercentile(boot, q[1], axis=0)
            return lower, upper

        seiz_norm_lo, seiz_norm_hi = ci_from_boot(seiz_norm_boot)
        alpha_norm_lo, alpha_norm_hi = ci_from_boot(alpha_norm_boot)
        grace_norm_lo, grace_norm_hi = ci_from_boot(grace_norm_boot)

        seiz_raw_lo, seiz_raw_hi = ci_from_boot(seiz_raw_boot)
        alpha_raw_lo, alpha_raw_hi = ci_from_boot(alpha_raw_boot)
        grace_raw_lo, grace_raw_hi = ci_from_boot(grace_raw_boot)

        # Save CSV with bootstrapped CIs and raw means
        cyc_df = pd.DataFrame({
            'month': months,
            'seiz_mean': seiz_mean,
            'seiz_raw_lo': seiz_raw_lo,
            'seiz_raw_hi': seiz_raw_hi,
            'seiz_mean_norm': seiz_mean_norm,
            'seiz_norm_lo': seiz_norm_lo,
            'seiz_norm_hi': seiz_norm_hi,
            'alpha_mean': alpha_mean,
            'alpha_raw_lo': alpha_raw_lo,
            'alpha_raw_hi': alpha_raw_hi,
            'alpha_mean_norm': alpha_mean_norm,
            'alpha_norm_lo': alpha_norm_lo,
            'alpha_norm_hi': alpha_norm_hi,
            'grace_mean': grace_mean,
            'grace_raw_lo': grace_raw_lo,
            'grace_raw_hi': grace_raw_hi,
            'grace_mean_norm': grace_mean_norm,
            'grace_norm_lo': grace_norm_lo,
            'grace_norm_hi': grace_norm_hi
        })

        cyc_csv = f"{output_prefix}_monthly_cycle.csv"
        cyc_df.to_csv(cyc_csv, index=False)
        print(f"Saved month-of-year cycle CSV with bootstrap CIs: {cyc_csv}")

        # Plot normalized means with bootstrap CIs and raw means on twin axis
        fig_c, axc = plt.subplots(1,1, figsize=(9,5))
        month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

        axc.plot(months, seiz_mean_norm, '-o', color='tab:green', label='Seismicity (mean, norm)')
        axc.fill_between(months, seiz_norm_lo, seiz_norm_hi, color='tab:green', alpha=0.2)

        axc.plot(months, alpha_mean_norm, '-s', color='magenta', label='Alpha (mean, norm)')
        axc.fill_between(months, alpha_norm_lo, alpha_norm_hi, color='magenta', alpha=0.15)

        axc.plot(months, grace_mean_norm, '-^', color='red', label='GRACE (mean, norm)')
        axc.fill_between(months, grace_norm_lo, grace_norm_hi, color='red', alpha=0.15)

        axc.set_xticks(months)
        axc.set_xticklabels(month_labels)
        axc.set_xlabel('Month of Year')
        axc.set_ylabel('Normalized mean (z-score across months)')
        axc.set_title('Seasonal Cycle (Month-of-Year Average)')
        axc.grid(True, alpha=0.3)

        # Keep the plot focused: remove raw-axis to reduce clutter. Only show normalized means + CIs.
        axc.legend(loc='upper left', fontsize=9)

        plt.tight_layout()
        cyc_png = f"{output_prefix}_monthly_cycle.png"
        plt.savefig(cyc_png, dpi=300, bbox_inches='tight')
        plt.close(fig_c)
        print(f"Saved month-of-year cycle figure with bootstrap CIs and raw-axis: {cyc_png}")

    except Exception as e:
        print('Could not create month-of-year cycle figure with bootstrap:', e)

    # --- NEW: Week-of-year (seasonal cycle) with bootstrap CIs ---
    try:
        print('\nCreating week-of-year average (seasonal cycle) figure with bootstrap CIs...')

        # Reconstruct weekly_df if needed
        if 'weekly_df' not in locals():
            start_week = pd.to_datetime(time_grid_daily.min()).to_period('W').to_timestamp()
            end_week = pd.to_datetime(time_grid_daily.max()).to_period('W').to_timestamp()
            weekly_index = pd.date_range(start=start_week, end=end_week, freq='W')
            cat_times = pd.to_datetime(catalog['time'])
            cat_df = pd.DataFrame({'time': cat_times})
            cat_df['week_start'] = pd.to_datetime(cat_df['time'].dt.to_period('W').dt.to_timestamp())
            weekly_counts = cat_df.groupby('week_start').size().reindex(weekly_index, fill_value=0)
            weekly_alpha = pd.Series(index=weekly_index, dtype=float)
            if alpha_df is not None and not alpha_df.empty:
                alpha_series = pd.Series(alpha_df['alpha_coulomb'].values if 'alpha_coulomb' in alpha_df.columns else (alpha_df['alpha'].values if 'alpha' in alpha_df.columns else None), index=pd.to_datetime(alpha_df['time']))
                weekly_alpha = alpha_series.resample('W').mean().reindex(weekly_index)
            grace_daily = downsample_for_comparison(grace_coulomb_data, time_grid_data, time_grid_daily)
            grace_dates = pd.DatetimeIndex(time_grid_daily)
            grace_df_weekly = pd.DataFrame({'time': grace_dates, 'grace': grace_daily}).set_index('time')
            grace_weekly_mean = grace_df_weekly['grace'].resample('W').mean().reindex(weekly_index)
            weekly_df = pd.DataFrame({'time': weekly_index, 'seismicity_count': weekly_counts.values, 'alpha_weekly': weekly_alpha.values, 'grace_weekly': grace_weekly_mean.values})

        weekly_df['time'] = pd.to_datetime(weekly_df['time'])
        weekly_df['week'] = weekly_df['time'].dt.isocalendar().week

        weeks = np.arange(1, 53)  # ISO weeks 1-52

        # helper to get per-week arrays across years
        def week_arrays(col):
            arrs = []
            for w in weeks:
                vals = weekly_df.loc[weekly_df['week'] == w, col].dropna().values.astype(float)
                arrs.append(vals)
            return arrs

        seiz_arrs = week_arrays('seismicity_count')
        alpha_arrs = week_arrays('alpha_weekly')
        grace_arrs = week_arrays('grace_weekly')

        # Compute observed weekly means (raw)
        seiz_mean = np.array([np.nan if len(a)==0 else np.nanmean(a) for a in seiz_arrs])
        alpha_mean = np.array([np.nan if len(a)==0 else np.nanmean(a) for a in alpha_arrs])
        grace_mean = np.array([np.nan if len(a)==0 else np.nanmean(a) for a in grace_arrs])

        # Bootstrap raw means per week and normalized means across weeks
        def bootstrap_weekly(arrs, B=2000, seed=0):
            rng = np.random.default_rng(seed)
            raw_boot = np.full((B, 52), np.nan)
            for b in range(B):
                for i, vals in enumerate(arrs):
                    if len(vals) > 0:
                        res = rng.choice(vals, size=len(vals), replace=True)
                        raw_boot[b, i] = np.nanmean(res)
            # normalize each bootstrap sample across weeks
            norm_boot = np.full_like(raw_boot, np.nan)
            for b in range(B):
                arr = raw_boot[b, :]
                mask = np.isfinite(arr)
                if mask.sum() > 1 and np.nanstd(arr[mask]) > 0:
                    norm_boot[b, mask] = (arr[mask] - np.nanmean(arr[mask])) / np.nanstd(arr[mask])
                elif mask.sum() > 0:
                    norm_boot[b, mask] = arr[mask] - np.nanmean(arr[mask])
            return raw_boot, norm_boot

        B = 2000
        seiz_raw_boot, seiz_norm_boot = bootstrap_weekly(seiz_arrs, B=B, seed=4)
        alpha_raw_boot, alpha_norm_boot = bootstrap_weekly(alpha_arrs, B=B, seed=5)
        grace_raw_boot, grace_norm_boot = bootstrap_weekly(grace_arrs, B=B, seed=6)

        # Observed normalized means (z-score across weeks)
        def znorm(v):
            mask = np.isfinite(v)
            res = np.full_like(v, np.nan)
            if mask.sum() > 1 and np.nanstd(v[mask]) > 0:
                res[mask] = (v[mask] - np.nanmean(v[mask])) / np.nanstd(v[mask])
            elif mask.sum() > 0:
                res[mask] = v[mask] - np.nanmean(v[mask])
            return res

        seiz_mean_norm = znorm(seiz_mean)
        alpha_mean_norm = znorm(alpha_mean)
        grace_mean_norm = znorm(grace_mean)

        # Compute bootstrap CIs (2.5/97.5) for normalized and raw means
        def ci_from_boot(boot, q=(2.5,97.5)):
            lower = np.nanpercentile(boot, q[0], axis=0)
            upper = np.nanpercentile(boot, q[1], axis=0)
            return lower, upper

        seiz_norm_lo, seiz_norm_hi = ci_from_boot(seiz_norm_boot)
        alpha_norm_lo, alpha_norm_hi = ci_from_boot(alpha_norm_boot)
        grace_norm_lo, grace_norm_hi = ci_from_boot(grace_norm_boot)

        seiz_raw_lo, seiz_raw_hi = ci_from_boot(seiz_raw_boot)
        alpha_raw_lo, alpha_raw_hi = ci_from_boot(alpha_raw_boot)
        grace_raw_lo, grace_raw_hi = ci_from_boot(grace_raw_boot)

        # Save CSV with bootstrapped CIs and raw means
        week_cyc_df = pd.DataFrame({
            'week': weeks,
            'seiz_mean': seiz_mean,
            'seiz_raw_lo': seiz_raw_lo,
            'seiz_raw_hi': seiz_raw_hi,
            'seiz_mean_norm': seiz_mean_norm,
            'seiz_norm_lo': seiz_norm_lo,
            'seiz_norm_hi': seiz_norm_hi,
            'alpha_mean': alpha_mean,
            'alpha_raw_lo': alpha_raw_lo,
            'alpha_raw_hi': alpha_raw_hi,
            'alpha_mean_norm': alpha_mean_norm,
            'alpha_norm_lo': alpha_norm_lo,
            'alpha_norm_hi': alpha_norm_hi,
            'grace_mean': grace_mean,
            'grace_raw_lo': grace_raw_lo,
            'grace_raw_hi': grace_raw_hi,
            'grace_mean_norm': grace_mean_norm,
            'grace_norm_lo': grace_norm_lo,
            'grace_norm_hi': grace_norm_hi
        })

        week_cyc_csv = f"{output_prefix}_weekly_cycle.csv"
        week_cyc_df.to_csv(week_cyc_csv, index=False)
        print(f"Saved week-of-year cycle CSV with bootstrap CIs: {week_cyc_csv}")

        # Plot normalized means with bootstrap CIs
        fig_wc, axwc = plt.subplots(1,1, figsize=(14,6))

        axwc.plot(weeks, seiz_mean_norm, '-', color='tab:green', linewidth=1.5, alpha=0.8, label='Seismicity (mean, norm)')
        axwc.fill_between(weeks, seiz_norm_lo, seiz_norm_hi, color='tab:green', alpha=0.2)

        axwc.plot(weeks, alpha_mean_norm, '-', color='magenta', linewidth=2, label='Alpha (mean, norm)')
        axwc.fill_between(weeks, alpha_norm_lo, alpha_norm_hi, color='magenta', alpha=0.15)

        axwc.plot(weeks, grace_mean_norm, '-', color='red', linewidth=1.5, alpha=0.9, label='GRACE (mean, norm)')
        axwc.fill_between(weeks, grace_norm_lo, grace_norm_hi, color='red', alpha=0.15)

        axwc.set_xticks(np.arange(1, 53, 4))  # Show every 4th week
        axwc.set_xlabel('Week of Year')
        axwc.set_ylabel('Normalized mean (z-score across weeks)')
        axwc.set_title('Seasonal Cycle (Week-of-Year Average)')
        axwc.grid(True, alpha=0.3)
        axwc.legend(loc='upper left', fontsize=10)

        plt.tight_layout()
        week_cyc_png = f"{output_prefix}_weekly_cycle.png"
        plt.savefig(week_cyc_png, dpi=300, bbox_inches='tight')
        plt.close(fig_wc)
        print(f"Saved week-of-year cycle figure with bootstrap CIs: {week_cyc_png}")

    except Exception as e:
        print('Could not create week-of-year cycle figure with bootstrap:', e)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=str, required=True,
                        help='Path to the earthquake catalog CSV file.')
    parser.add_argument('--spotl-output', type=str, required=True,
                        help='Path to the SPOTL strain output CSV file.')
    parser.add_argument('--grace-file', type=str, required=True,
                        help='Path to the GRACE NetCDF file.')
    parser.add_argument('--projection', type=str, default='fault_plane',
                        choices=['fault_plane', 'volumetric', 'areal'],
                        help='Projection mode for stress calculation.')
    parser.add_argument('--strike', type=float, default=None,
                        help='Fault strike in degrees (required for fault_plane).')
    parser.add_argument('--dip', type=float, default=None,
                        help='Fault dip in degrees (required for fault_plane).')
    parser.add_argument('--rake', type=float, default=None,
                        help='Fault rake in degrees (required for fault_plane).')
    parser.add_argument('--grace-lon-min', type=float, default=-92,
                        help='Minimum longitude for GRACE data region.')
    parser.add_argument('--grace-lon-max', type=float, default=-88,
                        help='Maximum longitude for GRACE data region.')
    parser.add_argument('--grace-lat-min', type=float, default=34,
                        help='Minimum latitude for GRACE data region.')
    parser.add_argument('--grace-lat-max', type=float, default=38,
                        help='Maximum latitude for GRACE data region.')
    parser.add_argument('--output-prefix', type=str, default='coherence_test',
                        help='Prefix for output plot files.')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations for alpha significance.')
    parser.add_argument('--output-alpha-timeseries', type=str, default=None,
                        help='Path to save alpha time series CSV file (optional).')

    args = parser.parse_args()

    try:
        analyze_signal_coherence(
            catalog_path=args.catalog,
            spotl_path=args.spotl_output,
            grace_file=args.grace_file,
            strike=args.strike,
            dip=args.dip,
            rake=args.rake,
            grace_lon_min=args.grace_lon_min,
            grace_lon_max=args.grace_lon_max,
            grace_lat_min=args.grace_lat_min,
            grace_lat_max=args.grace_lat_max,
            output_prefix=args.output_prefix,
            projection=args.projection,
            n_boot=args.n_bootstrap,
            output_alpha_timeseries=args.output_alpha_timeseries
        )

    except Exception as e:
        print(f"Error in script: {e}")

