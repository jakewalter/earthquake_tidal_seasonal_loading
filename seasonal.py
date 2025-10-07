import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy import signal
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

class SeasonalSeismicityAnalyzer:
    """
    Reproduces the seasonal seismicity analysis from Craig et al. (2017)
    for the New Madrid Seismic Zone study
    """
    
    def __init__(self, earthquake_csv, grace_csv=None):
        """
        Initialize the analyzer with earthquake and optional GRACE data
        
        Parameters:
        earthquake_csv: str, path to CSV with columns: 'datetime', 'magnitude', 'latitude', 'longitude'
        grace_csv: str, optional path to CSV with columns: 'datetime', 'equivalent_water_height'
        """
        self.load_earthquake_data(earthquake_csv)
        if grace_csv:
            self.load_grace_csv(grace_csv)
        else:
            self.grace_data = None
            
    def load_earthquake_data(self, csv_path):
        """Load and prepare earthquake catalog data using nm_tidal_seasonal logic"""
        df = pd.read_csv(csv_path)
        # map likely column names
        if 'origin_time' in df.columns:
            df = df.rename(columns={'origin_time':'time'})
        if 'Ml' in df.columns:
            df = df.rename(columns={'Ml':'magnitude'})
        if 'time' not in df.columns:
            raise ValueError("Catalog missing 'time' or 'origin_time' column.")
        df['time'] = pd.to_datetime(df['time'])
        # Apply basic quality filters (remove hardcoded date range for flexibility)
        mask = pd.Series(True, index=df.index)  # Start with all events
        if 'magnitude' in df.columns:
            mask &= (df['magnitude'] >= 0.0)
        if 'depth' in df.columns:
            mask &= (df['depth'] <= 20.0)
        df = df[mask].copy()
        print(f"Loaded catalog: {len(df)} events from {df['time'].min()} to {df['time'].max()}")
        
        # Convert to expected format for seasonal analysis
        df = df.rename(columns={'time': 'datetime'})
        self.eq_data = df.sort_values('datetime').reset_index(drop=True)
        
        # Add derived time columns
        self.eq_data['year'] = self.eq_data['datetime'].dt.year
        self.eq_data['month'] = self.eq_data['datetime'].dt.month
        self.eq_data['day_of_year'] = self.eq_data['datetime'].dt.dayofyear
        print(f"Magnitude range: {self.eq_data['magnitude'].min():.1f} to {self.eq_data['magnitude'].max():.1f}")
        
    def load_grace_csv(self, csv_path):
        # Load GRACE CSV, ensure time is datetime
        df = pd.read_csv(csv_path)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        elif 'datetime' in df.columns:
            df['time'] = pd.to_datetime(df['datetime'])
        else:
            raise ValueError("GRACE CSV missing 'time' or 'datetime' column.")
        self.grace_data = df
        print(f"Loaded GRACE data: {len(df)} records from {df['time'].min()} to {df['time'].max()}")
        
    def calculate_completeness_magnitude(self, magnitude_bins=0.1):
        """
        Set magnitude of completeness to 1.5 (based on New Madrid study)
        """
        self.completeness_magnitude = 1.5
        print(f"Using completeness magnitude: {self.completeness_magnitude:.1f}")
        return self.completeness_magnitude
        
    def gutenberg_richter_analysis(self, min_mag=None):
        """Perform Gutenberg-Richter analysis"""
        if min_mag is None:
            min_mag = self.completeness_magnitude
            
        # Filter data above completeness magnitude
        complete_data = self.eq_data[self.eq_data['magnitude'] >= min_mag]
        
        # Create magnitude bins
        mag_bins = np.arange(min_mag, self.eq_data['magnitude'].max() + 0.2, 0.1)
        counts, bin_edges = np.histogram(complete_data['magnitude'], bins=mag_bins)
        
        # Cumulative counts (N >= M)
        cumulative_counts = np.array([np.sum(counts[i:]) for i in range(len(counts))])
        
        # Fit linear relationship: log10(N) = a - b*M
        valid_idx = cumulative_counts > 10  # Require at least 10 events for fitting
        if np.sum(valid_idx) > 2:
            mags_for_fit = mag_bins[:-1][valid_idx]
            log_counts_for_fit = np.log10(cumulative_counts[valid_idx])
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(mags_for_fit, log_counts_for_fit)
            
            self.b_value = -slope
            self.a_value = intercept
            
            print(f"Gutenberg-Richter parameters: a={self.a_value:.2f}, b={self.b_value:.2f}, R²={r_value**2:.3f}")
        else:
            print("Warning: Insufficient data for reliable G-R analysis")
            self.b_value = 1.0  # Default value
            self.a_value = 3.0
            
    def decluster_catalog(self, use_declustered=True, time_window_days=None, distance_km=None):
        """
        Optimized Gardner-Knopoff declustering algorithm
        Removes dependent events (aftershocks and foreshocks) from catalog
        """
        if not use_declustered:
            self.eq_data_declustered = self.eq_data.copy()
            print(f"Declustering disabled. Using full catalog: {len(self.eq_data)} events")
            return self.eq_data_declustered
            
        if len(self.eq_data) == 0:
            self.eq_data_declustered = self.eq_data.copy()
            return self.eq_data_declustered
            
        # Gardner-Knopoff space-time windows as function of magnitude
        # Standard Gardner & Knopoff (1974) formulas
        def gk_time_window(mag):
            """Time window in days for Gardner-Knopoff declustering"""
            # Standard Gardner & Knopoff (1974) formula
            # T = 10^(0.032*M + 2.7389) seconds, converted to days
            return 10**(0.032*mag + 2.7389) / 86400
                
        def gk_distance_window(mag):
            """Distance window in km for Gardner-Knopoff declustering"""
            # Standard Gardner & Knopoff (1974) formula
            # D = 10^(0.1238*M + 0.983) km
            return 10**(0.1238*mag + 0.983)
        
        # Use fixed windows if provided, otherwise use Gardner-Knopoff
        use_adaptive_time = time_window_days is None
        use_adaptive_distance = distance_km is None
            
        # Start with all events as potential mainshocks
        data = self.eq_data.copy().sort_values('datetime').reset_index(drop=True)
        is_mainshock = np.ones(len(data), dtype=bool)
        
        print(f"Starting optimized declustering of {len(data)} events...")
        
        # Convert to numpy arrays for faster access
        times = data['datetime'].values
        mags = data['magnitude'].values
        
        # Pre-calculate coordinates if available
        if 'latitude' in data.columns and 'longitude' in data.columns:
            lats = np.radians(data['latitude'].values)
            lons = np.radians(data['longitude'].values)
            has_coords = True
        else:
            lats = lons = None
            has_coords = False
        
        # Optimized Gardner-Knopoff with smart lookback limits
        for i in range(len(data)):
            if not is_mainshock[i]:
                continue
                
            current_time = times[i]
            current_mag = mags[i]
            
            # Smart lookback: only check events within reasonable time window
            # Maximum possible time window is for largest earthquake (M~4.0)
            max_time_window_days = gk_time_window(4.0) if use_adaptive_time else (time_window_days or 100)
            
            # Find reasonable starting point (don't go back more than max time window)
            start_idx = 0
            for j in range(i-1, -1, -1):
                time_diff_days = (current_time - times[j]).astype('timedelta64[D]').astype(float)
                if time_diff_days > max_time_window_days:
                    start_idx = j + 1
                    break
            
            # Check backward from current event to start_idx
            for j in range(start_idx, i):
                if not is_mainshock[j]:
                    continue
                    
                earlier_mag = mags[j]
                
                # Only consider if earlier event is larger magnitude
                if earlier_mag <= current_mag:
                    continue
                
                # Time difference in days (faster numpy operation)
                time_diff_days = (current_time - times[j]).astype('timedelta64[D]').astype(float)
                
                # Determine time window based on larger (earlier) event
                if use_adaptive_time:
                    time_window = gk_time_window(earlier_mag)
                else:
                    time_window = time_window_days
                    
                # Skip if outside time window
                if time_diff_days > time_window:
                    continue
                    
                # Calculate distance if coordinates available
                if has_coords:
                    # Vectorized haversine distance calculation
                    dlat = lats[i] - lats[j]
                    dlon = lons[i] - lons[j]
                    a = np.sin(dlat/2)**2 + np.cos(lats[i]) * np.cos(lats[j]) * np.sin(dlon/2)**2
                    distance = 2 * 6371.0 * np.arcsin(np.sqrt(a))
                else:
                    distance = 0
                
                # Determine distance window based on larger (earlier) event
                if use_adaptive_distance:
                    distance_window = gk_distance_window(earlier_mag)
                else:
                    distance_window = distance_km if distance_km else 1000  # Default large value
                
                # Mark as dependent if within space-time window of larger earlier event
                if distance <= distance_window:
                    is_mainshock[i] = False
                    break  # Found a mainshock, this is dependent
                    
        # Create declustered catalog
        self.eq_data_declustered = data[is_mainshock].copy().reset_index(drop=True)
        
        n_removed = len(self.eq_data) - len(self.eq_data_declustered)
        removal_pct = (n_removed / len(self.eq_data)) * 100
        
        print(f"Declustering removed {n_removed} events ({removal_pct:.1f}%)")
        print(f"Declustered catalog: {len(self.eq_data_declustered)} events")
        
        return self.eq_data_declustered
                    
        # Create declustered catalog
        self.eq_data_declustered = data[is_mainshock].copy().reset_index(drop=True)
        
        n_removed = len(self.eq_data) - len(self.eq_data_declustered)
        removal_pct = (n_removed / len(self.eq_data)) * 100
        
        print(f"Declustering removed {n_removed} events ({removal_pct:.1f}%)")
        print(f"Declustered catalog: {len(self.eq_data_declustered)} events")
        
        return self.eq_data_declustered
        
    def seasonal_analysis(self, magnitude_threshold=None, use_declustered=True, 
                         season1_months=[1,2,3,4], season2_months=[7,8,9,10]):
        """
        Perform seasonal analysis comparing two 4-month periods
        Default compares JFMA (Jan-Apr) vs JASO (Jul-Oct) like the New Madrid study
        """
        if magnitude_threshold is None:
            magnitude_threshold = getattr(self, 'completeness_magnitude', 1.0)
            
        # Choose dataset
        if use_declustered and hasattr(self, 'eq_data_declustered'):
            data = self.eq_data_declustered
        else:
            data = self.eq_data
        
        # Filter by magnitude
        data_filtered = data[data['magnitude'] >= magnitude_threshold]
        
        # Separate into seasons
        season1_data = data_filtered[data_filtered['month'].isin(season1_months)]
        season2_data = data_filtered[data_filtered['month'].isin(season2_months)]
        
        n_season1 = len(season1_data)
        n_season2 = len(season2_data)
        
        # Calculate ratio
        ratio = n_season1 / n_season2 if n_season2 > 0 else np.inf
        
        # Statistical significance test using Monte Carlo
        n_trials = 10000
        total_events = n_season1 + n_season2
        
        if total_events > 0:
            # Simulate random distribution between seasons
            random_ratios = []
            for _ in range(n_trials):
                # Randomly assign events to seasons (4 months each)
                random_season1 = np.random.binomial(total_events, 4/8)  # 4 months out of 8
                random_season2 = total_events - random_season1
                if random_season2 > 0:
                    random_ratios.append(random_season1 / random_season2)
                    
            random_ratios = np.array(random_ratios)
            
            # Calculate confidence intervals
            p_95 = np.percentile(random_ratios, [2.5, 97.5])
            p_99 = np.percentile(random_ratios, [0.5, 99.5])
            
            # Check statistical significance
            is_significant_95 = ratio < p_95[0] or ratio > p_95[1]
            is_significant_99 = ratio < p_99[0] or ratio > p_99[1]
            
        else:
            p_95 = [np.nan, np.nan]
            p_99 = [np.nan, np.nan]
            is_significant_95 = False
            is_significant_99 = False
            
        results = {
            'magnitude_threshold': magnitude_threshold,
            'n_season1': n_season1,
            'n_season2': n_season2,
            'ratio': ratio,
            'confidence_95': p_95,
            'confidence_99': p_99,
            'significant_95': is_significant_95,
            'significant_99': is_significant_99
        }
        
        return results
        
    def magnitude_dependent_seasonality(self, mag_range=(1.0, 4.0), mag_step=0.1, 
                                       use_declustered=True):
        """
        Analyze seasonality as function of magnitude threshold
        Reproduces Figure 3/4 from the paper
        """
        mag_thresholds = np.arange(mag_range[0], mag_range[1] + mag_step, mag_step)
        results = []
        
        for mag_thresh in mag_thresholds:
            result = self.seasonal_analysis(mag_thresh, use_declustered)
            results.append(result)
            
        self.seasonality_results = pd.DataFrame(results)
        
        # Calculate both types of residuals for comparison
        self.calculate_magnitude_residuals(mag_range, mag_step, use_declustered)
        self.calculate_craig_residuals(mag_range, mag_step, use_declustered)
        
        return self.seasonality_results
        
    def calculate_magnitude_residuals(self, mag_range=(1.0, 4.0), mag_step=0.1, use_declustered=True):
        """
        Calculate residuals for magnitude binning analysis
        Following the methodology from Craig et al. (2017)
        
        Residuals represent the difference between observed and expected
        earthquake rates for each magnitude bin during seasonal periods
        """
        mag_bins = np.arange(mag_range[0], mag_range[1] + mag_step, mag_step)
        
        # Get the appropriate dataset
        if use_declustered and hasattr(self, 'eq_data_declustered'):
            data = self.eq_data_declustered
        else:
            data = self.eq_data
        
        # Define seasonal months
        season1_months = [1, 2, 3, 4]  # JFMA
        season2_months = [7, 8, 9, 10]  # JASO
        
        residuals_data = []
        
        for i in range(len(mag_bins) - 1):
            mag_min = mag_bins[i]
            mag_max = mag_bins[i + 1]
            
            # Filter data for magnitude bin
            bin_data = data[(data['magnitude'] >= mag_min) & (data['magnitude'] < mag_max)]
            
            if len(bin_data) == 0:
                continue
                
            # Calculate total observation time for each season across all years
            years = sorted(bin_data['year'].unique())
            
            # Count events in each season
            season1_events = bin_data[bin_data['month'].isin(season1_months)]
            season2_events = bin_data[bin_data['month'].isin(season2_months)]
            
            n_season1 = len(season1_events)
            n_season2 = len(season2_events)
            total_events = n_season1 + n_season2
            
            if total_events == 0:
                continue
            
            # Calculate expected rates assuming uniform distribution
            # Each season has 4 months, so expected proportion is 0.5
            expected_season1 = total_events * 0.5
            expected_season2 = total_events * 0.5
            
            # Calculate residuals (observed - expected)
            residual_season1 = n_season1 - expected_season1
            residual_season2 = n_season2 - expected_season2
            
            # Calculate normalized residuals (residual / sqrt(expected))
            # This accounts for Poisson statistics
            norm_residual_season1 = residual_season1 / np.sqrt(expected_season1) if expected_season1 > 0 else 0
            norm_residual_season2 = residual_season2 / np.sqrt(expected_season2) if expected_season2 > 0 else 0
            
            # Calculate chi-square statistic for this bin
            chi_square = (residual_season1**2 / expected_season1 + 
                         residual_season2**2 / expected_season2) if expected_season1 > 0 and expected_season2 > 0 else 0
            
            residuals_data.append({
                'magnitude_min': mag_min,
                'magnitude_max': mag_max,
                'magnitude_center': (mag_min + mag_max) / 2,
                'total_events': total_events,
                'observed_season1': n_season1,
                'observed_season2': n_season2,
                'expected_season1': expected_season1,
                'expected_season2': expected_season2,
                'residual_season1': residual_season1,
                'residual_season2': residual_season2,
                'normalized_residual_season1': norm_residual_season1,
                'normalized_residual_season2': norm_residual_season2,
                'chi_square': chi_square,
                'ratio_observed': n_season1 / n_season2 if n_season2 > 0 else np.inf,
                'ratio_expected': 1.0
            })
        
        self.magnitude_residuals = pd.DataFrame(residuals_data)
        
        print(f"Calculated residuals for {len(self.magnitude_residuals)} magnitude bins")
        return self.magnitude_residuals
        
    def calculate_craig_residuals(self, mag_range=(1.0, 4.0), mag_step=0.1, use_declustered=True):
        """
        Calculate monthly residuals following Craig et al. (2017) methodology
        
        For each magnitude bin and each month, calculate:
        residual = (observed - expected) / sqrt(expected)
        where expected is based on uniform distribution across months
        """
        mag_bins = np.arange(mag_range[0], mag_range[1] + mag_step, mag_step)
        
        # Get the appropriate dataset
        if use_declustered and hasattr(self, 'eq_data_declustered'):
            data = self.eq_data_declustered
        else:
            data = self.eq_data
        
        craig_residuals = []
        
        for i in range(len(mag_bins) - 1):
            mag_min = mag_bins[i]
            mag_max = mag_bins[i + 1]
            mag_center = (mag_min + mag_max) / 2
            
            # Filter data for magnitude bin
            bin_data = data[(data['magnitude'] >= mag_min) & (data['magnitude'] < mag_max)]
            
            if len(bin_data) == 0:
                continue
            
            total_events = len(bin_data)
            
            # Calculate monthly residuals
            for month in range(1, 13):
                # Count observed events in this month across all years
                observed = len(bin_data[bin_data['month'] == month])
                
                # Expected count assuming uniform distribution
                expected = total_events / 12.0
                
                # Calculate normalized residual (Craig et al. 2017 methodology)
                if expected > 0:
                    residual = (observed - expected) / np.sqrt(expected)
                else:
                    residual = 0
                
                craig_residuals.append({
                    'magnitude_min': mag_min,
                    'magnitude_max': mag_max,
                    'magnitude_center': mag_center,
                    'month': month,
                    'month_name': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1],
                    'observed': observed,
                    'expected': expected,
                    'residual': residual,
                    'total_events_bin': total_events
                })
        
        self.craig_residuals = pd.DataFrame(craig_residuals)
        
        print(f"Calculated Craig et al. (2017) style monthly residuals for {len(mag_bins)-1} magnitude bins")
        return self.craig_residuals
        
    def jackknife_analysis(self, magnitude_threshold=None, use_declustered=True):
        """
        Jackknife analysis: remove one year at a time and recalculate seasonality
        Tests robustness of seasonal signal following Craig et al. (2017) methodology
        
        This demonstrates that exclusion of each calendar year separately 
        does not result in any remaining catalogues losing statistically 
        significant seasonal variation between JFMA and JASO periods
        """
        if magnitude_threshold is None:
            magnitude_threshold = getattr(self, 'completeness_magnitude', 1.0)
            
        data = self.eq_data_declustered if use_declustered else self.eq_data
        years = sorted(data['year'].unique())
        
        print(f"Running jackknife analysis: removing each of {len(years)} years separately")
        print(f"Magnitude threshold: M≥{magnitude_threshold:.1f}")
        
        jackknife_results = []
        original_result = self.seasonal_analysis(magnitude_threshold, use_declustered)
        
        for year_to_remove in years:
            # Remove one year
            subset_data = data[data['year'] != year_to_remove]
            
            # Temporarily replace the data for analysis
            original_data = self.eq_data_declustered if use_declustered else self.eq_data
            if use_declustered:
                self.eq_data_declustered = subset_data
            else:
                self.eq_data = subset_data
                
            # Run seasonal analysis
            result = self.seasonal_analysis(magnitude_threshold, use_declustered)
            result['year_removed'] = year_to_remove
            result['n_years_remaining'] = len(years) - 1
            
            # Calculate residual from original analysis
            result['ratio_residual'] = result['ratio'] - original_result['ratio']
            
            jackknife_results.append(result)
            
            # Restore original data
            if use_declustered:
                self.eq_data_declustered = original_data
            else:
                self.eq_data = original_data
                
        self.jackknife_results = pd.DataFrame(jackknife_results)
        
        # Analyze significance retention
        n_significant_95 = sum(self.jackknife_results['significant_95'])
        n_significant_99 = sum(self.jackknife_results['significant_99'])
        n_total = len(self.jackknife_results)
        
        print(f"\nJackknife Analysis Results:")
        print(f"Original analysis: ratio={original_result['ratio']:.3f}, significant at 95%: {original_result['significant_95']}")
        print(f"After removing each year separately:")
        print(f"  Significant at 95% level: {n_significant_95}/{n_total} ({100*n_significant_95/n_total:.1f}%)")
        print(f"  Significant at 99% level: {n_significant_99}/{n_total} ({100*n_significant_99/n_total:.1f}%)")
        
        if n_significant_95 == n_total:
            print("✓ Seasonal signal is robust - significant in ALL jackknife samples")
        else:
            # Convert years list to numpy array for boolean indexing
            years_array = np.array(years)
            non_significant_years = years_array[~self.jackknife_results['significant_95'].values]
            print(f"⚠ Seasonal signal loses significance when removing: {non_significant_years}")
            
        return self.jackknife_results
        
    def plot_jackknife_residuals(self, figsize=(14, 8)):
        """
        Plot jackknife residuals analysis
        Following Craig et al. (2017) Figure showing robustness of seasonal signal
        """
        if not hasattr(self, 'jackknife_results'):
            print("No jackknife results available. Run jackknife_analysis() first.")
            return
            
        results = self.jackknife_results
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Seasonal ratio for each jackknife sample (top left)
        x_pos = range(len(results))
        colors = ['red' if not sig else 'blue' for sig in results['significant_95']]
        
        axes[0,0].bar(x_pos, results['ratio'], color=colors, alpha=0.7)
        axes[0,0].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No seasonal bias')
        axes[0,0].set_xlabel('Year Removed')
        axes[0,0].set_ylabel('JFMA/JASO Ratio')
        axes[0,0].set_title('Seasonal Ratio: Jackknife Analysis')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(results['year_removed'], rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Add legend
        import matplotlib.patches as mpatches
        sig_patch = mpatches.Patch(color='blue', alpha=0.7, label='Significant (p<0.05)')
        nonsig_patch = mpatches.Patch(color='red', alpha=0.7, label='Not significant')
        axes[0,0].legend(handles=[sig_patch, nonsig_patch])
        
        # Plot 2: Ratio residuals (top right)
        axes[0,1].bar(x_pos, results['ratio_residual'], color=colors, alpha=0.7)
        axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0,1].set_xlabel('Year Removed')
        axes[0,1].set_ylabel('Ratio Residual (from full dataset)')
        axes[0,1].set_title('Jackknife Residuals from Original Analysis')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(results['year_removed'], rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Event counts in each jackknife sample (bottom left)
        axes[1,0].bar(x_pos, results['n_season1'], width=0.4, label='JFMA', alpha=0.7, color='blue')
        axes[1,0].bar([x+0.4 for x in x_pos], results['n_season2'], width=0.4, label='JASO', alpha=0.7, color='red')
        axes[1,0].set_xlabel('Year Removed')
        axes[1,0].set_ylabel('Number of Events')
        axes[1,0].set_title('Event Counts per Season (Jackknife)')
        axes[1,0].set_xticks([x+0.2 for x in x_pos])
        axes[1,0].set_xticklabels(results['year_removed'], rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Statistical significance summary (bottom right)
        years = results['year_removed'].tolist()
        sig_95 = results['significant_95'].tolist()
        sig_99 = results['significant_99'].tolist()
        
        # Create significance level plot
        y_95 = [1 if sig else 0 for sig in sig_95]
        y_99 = [2 if sig else 0 for sig in sig_99]
        
        axes[1,1].bar(x_pos, y_95, width=0.4, label='p < 0.05', alpha=0.7, color='orange')
        axes[1,1].bar([x+0.4 for x in x_pos], y_99, width=0.4, label='p < 0.01', alpha=0.7, color='darkred')
        axes[1,1].set_xlabel('Year Removed')
        axes[1,1].set_ylabel('Statistical Significance')
        axes[1,1].set_title('Significance Retention in Jackknife')
        axes[1,1].set_xticks([x+0.2 for x in x_pos])
        axes[1,1].set_xticklabels(results['year_removed'], rotation=45)
        axes[1,1].set_yticks([0, 1, 2])
        axes[1,1].set_yticklabels(['Not Sig.', 'p < 0.05', 'p < 0.01'])
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = f"{getattr(self, 'output_prefix', 'seasonal_nm')}_jackknife_residuals.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved jackknife residuals plot: {output_path}")
        
        return fig
    
    def plot_enhanced_residuals(self, figsize=(12, 8), show_individual_years=True):
        """
        Create an enhanced residual plot following Craig et al. (2017) methodology
        Shows monthly residuals with magnitude binning and year-by-year traces
        """
        if not hasattr(self, 'craig_residuals'):
            print("No Craig residuals calculated. Run magnitude_dependent_seasonality() first.")
            return
            
        residuals = self.craig_residuals
        
        # Create the plot
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Get magnitude bins with sufficient data
        mag_summary = residuals.groupby('magnitude_center').agg({
            'total_events_bin': 'first'
        }).reset_index()
        mag_bins_sufficient = mag_summary[mag_summary['total_events_bin'] >= 20]['magnitude_center'].tolist()
        
        if len(mag_bins_sufficient) == 0:
            print("No magnitude bins with sufficient events for enhanced plotting")
            return
        
        # Select representative magnitude bins for individual traces
        selected_mags = sorted(mag_bins_sufficient)[:3]  # Take up to 3 for clarity
        colors = ['blue', 'red', 'green']
        
        # Plot 1: Monthly residuals for selected magnitude bins (top)
        ax = axes[0]
        
        for i, mag_center in enumerate(selected_mags):
            mag_data = residuals[residuals['magnitude_center'] == mag_center].sort_values('month')
            
            ax.plot(mag_data['month'], mag_data['residual'], 'o-', 
                   color=colors[i], linewidth=2, markersize=8, 
                   label=f'M {mag_center:.1f}±0.05 ({mag_data["total_events_bin"].iloc[0]} events)')
        
        # Add significance bands
        ax.axhline(y=2, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=-2, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Highlight seasonal periods
        ax.axvspan(1, 4, alpha=0.1, color='blue', label='Winter-Spring')
        ax.axvspan(7, 10, alpha=0.1, color='red', label='Summer-Fall')
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Normalized Residual (σ)')
        ax.set_title('Monthly Residuals by Magnitude (Craig et al. 2017 method)')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-3, 3)
        
        # Plot 2: All magnitude bins as heatmap (bottom)
        ax = axes[1]
        
        # Create pivot table for heatmap
        pivot_data = residuals.pivot(index='magnitude_center', columns='month', values='residual')
        
        # Filter to bins with sufficient data
        pivot_data = pivot_data.loc[pivot_data.index.isin(mag_bins_sufficient)]
        
        if not pivot_data.empty:
            im = ax.imshow(pivot_data.values, aspect='auto', cmap='RdBu_r', 
                          vmin=-3, vmax=3, interpolation='nearest', origin='lower')
            
            ax.set_xlabel('Month')
            ax.set_ylabel('Magnitude')
            ax.set_title('Residual Heatmap: All Magnitude Bins')
            ax.set_xticks(range(12))
            ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            
            # Set y-tick labels
            y_ticks = range(len(pivot_data.index))
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'{mag:.1f}' for mag in pivot_data.index])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Normalized Residual (σ)')
            
            # Add grid lines
            ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(pivot_data.index), 1), minor=True)
            ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = f"{getattr(self, 'output_prefix', 'seasonal_nm')}_enhanced_residuals.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved enhanced Craig residuals plot: {output_path}")
        
        return fig

    def plot_cumulative_residuals(self, figsize=(10, 6)):
        """
        Alternative residual plot showing cumulative magnitude-time pattern
        Similar to panel d in the reference figure
        """
        if not hasattr(self, 'eq_data_declustered'):
            data = self.eq_data
        else:
            data = self.eq_data_declustered
        
        # Filter by completeness magnitude
        mag_threshold = getattr(self, 'completeness_magnitude', 1.5)
        data_filtered = data[data['magnitude'] >= mag_threshold].copy()
        
        if len(data_filtered) == 0:
            print("No data above completeness magnitude")
            return
        
        # Sort by time
        data_filtered = data_filtered.sort_values('datetime').reset_index(drop=True)
        
        # Create month-magnitude matrix
        months = np.arange(1, 13)
        mag_bins = np.arange(1.0, 4.1, 0.1)
        mag_centers = (mag_bins[:-1] + mag_bins[1:]) / 2
        
        # Create cumulative count matrix
        cumulative_matrix = np.zeros((len(months), len(mag_centers)))
        
        for i, month in enumerate(months):
            month_data = data_filtered[data_filtered['month'] == month]
            
            for j, mag_center in enumerate(mag_centers):
                mag_min = mag_bins[j]
                mag_max = mag_bins[j + 1]
                count = len(month_data[(month_data['magnitude'] >= mag_min) & 
                                     (month_data['magnitude'] < mag_max)])
                
                # Cumulative sum over magnitude
                if j == 0:
                    cumulative_matrix[i, j] = count
                else:
                    cumulative_matrix[i, j] = cumulative_matrix[i, j-1] + count
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create color map similar to reference
        im = ax.imshow(cumulative_matrix.T, aspect='auto', origin='lower',
                       cmap='RdYlBu_r', interpolation='bilinear')
        
        # Set ticks and labels
        month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(month_labels)
        ax.set_xlabel('Months of the year')
        
        # Magnitude ticks - show every 5th bin
        mag_tick_indices = range(0, len(mag_centers), 5)
        ax.set_yticks(mag_tick_indices)
        ax.set_yticklabels([f'{mag_centers[i]:.1f}' for i in mag_tick_indices])
        ax.set_ylabel('Magnitude')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cumulative number of earthquakes')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = f"{getattr(self, 'output_prefix', 'seasonal_nm')}_cumulative_residuals.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved cumulative residual plot: {output_path}")
        
        return fig
        
    def plot_magnitude_residuals(self, figsize=(14, 10), use_craig_method=True):
        """
        Plot magnitude residuals analysis
        Can use either Craig et al. (2017) methodology or seasonal comparison
        """
        if use_craig_method:
            return self.plot_craig_residuals(figsize)
        else:
            return self.plot_seasonal_residuals(figsize)
            
    def plot_craig_residuals(self, figsize=(14, 10)):
        """
        Plot monthly residuals following Craig et al. (2017) methodology
        Shows residuals as function of month and magnitude
        """
        if not hasattr(self, 'craig_residuals'):
            print("No Craig residuals calculated. Run magnitude_dependent_seasonality() first.")
            return
            
        residuals = self.craig_residuals
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Get unique magnitude bins with sufficient data
        mag_bins = residuals.groupby('magnitude_center').first()
        mag_bins = mag_bins[mag_bins['total_events_bin'] >= 20]  # Only bins with >= 20 events
        selected_mags = sorted(mag_bins.index.tolist())[:4]  # Take up to 4 bins
        
        if len(selected_mags) == 0:
            print("No magnitude bins with sufficient events for plotting")
            return
        
        colors = ['blue', 'red', 'green', 'orange']
        
        # Plot 1: Monthly residuals for different magnitude bins (top left)
        ax = axes[0,0]
        for i, mag_center in enumerate(selected_mags):
            mag_data = residuals[residuals['magnitude_center'] == mag_center]
            mag_data = mag_data.sort_values('month')
            
            ax.plot(mag_data['month'], mag_data['residual'], 'o-', 
                   color=colors[i % len(colors)], 
                   label=f'M {mag_center:.1f}±0.05', linewidth=2, markersize=6)
        
        # Add significance bands
        ax.axhline(y=2, color='gray', linestyle='--', alpha=0.7, label='±2σ')
        ax.axhline(y=-2, color='gray', linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax.set_xlabel('Month')
        ax.set_ylabel('Normalized Residual (σ)')
        ax.set_title('Monthly Residuals by Magnitude (Craig et al. 2017 method)')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Residual heatmap (top right)
        ax = axes[0,1]
        
        # Create pivot table for heatmap
        pivot_data = residuals.pivot(index='magnitude_center', columns='month', values='residual')
        
        # Only include magnitude bins with sufficient data
        pivot_data = pivot_data.loc[pivot_data.index.isin(selected_mags)]
        
        if not pivot_data.empty:
            im = ax.imshow(pivot_data.values, aspect='auto', cmap='RdBu_r', 
                          vmin=-3, vmax=3, interpolation='nearest')
            
            ax.set_xlabel('Month')
            ax.set_ylabel('Magnitude')
            ax.set_title('Residual Heatmap')
            ax.set_xticks(range(12))
            ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            ax.set_yticks(range(len(pivot_data.index)))
            ax.set_yticklabels([f'{mag:.1f}' for mag in pivot_data.index])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Normalized Residual (σ)')
        
        # Plot 3: Seasonal averages (bottom left)
        ax = axes[1,0]
        
        # Calculate seasonal averages
        winter_months = [12, 1, 2]  # DJF
        spring_months = [3, 4, 5]   # MAM
        summer_months = [6, 7, 8]   # JJA
        autumn_months = [9, 10, 11] # SON
        
        for i, mag_center in enumerate(selected_mags):
            mag_data = residuals[residuals['magnitude_center'] == mag_center]
            
            winter_mean = mag_data[mag_data['month'].isin(winter_months)]['residual'].mean()
            spring_mean = mag_data[mag_data['month'].isin(spring_months)]['residual'].mean()
            summer_mean = mag_data[mag_data['month'].isin(summer_months)]['residual'].mean()
            autumn_mean = mag_data[mag_data['month'].isin(autumn_months)]['residual'].mean()
            
            seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
            means = [winter_mean, spring_mean, summer_mean, autumn_mean]
            
            x_pos = np.arange(len(seasons)) + i * 0.2
            ax.bar(x_pos, means, width=0.15, label=f'M {mag_center:.1f}', 
                  color=colors[i % len(colors)], alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Season')
        ax.set_ylabel('Mean Residual (σ)')
        ax.set_title('Seasonal Averages')
        ax.set_xticks(np.arange(len(seasons)) + 0.3)
        ax.set_xticklabels(seasons)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Event counts and statistics (bottom right)
        ax = axes[1,1]
        
        # Show statistics for selected magnitude bins
        stats_data = []
        for mag_center in selected_mags:
            mag_data = residuals[residuals['magnitude_center'] == mag_center]
            total_events = mag_data['total_events_bin'].iloc[0]
            max_residual = mag_data['residual'].abs().max()
            
            # Calculate chi-square for uniform distribution test
            observed = mag_data['observed'].values
            expected = mag_data['expected'].values
            chi_square = np.sum((observed - expected)**2 / expected)
            
            stats_data.append({
                'magnitude': mag_center,
                'events': total_events,
                'max_residual': max_residual,
                'chi_square': chi_square
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Bar plot of statistics
        x_pos = range(len(stats_df))
        bars = ax.bar(x_pos, stats_df['max_residual'], alpha=0.7, color='purple')
        
        # Add chi-square values as text
        for i, (bar, chi_val) in enumerate(zip(bars, stats_df['chi_square'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'χ²={chi_val:.1f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Magnitude Bin')
        ax.set_ylabel('Maximum |Residual| (σ)')
        ax.set_title('Statistical Summary')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'M {mag:.1f}' for mag in stats_df['magnitude']])
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = f"{getattr(self, 'output_prefix', 'seasonal_nm')}_craig_residuals.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved Craig et al. (2017) style residuals plot: {output_path}")
        
        return fig
        
    def plot_seasonal_residuals(self, figsize=(14, 10)):
        """
        Plot seasonal residuals analysis (original implementation)
        Similar to residual plots in Craig et al. (2017)
        """
        if not hasattr(self, 'magnitude_residuals'):
            print("No residuals calculated. Run magnitude_dependent_seasonality() first.")
            return
            
        residuals = self.magnitude_residuals
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Observed vs Expected events by magnitude (top left)
        axes[0,0].bar(residuals['magnitude_center'] - 0.02, residuals['observed_season1'], 
                     width=0.04, label='JFMA (observed)', alpha=0.7, color='blue')
        axes[0,0].bar(residuals['magnitude_center'] + 0.02, residuals['observed_season2'], 
                     width=0.04, label='JASO (observed)', alpha=0.7, color='red')
        axes[0,0].plot(residuals['magnitude_center'], residuals['expected_season1'], 
                      'b--', label='JFMA (expected)', linewidth=2)
        axes[0,0].plot(residuals['magnitude_center'], residuals['expected_season2'], 
                      'r--', label='JASO (expected)', linewidth=2)
        axes[0,0].set_xlabel('Magnitude')
        axes[0,0].set_ylabel('Number of Events')
        axes[0,0].set_title('Observed vs Expected Events by Magnitude')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Raw residuals (top right)
        width = 0.04
        axes[0,1].bar(residuals['magnitude_center'] - width/2, residuals['residual_season1'], 
                     width, label='JFMA residual', alpha=0.7, color='blue')
        axes[0,1].bar(residuals['magnitude_center'] + width/2, residuals['residual_season2'], 
                     width, label='JASO residual', alpha=0.7, color='red')
        axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0,1].set_xlabel('Magnitude')
        axes[0,1].set_ylabel('Residual (Observed - Expected)')
        axes[0,1].set_title('Raw Residuals by Magnitude')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Normalized residuals (bottom left)
        axes[1,0].bar(residuals['magnitude_center'] - width/2, residuals['normalized_residual_season1'], 
                     width, label='JFMA normalized', alpha=0.7, color='blue')
        axes[1,0].bar(residuals['magnitude_center'] + width/2, residuals['normalized_residual_season2'], 
                     width, label='JASO normalized', alpha=0.7, color='red')
        axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1,0].axhline(y=2, color='gray', linestyle='--', alpha=0.7, label='±2σ')
        axes[1,0].axhline(y=-2, color='gray', linestyle='--', alpha=0.7)
        axes[1,0].set_xlabel('Magnitude')
        axes[1,0].set_ylabel('Normalized Residual (σ)')
        axes[1,0].set_title('Normalized Residuals by Magnitude')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Chi-square values (bottom right)
        axes[1,1].bar(residuals['magnitude_center'], residuals['chi_square'], 
                     width=0.08, alpha=0.7, color='green')
        axes[1,1].axhline(y=3.84, color='red', linestyle='--', alpha=0.7, 
                         label='p=0.05 threshold (χ²=3.84)')
        axes[1,1].set_xlabel('Magnitude')
        axes[1,1].set_ylabel('Chi-square Value')
        axes[1,1].set_title('Chi-square Test by Magnitude')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = f"{getattr(self, 'output_prefix', 'seasonal_nm')}_magnitude_residuals.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved magnitude residuals plot: {output_path}")
        
        return fig
        
    def analyze_residual_significance(self):
        """
        Analyze statistical significance of residuals
        """
        if not hasattr(self, 'magnitude_residuals'):
            print("No residuals calculated. Run magnitude_dependent_seasonality() first.")
            return
            
        residuals = self.magnitude_residuals
        
        print("Magnitude Residuals Analysis")
        print("=" * 50)
        print("Mag Range | Events | JFMA Obs/Exp | JASO Obs/Exp | Norm Resid JFMA | Norm Resid JASO | χ² | Significant")
        print("-" * 110)
        
        significant_bins = 0
        total_bins = len(residuals)
        
        for _, row in residuals.iterrows():
            jfma_ratio = row['observed_season1'] / row['expected_season1'] if row['expected_season1'] > 0 else 0
            jaso_ratio = row['observed_season2'] / row['expected_season2'] if row['expected_season2'] > 0 else 0
            
            # Chi-square critical value for p=0.05 with 1 df is 3.84
            is_significant = row['chi_square'] > 3.84
            if is_significant:
                significant_bins += 1
                
            sig_marker = "**" if is_significant else "  "
            
            print(f"{row['magnitude_min']:4.1f}-{row['magnitude_max']:4.1f} | "
                  f"{row['total_events']:6.0f} | "
                  f"{jfma_ratio:11.2f} | {jaso_ratio:11.2f} | "
                  f"{row['normalized_residual_season1']:15.2f} | {row['normalized_residual_season2']:15.2f} | "
                  f"{row['chi_square']:6.2f} | {sig_marker}")
        
        print("-" * 110)
        print(f"Significant bins (p<0.05): {significant_bins}/{total_bins} ({100*significant_bins/total_bins:.1f}%)")
        
        # Overall statistics
        total_chi_square = residuals['chi_square'].sum()
        degrees_freedom = len(residuals)
        
        print(f"\nOverall Statistics:")
        print(f"Total χ² = {total_chi_square:.2f}")
        print(f"Degrees of freedom = {degrees_freedom}")
        print(f"Reduced χ² = {total_chi_square/degrees_freedom:.2f}")
        
        return residuals
        
    def plot_seasonal_histogram(self, magnitude_threshold=None, use_declustered=True, 
                               figsize=(12, 8)):
        """Plot seasonal histogram similar to Figure 3/4 in paper"""
        if magnitude_threshold is None:
            magnitude_threshold = getattr(self, 'completeness_magnitude', 1.0)
            
        data = self.eq_data_declustered if use_declustered else self.eq_data
        data_filtered = data[data['magnitude'] >= magnitude_threshold]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Monthly histogram (top left)
        monthly_counts = data_filtered.groupby('month').size()
        months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        
        axes[0,0].bar(range(1, 13), monthly_counts.reindex(range(1, 13), fill_value=0), color='black')
        axes[0,0].set_xlabel('Month')
        axes[0,0].set_ylabel('Number of Earthquakes')
        axes[0,0].set_title(f'Monthly Distribution (M≥{magnitude_threshold:.1f})')
        axes[0,0].set_xticks(range(1, 13))
        axes[0,0].set_xticklabels(months)
        
        # Magnitude-dependent seasonality (top right)
        if hasattr(self, 'seasonality_results'):
            results = self.seasonality_results
            axes[0,1].plot(results['magnitude_threshold'], results['ratio'], 'ko-')
            axes[0,1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            axes[0,1].set_xlabel('Magnitude Threshold')
            axes[0,1].set_ylabel('Event Count Ratio (JFMA/JASO)')
            axes[0,1].set_title('Seasonal Event Count Ratio vs Magnitude\n(JFMA = Jan-Apr, JASO = Jul-Oct)')
            
            # Add confidence intervals if available
            if 'confidence_95' in results.columns:
                ci_95 = np.array([x for x in results['confidence_95']])
                if len(ci_95) > 0 and len(ci_95[0]) == 2:
                    axes[0,1].fill_between(results['magnitude_threshold'], 
                                         [x[0] for x in ci_95], 
                                         [x[1] for x in ci_95],
                                         alpha=0.3, color='gray', label='95% CI')
        
        # Time series of cumulative events (bottom left)
        data_filtered_sorted = data_filtered.sort_values('datetime')
        data_filtered_sorted['cumulative'] = range(1, len(data_filtered_sorted) + 1)
        
        axes[1,0].plot(data_filtered_sorted['datetime'], data_filtered_sorted['cumulative'], color='black')
        axes[1,0].set_xlabel('Year')
        axes[1,0].set_ylabel('Cumulative Earthquakes')
        axes[1,0].set_title('Cumulative Seismicity')
        
        # Seasonal stacking (bottom right)
        # Stack all years to show average seasonal pattern
        data_filtered['month_day'] = data_filtered['datetime'].dt.strftime('%m-%d')
        
        # Create bins for 2-month periods
        bimonthly_bins = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12)]
        bimonthly_counts = []
        bin_labels = ['JF', 'MA', 'MJ', 'JA', 'SO', 'ND']
        
        for start_month, end_month in bimonthly_bins:
            count = len(data_filtered[data_filtered['month'].isin([start_month, end_month])])
            bimonthly_counts.append(count)
            
        axes[1,1].bar(range(len(bimonthly_bins)), bimonthly_counts, color='black')
        axes[1,1].set_xlabel('Bi-monthly Period')
        axes[1,1].set_ylabel('Number of Earthquakes')
        axes[1,1].set_title('Bi-monthly Distribution')
        axes[1,1].set_xticks(range(len(bimonthly_bins)))
        axes[1,1].set_xticklabels(bin_labels)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = f"{getattr(self, 'output_prefix', 'seasonal_nm')}_seasonal_histogram.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved seasonal histogram plot to: {output_path}")
        
        # plt.show()  # Commented out to avoid interactive display
        
    def plot_grace_correlation(self, figsize=(15, 10)):
        """Plot correlation with GRACE data if available"""
        if self.grace_data is None:
            print("No GRACE data loaded")
            return
            
        # Resample earthquake rate to match GRACE temporal resolution
        # Create monthly earthquake counts
        data = getattr(self, 'eq_data_declustered', self.eq_data)
        data = data[data['magnitude'] >= getattr(self, 'completeness_magnitude', 1.0)]
        
        # Monthly earthquake rate
        data['year_month'] = data['datetime'].dt.to_period('M')
        monthly_counts = data.groupby('year_month').size()
        
        # Match time periods
        grace_monthly = self.grace_data.copy()
        grace_monthly['year_month'] = grace_monthly['datetime'].dt.to_period('M')
        grace_monthly = grace_monthly.groupby('year_month')['equivalent_water_height'].mean()
        
        # Find common time period
        common_periods = monthly_counts.index.intersection(grace_monthly.index)
        
        if len(common_periods) < 12:  # Need at least one year
            print("Insufficient overlap between earthquake and GRACE data")
            return
            
        eq_rates = monthly_counts.reindex(common_periods, fill_value=0)
        grace_values = grace_monthly.reindex(common_periods)
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Time series
        time_axis = [pd.Timestamp(str(p)) for p in common_periods]
        
        axes[0].plot(time_axis, eq_rates.values, 'ko-', label='Earthquake Rate')
        axes[0].set_ylabel('Earthquakes/Month')
        axes[0].set_title('Earthquake Rate vs GRACE Loading')
        axes[0].legend()
        
        axes[1].plot(time_axis, grace_values.values, 'bo-', label='GRACE EWH')
        axes[1].set_ylabel('Equivalent Water Height (mm)')
        axes[1].legend()
        
        # Annual stacks
        months = np.array([p.month for p in common_periods])
        annual_eq = np.zeros(12)
        annual_grace = np.zeros(12)
        
        for month in range(1, 13):
            mask = months == month
            if np.sum(mask) > 0:
                annual_eq[month-1] = np.mean(eq_rates.values[mask])
                annual_grace[month-1] = np.mean(grace_values.values[mask])
        
        month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        
        ax2 = axes[2]
        ax3 = ax2.twinx()
        
        line1 = ax2.plot(range(12), annual_eq, 'ko-', label='Earthquake Rate')
        ax2.set_ylabel('Avg Earthquakes/Month', color='black')
        ax2.set_xlabel('Month')
        
        line2 = ax3.plot(range(12), annual_grace, 'bo-', label='GRACE EWH')
        ax3.set_ylabel('Avg EWH (mm)', color='blue')
        
        ax2.set_xticks(range(12))
        ax2.set_xticklabels(month_names)
        ax2.set_title('Annual Stacks')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right')
        
        plt.tight_layout()
        # plt.show()  # Commented out to avoid interactive display
        
        # Calculate correlation
        correlation = np.corrcoef(eq_rates.values, grace_values.values)[0,1]
        print(f"Correlation between earthquake rate and GRACE loading: {correlation:.3f}")
        
    def generate_report(self):
        """Generate a summary report of the analysis"""
        print("="*60)
        print("SEASONAL SEISMICITY ANALYSIS REPORT")
        print("="*60)
        
        # Basic statistics
        print(f"Earthquake Catalog: {len(self.eq_data)} events")
        if hasattr(self, 'eq_data_declustered'):
            print(f"Declustered Catalog: {len(self.eq_data_declustered)} events")
        
        print(f"Time Period: {self.eq_data['datetime'].min()} to {self.eq_data['datetime'].max()}")
        print(f"Magnitude Range: {self.eq_data['magnitude'].min():.1f} to {self.eq_data['magnitude'].max():.1f}")
        
        if hasattr(self, 'completeness_magnitude'):
            print(f"Completeness Magnitude: {self.completeness_magnitude:.1f}")
        
        if hasattr(self, 'b_value'):
            print(f"Gutenberg-Richter b-value: {self.b_value:.2f}")
        
        # Seasonal analysis results
        if hasattr(self, 'seasonality_results'):
            print("\nSEASONAL ANALYSIS RESULTS:")
            results = self.seasonality_results
            significant = results[results['significant_95']]
            if len(significant) > 0:
                print(f"Significant seasonal variations found for magnitude thresholds:")
                for _, row in significant.iterrows():
                    print(f"  M≥{row['magnitude_threshold']:.1f}: Ratio={row['ratio']:.2f}, "
                          f"JFMA={row['n_season1']}, JASO={row['n_season2']}")
            else:
                print("No statistically significant seasonal variations found")
        
        # Jackknife analysis results  
        if hasattr(self, 'jackknife_results'):
            print("\nJACKKNIFE ROBUSTNESS TEST:")
            jk_results = self.jackknife_results
            n_significant_95 = sum(jk_results['significant_95'])
            n_significant_99 = sum(jk_results['significant_99'])
            n_total = len(jk_results)
            
            print(f"Significance retention after removing each year:")
            print(f"  95% confidence level: {n_significant_95}/{n_total} ({100*n_significant_95/n_total:.1f}%)")
            print(f"  99% confidence level: {n_significant_99}/{n_total} ({100*n_significant_99/n_total:.1f}%)")
            
            if n_significant_95 == n_total:
                print("✓ Seasonal signal is ROBUST - significant in all jackknife samples")
            else:
                problematic_years = jk_results[~jk_results['significant_95']]['year_removed'].tolist()
                print(f"⚠ Signal loses significance when removing: {problematic_years}")
            
            # Show range of ratios
            ratio_min = jk_results['ratio'].min()
            ratio_max = jk_results['ratio'].max()
            ratio_std = jk_results['ratio'].std()
            print(f"Ratio range across jackknife samples: {ratio_min:.3f} - {ratio_max:.3f} (std: {ratio_std:.3f})")
        
        print("\n" + "="*60)


# Command line interface function
def run_seasonal_analysis(catalog_path='nm_usgs_processed_longer.csv',
                         grace_csv=None,
                         output_prefix='seasonal_nm',
                         mag_min=1.0,
                         mag_max=3.5,
                         use_declustered=False,
                         completeness_mag=1.5):
    """
    Run the complete seasonal seismicity analysis
    """
    print(f"Running seasonal analysis on: {catalog_path}")
    
    # Initialize analyzer
    analyzer = SeasonalSeismicityAnalyzer(catalog_path, grace_csv)
    analyzer.output_prefix = output_prefix  # Set output prefix for file naming
    
    # Set completeness magnitude
    analyzer.completeness_magnitude = completeness_mag
    print(f"Using completeness magnitude: {completeness_mag:.1f}")
    
    # Apply declustering
    analyzer.decluster_catalog(use_declustered=use_declustered)
    
    # Run analysis steps
    analyzer.gutenberg_richter_analysis()
    analyzer.magnitude_dependent_seasonality(mag_range=(mag_min, mag_max))
    
    # Run jackknife analysis for robustness testing
    analyzer.jackknife_analysis(magnitude_threshold=completeness_mag, use_declustered=use_declustered)
    
    # Analyze residuals significance (magnitude binning residuals)
    analyzer.analyze_residual_significance()
    
    # Generate plots
    analyzer.plot_seasonal_histogram()
    analyzer.plot_magnitude_residuals()
    analyzer.plot_jackknife_residuals()
    
    # Generate enhanced residual plots
    analyzer.plot_enhanced_residuals()
    analyzer.plot_cumulative_residuals()
    
    # If GRACE data is available, plot correlation
    if grace_csv:
        analyzer.plot_grace_correlation()
    
    # Generate report
    analyzer.generate_report()
    
    return analyzer
# Example usage and test function
def example_analysis():
    """Legacy example function - use command line interface instead"""
    return run_seasonal_analysis()


# Command line interface
if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Run Seasonal Seismicity Analysis for New Madrid')
    p.add_argument('--catalog', type=str, default='nm_usgs_processed_longer.csv', 
                   help='Earthquake catalog CSV file')
    p.add_argument('--grace-csv', type=str, default=None, 
                   help='GRACE data CSV file (optional)')
    p.add_argument('--output-prefix', type=str, default='seasonal_nm', 
                   help='Output file prefix')
    p.add_argument('--mag-min', type=float, default=1.0, 
                   help='Minimum magnitude for seasonality analysis')
    p.add_argument('--mag-max', type=float, default=3.5, 
                   help='Maximum magnitude for seasonality analysis')
    p.add_argument('--use-declustered', action='store_true',
                   help='Use declustered catalog (Gardner-Knopoff algorithm)')
    p.add_argument('--completeness-mag', type=float, default=1.5,
                   help='Magnitude of completeness (default: 1.5)')
    p.add_argument('--time-window', type=float, default=None,
                   help='Fixed time window for declustering (days, default: adaptive)')
    p.add_argument('--distance-window', type=float, default=None,
                   help='Fixed distance window for declustering (km, default: adaptive)')
    
    args = p.parse_args()
    
    # Check if catalog file exists
    if not os.path.exists(args.catalog):
        print(f"Error: Catalog file '{args.catalog}' not found")
        print("Available CSV files in current directory:")
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                print(f"  {f}")
        exit(1)
    
    # Check GRACE file if provided
    if args.grace_csv and not os.path.exists(args.grace_csv):
        print(f"Warning: GRACE file '{args.grace_csv}' not found, proceeding without GRACE data")
        args.grace_csv = None
    
    # Run the analysis
    analyzer = run_seasonal_analysis(
        catalog_path=args.catalog,
        grace_csv=args.grace_csv,
        output_prefix=args.output_prefix,
        mag_min=args.mag_min,
        mag_max=args.mag_max,
        use_declustered=args.use_declustered,
        completeness_mag=args.completeness_mag
    )
    
    # Apply custom declustering windows if specified
    if args.use_declustered and (args.time_window or args.distance_window):
        print("Re-running declustering with custom parameters...")
        analyzer.decluster_catalog(
            use_declustered=True,
            time_window_days=args.time_window,
            distance_km=args.distance_window
        )
        # Re-run analysis with new declustered catalog
        analyzer.magnitude_dependent_seasonality(mag_range=(args.mag_min, args.mag_max))
        analyzer.plot_seasonal_histogram()
        analyzer.generate_report()
    
    print(f"\nAnalysis complete! Check output files with prefix '{args.output_prefix}'")