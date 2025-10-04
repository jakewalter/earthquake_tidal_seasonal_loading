#!/usr/bin/env python3
"""
New Madrid Tidal Seasonal Analysis - Enhanced version of Beaucé et al. (2023) 
with GRACE water loading stress combination and multiband analysis.

Features:
 - Choose projection: optimally_oriented (default) or fault_plane (user strike/dip/rake).
 - Builds 3x3 stress tensor from SPOTL strain (plane strain).
 - Phase & amplitude via Hilbert transform.
 - Sliding-window analysis, alpha fitting with cosine model.
 - Bootstrap significance using phase-randomization null (shuffling event-phase assignments).
 - GRACE water loading stress combination
 - Multiband frequency analysis (tidal vs seasonal)
 - Comparative plots with/without GRACE loading
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy.signal import hilbert
import warnings
import os
import json
import argparse
import subprocess
from scipy import signal

warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy import interpolate

# Add xarray for GRACE data handling
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    print("Warning: xarray not available, GRACE functionality disabled")

def _unique_time_average(x, y):
    # Helper: collapse duplicate timestamps by averaging their values
    # - ensures monotonic x for interpolation routines
    # - returns (unique_times, averaged_values)
    # This mirrors the short, pragmatic comment style used in easyQuake.
    x = np.asarray(x); y = np.asarray(y)
    good = np.isfinite(x) & np.isfinite(y)
    if not good.any():
        return np.array([]), np.array([])
    xg, yg = x[good], y[good]
    order = np.argsort(xg)
    xg, yg = xg[order], yg[order]
    uniq, idx, counts = np.unique(xg, return_index=True, return_counts=True)
    if np.all(counts == 1):
        return xg, yg
    ys = []
    j = 0
    for c in counts:
        ys.append(np.mean(yg[j:j+c]))
        j += c
    return uniq, np.array(ys)

def safe_interp1d(x, y, kind='cubic', bounds_error=False, fill_value=np.nan):
    # Create an interp1d-like callable that is robust to duplicates and sparse data
    # - uses `_unique_time_average` to remove duplicates
    # - falls back to linear if cubic is not supported
    # - on failure, returns a constant function (mean)
    x_u, y_u = _unique_time_average(x, y)
    if x_u.size == 0:
        raise ValueError("safe_interp1d: no valid data")
    if kind == 'cubic' and x_u.size < 4:
        kind_use = 'linear'
    else:
        kind_use = kind
    try:
        return interpolate.interp1d(x_u, y_u, kind=kind_use,
                                    bounds_error=bounds_error,
                                    fill_value=fill_value)
    except Exception:
        # If interp fails (e.g. singular) return constant function of the mean
        meanv = float(np.nanmean(y_u))
        return lambda t: np.full_like(np.atleast_1d(t), meanv, dtype=float)

# ----------------------------
# Helper: convert to epoch sec (consolidated)
# ----------------------------
def to_epoch_seconds(times):
    """Convert timestamps to integer epoch seconds (UTC).

    Robust implementation that accepts:
    - numeric epoch arrays (returned as-is),
    - pandas DatetimeIndex, Series, or single timestamp-like inputs.

    Returns numpy array or scalar of seconds since epoch.
    """
    # If already numeric (epoch seconds), return as-is
    if hasattr(times, 'dtype') and np.issubdtype(getattr(times, 'dtype', None), np.number):
        return times

    dt = pd.to_datetime(times)
    # DatetimeIndex path
    if isinstance(dt, pd.DatetimeIndex):
        if dt.tz is None:
            dt = dt.tz_localize('UTC')
        else:
            dt = dt.tz_convert('UTC')
        return dt.astype('int64') // 10**9

    # Series path
    if isinstance(dt, pd.Series) or (hasattr(dt, 'dt') and getattr(dt, 'dt') is not None):
        try:
            if dt.dt.tz is None:
                dt = dt.dt.tz_localize('UTC')
            else:
                dt = dt.dt.tz_convert('UTC')
            return dt.astype('int64') // 10**9
        except Exception:
            return pd.to_datetime(times).astype('int64') // 10**9

    # Single timestamp fallback
    if getattr(dt, 'tz', None) is None:
        dt = dt.tz_localize('UTC')
    else:
        dt = dt.tz_convert('UTC')
    # .value gives ns as int64
    return int(dt.value // 10**9)

# ----------------------------
# GRACE Water Loading Functions
# ----------------------------
def load_grace_data(grace_ncfile, lon_min, lon_max, lat_min, lat_max,
                    strike=None, dip=None, rake=None, mu_eff=0.4,
                    start_time=None, end_time=None):
    """Load GRACE mascon series and return (stress_fn, df).

    Short notes:
    - Returns a callable stress_fn(t) accepting epoch seconds and a DataFrame
      of time + stress columns for inspection.
    - If strike/dip/rake provided, computes dCFS; otherwise returns load_pa.
    """
    # Support using a pre-existing processing routine to avoid duplication
    if not HAS_XARRAY:
        raise ImportError("xarray required for GRACE data processing")

    try:
        from grace_data import process_grace_to_coulomb
    except Exception:
        process_grace_to_coulomb = None

    if process_grace_to_coulomb is not None:
        grace_df = process_grace_to_coulomb(
            grace_ncfile, lon_min, lon_max, lat_min, lat_max,
            strike, dip, rake, mu_eff=mu_eff
        )
        grace_df['time'] = pd.to_datetime(grace_df['time'])
        stress_col = 'dCFS' if 'dCFS' in grace_df.columns else ('load_pa' if 'load_pa' in grace_df.columns else None)
    else:
        ds = xr.open_dataset(grace_ncfile)

        # detect plausible variable name
        var_name = "water_thickness"
        if var_name not in ds.variables:
            for alt in ("we_thickness", "water_thickness", "lwe_thickness", "equivalent_water_thickness"):
                if alt in ds.variables:
                    var_name = alt
                    break
            else:
                raise ValueError(f"Couldn't find variable; available: {list(ds.data_vars)}")

        # normalize longitudes if negative
        if lon_min < 0:
            lon_min = lon_min + 360
        if lon_max < 0:
            lon_max = lon_max + 360

        lwe = ds[var_name].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        lwe_mean = lwe.mean(dim=['lat', 'lon'])

        # convert cm -> Pa (approx)
        load_pa = lwe_mean * 98.1

        times = pd.to_datetime(lwe_mean['time'].values)
        grace_df = pd.DataFrame({'time': times, 'load_pa': load_pa.values})

        if all(p is not None for p in [strike, dip, rake]):
            dCFS_values = []
            for load in load_pa.values:
                if np.isfinite(load):
                    dCFS, _, _ = coulomb_stress_change(load, strike, dip, rake, mu_eff)
                else:
                    dCFS = np.nan
                dCFS_values.append(dCFS)
            grace_df['dCFS'] = dCFS_values
            stress_col = 'dCFS'
        else:
            stress_col = 'load_pa'

    # trim by requested time window
    if start_time is not None:
        st = pd.to_datetime(start_time)
        grace_df = grace_df[grace_df['time'] >= st]
    if end_time is not None:
        et = pd.to_datetime(end_time)
        grace_df = grace_df[grace_df['time'] <= et]

    grace_df = grace_df.dropna()
    if len(grace_df) == 0:
        raise ValueError("No valid GRACE data found")

    times_epoch = to_epoch_seconds(grace_df['time'])
    stress_values = grace_df[stress_col].values
    interp_fn = safe_interp1d(times_epoch, stress_values, kind='linear', bounds_error=False, fill_value=0.0)

    return lambda t: interp_fn(t), grace_df

def coulomb_stress_change(load_pa, strike, dip, rake, mu_eff=0.4, nu=0.25):
    """Compute dCFS from a vertical surface load (Pa).

    Returns tuple (dCFS, shear, normal). Small helper used by GRACE routines.
    """
    sigma = stress_tensor_from_load(load_pa, nu)
    n, s = fault_vectors(strike, dip, rake)
    traction = sigma @ n
    sigma_n = np.dot(traction, n)
    tau = np.dot(traction, s)
    dCFS = tau + mu_eff * sigma_n
    return dCFS, tau, sigma_n

def stress_tensor_from_load(load_pa, nu=0.25):
    """Create 3x3 stress tensor from vertical load.

    Simple plane-strain approximation: horizontal stress = nu/(1-nu)*load.
    """
    horiz = nu / (1 - nu) * load_pa
    return np.array([[horiz, 0, 0],
                     [0, horiz, 0],
                     [0, 0, load_pa]])

# Multiband analysis removed: functionality consolidated into single-band
# analysis using the TidalSensitivityAnalyzer. The previous frequency_filter
# and analyze_multiband_sensitivity helpers have been removed to simplify the
# script and avoid duplicate analysis pathways.

# NOTE: `to_epoch_seconds` consolidated earlier in this file. Duplicate
# implementations were removed to avoid confusion.

# ------------------------------------
# Fault geometry & Coulomb-on-plane
# ------------------------------------
def fault_vectors(strike, dip, rake, degrees=True):
    """Return normal n and slip s vectors (3-element) from strike,dip,rake."""
    
    # Check for NaN values and handle them
    if np.any(np.isnan([strike, dip, rake])):
        print(f"Warning: NaN values detected in fault parameters: strike={strike}, dip={dip}, rake={rake}")
        # Return zero vectors for NaN inputs to prevent downstream NaN propagation
        n = np.array([0.0, 0.0, 1.0])  # Default normal vector (vertical)
        s = np.array([1.0, 0.0, 0.0])  # Default slip vector (horizontal)
        return n, s
    
    if degrees:
        strike = np.radians(strike)
        dip = np.radians(dip)
        rake = np.radians(rake)

    # Aki & Richards convention
    # Normal vector (pointing outward from hanging wall)
    n = np.array([
        -np.sin(dip) * np.sin(strike),
         np.sin(dip) * np.cos(strike),
        -np.cos(dip)
    ])
    
    # Slip
    s = np.array([
         np.cos(rake) * np.cos(strike) + np.sin(rake) * np.cos(dip) * np.sin(strike),
         np.cos(rake) * np.sin(strike) - np.sin(rake) * np.cos(dip) * np.cos(strike),
        -np.sin(rake) * np.sin(dip)  # Note: negative sign for proper down direction
    ])
    
    # Normalize vectors
    n_norm = np.linalg.norm(n)
    s_norm = np.linalg.norm(s)
    
    # Handle zero-length vectors
    if n_norm == 0:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = n / n_norm
        
    if s_norm == 0:
        s = np.array([1.0, 0.0, 0.0])
    else:
        s = s / s_norm
    
    return n, s
def coulomb_on_plane(stress_tensor, n, s, mu=0.6):
    """
    Compute Coulomb components on an arbitrary plane.
    - `stress_tensor` should be 3x3 (Pa)
    - `n` (normal) and `s` (slip) are 3-element unit vectors
    Returns (dCFS, shear, normal)
    """
    """
    Compute Coulomb stress change on a specific fault plane
    for a stress tensor time series.

    Parameters
    ----------
    stress_tensor : ndarray (3,3,N)
        Stress tensors (Pa), ENZ coords, time along last axis.
    n : (3,) array
        Fault normal vector.
    s : (3,) array
        Slip direction vector.
    mu : float
        Friction coefficient.

    Returns
    -------
    dCFS : (N,) array
        Coulomb stress change (Pa).
    sigma_n : (N,) array
        Normal stress (Pa).
    tau : (N,) array
        Shear stress (Pa).
    """
    # Traction on plane (3,N)
    traction = np.einsum('ijt,j->it', stress_tensor, n)

    # Normal and shear stresses
    sigma_n = np.einsum('it,i->t', traction, n)  # (N,)
    tau = np.einsum('it,i->t', traction, s)      # (N,)

    dCFS = tau - mu * sigma_n
    return dCFS, sigma_n, tau

# -------------------------
# Data load
# -------------------------
def load_catalog_for_paper(catalog_path: str) -> pd.DataFrame:
    df = pd.read_csv(catalog_path)
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
    return df.sort_values('time').reset_index(drop=True)

def load_spotl_strain_csv(spotl_path: str) -> pd.DataFrame:
    df = pd.read_csv(spotl_path)
    df['time'] = pd.to_datetime(df['time'])
    # Example column names expected: strain_N_nanostrain, strain_E_nanostrain, strain_NE_nanostrain
    # Normalize: convert nanostrain -> strain
    for col in list(df.columns):
        lc = col.lower()
        if 'nanostrain' in lc:
            base = col.replace('_nanostrain','').replace('nanostrain','').strip('_')
            df[base] = df[col] * 1e-9
    # ensure some columns exist; if not, set zeros
    # prefer names: strain_n, strain_e, strain_ne or strain_north/strain_east
    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None
    eps_xx_col = find_col(df, ['strain_N','strain_n','strain_north','strain_x','eps_xx'])
    eps_yy_col = find_col(df, ['strain_E','strain_e','strain_east','strain_y','eps_yy'])
    eps_xy_col = find_col(df, ['strain_NE','strain_ne','strain_ne','strain_xy','eps_xy'])
    if eps_xx_col is None:
        df['eps_xx'] = 0.0
    else:
        df['eps_xx'] = df[eps_xx_col].astype(float)
    if eps_yy_col is None:
        df['eps_yy'] = 0.0
    else:
        df['eps_yy'] = df[eps_yy_col].astype(float)
    if eps_xy_col is None:
        df['eps_xy'] = 0.0
    else:
        df['eps_xy'] = df[eps_xy_col].astype(float)
    # normalize time to naive UTC
    try:
        df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('UTC').dt.tz_localize(None)
    except Exception:
        df['time'] = pd.to_datetime(df['time'])
    return df[['time','eps_xx','eps_yy','eps_xy']].copy()

# -------------------------------------------
# tidal stress interpolation
# -------------------------------------------
from scipy import interpolate

def build_tidal_interpolator(strain_df: pd.DataFrame,
                             E=100e9, nu=0.25, mu=0.6,
                             projection_mode='optimally_oriented',
                             strike=None, dip=None, rake=None,
                             sigma1_azimuth=0.0):
    """
    Return a function compute_tidal_stress(times) -> dict with coulomb_stress, normal_stress, shear_stress, volumetric_strain
    projection_mode: 'optimally_oriented' or 'fault_plane'
    If 'fault_plane', pass strike,dip,rake in degrees.
    """
    times = strain_df['time']
    t_secs = to_epoch_seconds(times)
    eps_xx = strain_df['eps_xx'].values
    eps_yy = strain_df['eps_yy'].values
    eps_xy = strain_df['eps_xy'].values

    # Elastic constants -> stresses (plane strain)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    G = E / (2 * (1 + nu))
    eps_zz = -nu / (1 - nu) * (eps_xx + eps_yy)
    trace = eps_xx + eps_yy + eps_zz

    sigma_xx = 2 * G * eps_xx + lam * trace
    sigma_yy = 2 * G * eps_yy + lam * trace
    sigma_zz = 2 * G * eps_zz + lam * trace
    sigma_xy = 2 * G * eps_xy

    # Build 3x3 stress tensor as (3,3,N)
    N = len(t_secs)
    st = np.zeros((3,3,N))
    st[0,0,:] = sigma_xx
    st[1,1,:] = sigma_yy
    st[2,2,:] = sigma_zz
    st[0,1,:] = sigma_xy
    st[1,0,:] = sigma_xy

    # Compute Coulomb projection depending on mode
    if projection_mode == 'fault_plane':
        if strike is None or dip is None or rake is None:
            raise ValueError("strike,dip,rake must be provided for 'fault_plane' projection.")
        n, s = fault_vectors(strike, dip, rake, degrees=True)
        coulomb_series, sigma_n_series, tau_series = coulomb_on_plane(st, n, s, mu=mu)
    elif projection_mode == 'optimally_oriented':
        # we'll do 2D projection similar to earlier: strike = sigma1_azimuth + 45deg
        strike_rad = sigma1_azimuth + np.pi/4.0
        # construct 3-vector normal and slip with zero vertical
        n2 = np.array([np.sin(strike_rad), np.cos(strike_rad), 0.0])
        t2 = np.array([np.cos(strike_rad), -np.sin(strike_rad), 0.0])
        # traction vector components: t_x = s_xx * n_x + s_xy * n_y (2D)
        t_x = sigma_xx * n2[0] + sigma_xy * n2[1]
        t_y = sigma_xy * n2[0] + sigma_yy * n2[1]
        # normal stress and shear magnitude
        sigma_n_series = t_x * n2[0] + t_y * n2[1]
        tau_series = t_x * t2[0] + t_y * t2[1]
        coulomb_series = tau_series - mu * sigma_n_series
    else:
        raise ValueError("projection_mode must be 'optimally_oriented' or 'fault_plane'")
    
    # Check if Coulomb stress calculation was successful
    if np.all(coulomb_series == 0):
        print("WARNING: All Coulomb stress values are zero. Check fault orientation and strain data.")
    else:
        print(f"INFO: Coulomb stress range: [{np.nanmin(coulomb_series):.1f}, {np.nanmax(coulomb_series):.1f}] Pa")

    # Also create interpolators
    print(f"Creating interpolators with t_secs shape: {t_secs.shape}, type: {type(t_secs)}")
    print(f"Coulomb series shape: {coulomb_series.shape}, type: {type(coulomb_series)}")
    interp_cfs = safe_interp1d(t_secs, coulomb_series, kind='cubic', bounds_error=False, fill_value=np.nan)
    interp_xx = safe_interp1d(t_secs, sigma_xx, kind='cubic', bounds_error=False, fill_value=np.nan)
    interp_xy = safe_interp1d(t_secs, sigma_xy, kind='cubic', bounds_error=False, fill_value=np.nan)
    interp_trace = safe_interp1d(t_secs, trace, kind='cubic', bounds_error=False, fill_value=np.nan)

    def compute_tidal_stress(event_times):
        secs = to_epoch_seconds(event_times)
        # Ensure secs is a numpy array for interpolation
        secs = np.asarray(secs)
        cfs_result = interp_cfs(secs)
        return {
            'coulomb_stress': cfs_result,
            'normal_stress': interp_xx(secs),
            'shear_stress': interp_xy(secs),
            'volumetric_strain': interp_trace(secs)
        }

    return compute_tidal_stress

# ----------------------------------
# Analyzer: phase + amplitude statistics
# ----------------------------------
class TidalSensitivityAnalyzer:
    def __init__(self, phase_bins=20, stress_bins=20):
        self.phase_bins = phase_bins
        self.stress_bins = stress_bins

    # compute_phase_and_amplitude
    # - returns phase (radians) and analytic amplitude for a stress time series

    def compute_phase_and_amplitude(self, stress_series):
        stress = np.asarray(stress_series, dtype=float)
        finite = np.isfinite(stress)
        if finite.sum() < 4 or np.allclose(stress[finite], stress[finite][0]):
            return np.full(len(stress), np.nan), np.zeros(len(stress))
        stress_clean = np.nan_to_num(stress, nan=0.0)
        analytic = hilbert(stress_clean)
        return np.angle(analytic), np.abs(analytic)

    # compute_ratio_distributions
    # - builds R_phi (phase ratio) and R_sigma (amplitude ratio) comparing event
    #   subset to reference distribution. Uses histogram normalization.
    def compute_ratio_distributions(self, eq_indices, ref_indices, phases, amplitudes):
        phase_edges = np.linspace(-np.pi, np.pi, self.phase_bins + 1)
        eq_hist, _ = np.histogram(phases[eq_indices], bins=phase_edges)
        ref_hist, _ = np.histogram(phases[ref_indices], bins=phase_edges)
        ref_hist = np.maximum(ref_hist, 1)
        eq_dist = eq_hist / max(1, len(eq_indices))
        ref_dist = ref_hist / max(1, len(ref_indices))
        R_phi = eq_dist / ref_dist
        R_phi = np.where(np.isfinite(R_phi) & (R_phi > 0), R_phi, 1.0)

        # amplitude ratio
        valid_amps = amplitudes[~np.isnan(amplitudes)]
        if len(valid_amps) == 0:
            R_sigma = np.ones(self.stress_bins)
        else:
            amp_edges = np.linspace(valid_amps.min(), valid_amps.max(), self.stress_bins + 1)
            eq_amp_hist, _ = np.histogram(amplitudes[eq_indices], bins=amp_edges)
            ref_amp_hist, _ = np.histogram(amplitudes[ref_indices], bins=amp_edges)
            ref_amp_hist = np.maximum(ref_amp_hist, 1)
            eq_amp_dist = eq_amp_hist / max(1, len(eq_indices))
            ref_amp_dist = ref_amp_hist / max(1, len(ref_indices))
            R_sigma = eq_amp_dist / ref_amp_dist
            R_sigma = np.where(np.isfinite(R_sigma) & (R_sigma > 0), R_sigma, 1.0)
            if not np.any(np.isfinite(R_sigma)):
                R_sigma = np.ones_like(R_sigma)

        return R_phi, R_sigma

    # cosine_model
    # - simple parametric model used to fit phase modulation amplitude (alpha)
    def cosine_model(self, phi, alpha, phi_0):
        """Cosine model: R_phi = 1 + alpha * cos(phi - phi_0)"""
        return 1 + alpha * np.cos(phi - phi_0)

    # fit_phase_modulation
    # - fit the cosine_model to binned R_phi and return (alpha, phi_0)
    def fit_phase_modulation(self,R_phi):
        phase_centers = np.linspace(-np.pi, np.pi, len(R_phi), endpoint=False)
        mask = np.isfinite(R_phi) & (R_phi > 0)
        if mask.sum() < 3:
            return 0.0, 0.0
        xs, ys = phase_centers[mask], np.asarray(R_phi)[mask]
        try:
            popt, _ = curve_fit(
                self.cosine_model, xs, ys,
                p0=[0.1, 0.0],
                bounds=[[-1.0, -np.pi], [1.0, np.pi]],
                maxfev=10000
            )
            alpha, phi_0 = popt
        except Exception:
            alpha, phi_0 = 0.0, 0.0
        return alpha, phi_0

    # fit_amplitude_modulation
    # - linear fit to amplitude-ratio curve; returns slope-based metrics
    def fit_amplitude_modulation(self, R_sigma, amplitudes):
        amps = np.asarray(amplitudes)
        valid = amps[~np.isnan(amps)]
        if len(valid) == 0:
            return 0.0, np.inf
        centers = np.linspace(valid.min(), valid.max(), len(R_sigma))
        try:
            coeffs = np.polyfit(centers, R_sigma, 1)
            gamma = float(coeffs[0])
            a_sigma_n = (1.0 / gamma) if (gamma != 0.0) else np.inf
            return gamma, a_sigma_n
        except Exception:
            return 0.0, np.inf

    # bootstrap_alpha_phase_randomization
    # - generate null distribution of alpha by randomly reassigning event indices
    def bootstrap_alpha_phase_randomization(self, phases_ref, eq_indices, ref_indices, n_boot=500, random_state=None):
        """Phase-randomization bootstrap: shuffle event positions by drawing random ref indices (without replacement) for events"""
        rs = np.random.RandomState(seed=random_state)
        # observed R_phi
        R_obs, _ = self.compute_ratio_distributions(eq_indices, ref_indices, phases_ref, np.zeros_like(phases_ref))
        alpha_obs, _ = self.fit_phase_modulation(R_obs)
        if alpha_obs == 0.0:
            return alpha_obs, 1.0, 0.0
        boot_alphas = []
        n_events = len(eq_indices)
        ref_pool = np.array(ref_indices)
        for _ in range(n_boot):
            # sample random indices (without replacement) of same count to simulate random event phase assignment
            sampled = rs.choice(ref_pool, size=n_events, replace=False)
            R_boot, _ = self.compute_ratio_distributions(sampled, ref_indices, phases_ref, np.zeros_like(phases_ref))
            a_b, _ = self.fit_phase_modulation(R_boot)
            boot_alphas.append(a_b)
        boot_alphas = np.array(boot_alphas)
        crit95 = np.nanpercentile(boot_alphas, 95)
        p_val = np.mean(boot_alphas >= alpha_obs)
        return float(alpha_obs), float(p_val), float(crit95)

    # main analysis functions
    def analyze_tidal_sensitivity(self, catalog, tidal_stress_func):
        # Check catalog time column
        print(f"Catalog shape: {catalog.shape}")
        print(f"Catalog time column type: {type(catalog['time'].iloc[0])}")
        print(f"Catalog time range: {catalog['time'].min()} to {catalog['time'].max()}")
        
        # Ensure times are datetime
        if not pd.api.types.is_datetime64_any_dtype(catalog['time']):
            print("Converting time column to datetime...")
            catalog = catalog.copy()
            catalog['time'] = pd.to_datetime(catalog['time'])
        
        # Check for NaT values
        nat_count = catalog['time'].isna().sum()
        if nat_count > 0:
            print(f"Warning: Found {nat_count} NaT values in time column, removing...")
            catalog = catalog.dropna(subset=['time']).reset_index(drop=True)
        
        # build reference hourly grid
        start_time = catalog['time'].min()
        end_time = catalog['time'].max()
        
        # Validate start/end times
        if pd.isna(start_time) or pd.isna(end_time):
            raise ValueError(f"Invalid time range: start={start_time}, end={end_time}")
            
        print(f"Analysis time range: {start_time} to {end_time}")
        ref_times = pd.date_range(start=start_time, end=end_time, freq='1h')
        tidal_ref = tidal_stress_func(ref_times)

        # map events to nearest ref index
        ref_secs = to_epoch_seconds(ref_times)
        evt_secs = to_epoch_seconds(catalog['time'])
        inds = np.searchsorted(ref_secs, evt_secs)
        inds_clipped = np.clip(inds, 0, len(ref_secs)-1)
        prev = np.clip(inds_clipped-1, 0, len(ref_secs)-1)
        choose_prev = (np.abs(ref_secs[prev] - evt_secs) < np.abs(ref_secs[inds_clipped] - evt_secs))
        evt_ref_idx = inds_clipped.copy()
        evt_ref_idx[choose_prev] = prev[choose_prev]

        components = ['coulomb_stress', 'normal_stress', 'volumetric_strain']
        results = {}
        for comp in components:
            series_ref = tidal_ref[comp]
            phases_ref, amps_ref = self.compute_phase_and_amplitude(series_ref)
            ref_indices = np.arange(len(ref_times))
            eq_indices = evt_ref_idx
            R_phi, R_sigma = self.compute_ratio_distributions(eq_indices, ref_indices, phases_ref, amps_ref)
            alpha, phi0 = self.fit_phase_modulation(R_phi)
            event_amps = amps_ref[eq_indices]
            gamma, a_sigma_n = self.fit_amplitude_modulation(R_sigma, event_amps)
            results[comp] = {
                'alpha': alpha,
                'phi_0': phi0,
                'gamma': gamma,
                'a_sigma_n': a_sigma_n,
                'R_phi': R_phi,
                'R_sigma': R_sigma,
                'phases': phases_ref,
                'amplitudes': amps_ref,
                'ref_times': ref_times,
                'evt_ref_idx': evt_ref_idx
            }
        return results, tidal_ref

    def analyze_sliding_windows(self, catalog, tidal_stress_func, window_days=90, overlap_percent=0.9, min_events=10):
        window_seconds = window_days * 24 * 3600
        step_seconds = int(window_seconds * (1 - overlap_percent))
        start_time = catalog['time'].min()
        end_time = catalog['time'].max()
        total_seconds = int((end_time - start_time).total_seconds())
        if total_seconds < window_seconds:
            return {
                'coulomb_stress': {'phase_ratios': np.array([]), 'amplitude_ratios': np.array([]), 'phases': [], 'amplitudes': []},
                'normal_stress': {'phase_ratios': np.array([]), 'amplitude_ratios': np.array([]), 'phases': [], 'amplitudes': []},
                'volumetric_strain': {'phase_ratios': np.array([]), 'amplitude_ratios': np.array([]), 'phases': [], 'amplitudes': []}
            }
        n_windows = int((total_seconds - window_seconds) / max(1, step_seconds)) + 1

        ref_times = pd.date_range(start=start_time, end=end_time, freq='1h')
        tidal_ref = tidal_stress_func(ref_times)
        ref_secs = to_epoch_seconds(ref_times)
        evt_secs = to_epoch_seconds(catalog['time'])
        inds = np.searchsorted(ref_secs, evt_secs)
        inds_clipped = np.clip(inds, 0, len(ref_secs)-1)
        prev = np.clip(inds_clipped-1, 0, len(ref_secs)-1)
        choose_prev = (np.abs(ref_secs[prev] - evt_secs) < np.abs(ref_secs[inds_clipped] - evt_secs))
        evt_ref_idx = inds_clipped.copy()
        evt_ref_idx[choose_prev] = prev[choose_prev]

        window_results = {
            'coulomb_stress': {'phase_ratios': [], 'amplitude_ratios': [], 'phases': [], 'amplitudes': []},
            'normal_stress': {'phase_ratios': [], 'amplitude_ratios': [], 'phases': [], 'amplitudes': []},
            'volumetric_strain': {'phase_ratios': [], 'amplitude_ratios': [], 'phases': [], 'amplitudes': []}
        }

        for i in range(n_windows):
            ws = start_time + pd.Timedelta(seconds=i * step_seconds)
            we = ws + pd.Timedelta(seconds=window_seconds)
            window_mask = (catalog['time'] >= ws) & (catalog['time'] < we)
            event_inds = np.where(window_mask)[0]
            if len(event_inds) < min_events:
                continue
            ref_mask = (ref_times >= ws) & (ref_times < we)
            ref_inds_window = np.where(ref_mask)[0]
            if len(ref_inds_window) == 0:
                continue
            evt_ref_in_window = evt_ref_idx[event_inds]
            evt_in_window_mask = np.isin(evt_ref_in_window, ref_inds_window)
            if np.sum(evt_in_window_mask) < min_events:
                continue

            # Compute alpha for each stress component
            alphas = {}
            for comp in ['coulomb_stress', 'normal_stress', 'volumetric_strain']:
                # OPTIMIZATION: compute phase/amplitude only on the data within the current window.
                series_ref_window = tidal_ref[comp][ref_inds_window]
                phases_ref_window, amps_ref_window = analyzer.compute_phase_and_amplitude(series_ref_window)

                # The event and reference indices must be relative to the window's data, not the global series.
                eq_indices_global = evt_ref_in_window[evt_in_window_mask]
                
                # Find the positions of the global event indices within the global window indices array.
                # This maps the global indices to local indices (0 to N-1) relative to the window's start.
                eq_indices_local = np.searchsorted(ref_inds_window, eq_indices_global)
                
                ref_indices_local = np.arange(len(ref_inds_window))

                if len(eq_indices_local) == 0 or len(ref_indices_local) == 0:
                    alphas[comp] = np.nan
                    continue

                # Use the local indices and window-specific phase/amplitude data.
                R_phi, _ = analyzer.compute_ratio_distributions(eq_indices_local, ref_indices_local, phases_ref_window, amps_ref_window)
                
                try:
                    alpha, _ = analyzer.fit_phase_modulation(R_phi)
                    alphas[comp] = alpha
                except Exception:
                    alphas[comp] = np.nan

            window_results['coulomb_stress']['phase_ratios'].append(alphas['coulomb_stress'])
            window_results['coulomb_stress']['amplitude_ratios'].append(R_sigma)
            window_results['coulomb_stress']['phases'].append(phases_ref_window[eq_indices_local])
            window_results['coulomb_stress']['amplitudes'].append(amps_ref_window[eq_indices_local])

        # stack arrays where applicable
        for comp in window_results:
            if len(window_results[comp]['phase_ratios']) == 0:
                window_results[comp]['phase_ratios'] = np.array([])
            else:
                window_results[comp]['phase_ratios'] = np.vstack(window_results[comp]['phase_ratios'])
            if len(window_results[comp]['amplitude_ratios']) == 0:
                window_results[comp]['amplitude_ratios'] = np.array([])
            else:
                window_results[comp]['amplitude_ratios'] = np.vstack(window_results[comp]['amplitude_ratios'])
        return window_results

def compute_alpha_timeseries(catalog, tidal_stress_func, window_days=90, overlap_percent=0.9, min_events=10):
    """
    Compute alpha (phase modulation amplitude) over sliding time windows.
    
    Returns:
    - DataFrame with columns: ['time', 'alpha_coulomb', 'alpha_normal', 'alpha_volumetric', 'n_events']
    """
    analyzer = TidalSensitivityAnalyzer()
    
    window_duration = pd.Timedelta(days=window_days)
    step = window_duration * (1 - overlap_percent)
    
    start_time = catalog['time'].min()
    end_time = catalog['time'].max()
    
    if (end_time - start_time) < window_duration:
        return pd.DataFrame(columns=['time', 'alpha_coulomb', 'alpha_normal', 'alpha_volumetric', 'n_events'])
    
    # Prepare reference time series and pre-compute phases/amplitudes
    ref_times = pd.date_range(start=start_time, end=end_time, freq='1h')
    tidal_ref = tidal_stress_func(ref_times)
    
    # Use pandas rolling windows for efficiency
    events_series = catalog.set_index('time').sort_index()
    
    # Create a series of event counts on the daily grid
    daily_counts = events_series.resample('D').size()
    
    # Use rolling window on the daily counts to find valid window centers
    rolling_counts = daily_counts.rolling(window=f'{window_days}D', center=True).sum()
    
    valid_windows = rolling_counts[rolling_counts >= min_events]
    
    results = []
    
    # Match events to reference times once
    ref_secs = to_epoch_seconds(ref_times)
    evt_secs = to_epoch_seconds(catalog['time'])
    inds = np.searchsorted(ref_secs, evt_secs)
    inds_clipped = np.clip(inds, 0, len(ref_secs)-1)
    prev = np.clip(inds_clipped-1, 0, len(ref_secs)-1)
    choose_prev = (np.abs(ref_secs[prev] - evt_secs) < np.abs(ref_secs[inds_clipped] - evt_secs))
    evt_ref_idx = inds_clipped.copy()
    evt_ref_idx[choose_prev] = prev[choose_prev]
    
    for window_center_time in valid_windows.index:
        ws = window_center_time - window_duration / 2
        we = window_center_time + window_duration / 2
        
        window_mask = (catalog['time'] >= ws) & (catalog['time'] < we)
        event_inds = np.where(window_mask)[0]
        n_events_in_window = len(event_inds)

        if n_events_in_window < min_events:
            continue

        ref_mask = (ref_times >= ws) & (ref_times < we)
        ref_inds_window = np.where(ref_mask)[0]
        
        if len(ref_inds_window) == 0:
            continue
            
        evt_ref_in_window = evt_ref_idx[event_inds]
        evt_in_window_mask = np.isin(evt_ref_in_window, ref_inds_window)
        
        if np.sum(evt_in_window_mask) < min_events:
            continue

        # Compute alpha for each stress component
        alphas = {}
        for comp in ['coulomb_stress', 'normal_stress', 'volumetric_strain']:
            # OPTIMIZATION: compute phase/amplitude only on the data within the current window.
            series_ref_window = tidal_ref[comp][ref_inds_window]
            phases_ref_window, amps_ref_window = analyzer.compute_phase_and_amplitude(series_ref_window)

            # The event and reference indices must be relative to the window's data, not the global series.
            eq_indices_global = evt_ref_in_window[evt_in_window_mask]
            
            # Find the positions of the global event indices within the global window indices array.
            # This maps the global indices to local indices (0 to N-1) relative to the window's start.
            eq_indices_local = np.searchsorted(ref_inds_window, eq_indices_global)
            
            ref_indices_local = np.arange(len(ref_inds_window))

            if len(eq_indices_local) == 0 or len(ref_indices_local) == 0:
                alphas[comp] = np.nan
                continue

            # Use the local indices and window-specific phase/amplitude data.
            R_phi, _ = analyzer.compute_ratio_distributions(eq_indices_local, ref_indices_local, phases_ref_window, amps_ref_window)
            
            try:
                alpha, _ = analyzer.fit_phase_modulation(R_phi)
                alphas[comp] = alpha
            except Exception:
                alphas[comp] = np.nan

        results.append({
            'time': window_center_time,
            'alpha_coulomb': alphas.get('coulomb_stress', np.nan),
            'alpha_normal': alphas.get('normal_stress', np.nan),
            'alpha_volumetric': alphas.get('volumetric_strain', np.nan),
            'n_events': n_events_in_window
        })
        
    return pd.DataFrame(results)

# -----------------------
# Plotting & figures
# -----------------------
def create_paper_figures(results, tidal_data, catalog, window_results, tidal_stress_func, output_prefix='beauce_2023'):
    # Figure 1c: coulomb stress time series
    fig, ax = plt.subplots(figsize=(12,6))
    start_time = catalog['time'].min()
    end_time = catalog['time'].max()
    regular_times = pd.date_range(start=start_time, end=end_time, freq='1h')
    regular_stresses = tidal_stress_func(regular_times)
    valid = ~np.isnan(regular_stresses['coulomb_stress'])
    if np.any(valid):
        ax.plot(regular_times[valid], regular_stresses['coulomb_stress'][valid], color='#1f77b4', linewidth=2)
        try:
            from scipy import interpolate as _interp
            reg_secs = to_epoch_seconds(regular_times)
            reg_stress = np.asarray(regular_stresses['coulomb_stress'])
            curve_interp = _interp.interp1d(reg_secs, reg_stress, kind='cubic', bounds_error=False, fill_value=np.nan)
            evt_secs = to_epoch_seconds(catalog['time'])
            evt_s = curve_interp(evt_secs)
            vs = ~np.isnan(evt_s)
            if np.any(vs):
                ax.scatter(catalog['time'][vs], evt_s[vs], color='red', s=30, zorder=5, label='Events')
        except Exception:
            pass
    ax.set_xlabel('Time'); ax.set_ylabel('Coulomb Stress (Pa)')
    ax.set_title('Figure 1c: Coulomb Stress Time Series')
    ax.grid(alpha=0.3); ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    fig.savefig(f'{output_prefix}_figure_1c.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Figure 2 and others: reuse earlier corrected plotting logic
    # We'll build a similar 4x2 layout as before.
    fig, axes = plt.subplots(4,2, figsize=(15,20))
    fig.suptitle('Figure 2: Tidal Sensitivity (sliding windows)')
    coul = results['coulomb_stress']
    evt_ref_idx = coul.get('evt_ref_idx', None)
    if evt_ref_idx is not None:
        phases = np.asarray(coul['phases'])[evt_ref_idx]
        amps = np.asarray(coul['amplitudes'])[evt_ref_idx]
    else:
        phases = coul['phases']
        amps = coul['amplitudes']

    # panel a: phase scatter
    ax_a = axes[0,0]; ax_a.scatter(catalog['time'], phases, s=6, color='k', alpha=0.6)
    ax_a.set_ylim(-np.pi, np.pi); ax_a.set_title('(a)'); ax_a.set_ylabel('phase [rad]'); ax_a.grid(alpha=0.2)

    # panels b,c: example PDFs (if available)
    ax_b = axes[1,0]; ax_c = axes[2,0]
    total_win = window_results['coulomb_stress']['phase_ratios'].shape[0] if isinstance(window_results['coulomb_stress']['phase_ratios'], np.ndarray) else 0
    if total_win == 0:
        w_inds = []
    else:
        w_inds = [0, min(total_win-1, total_win//2)]
    for ax, wi, label in zip([ax_b, ax_c], w_inds, ['(b)', '(c)']):
        R = window_results['coulomb_stress']['phase_ratios'][wi]
        centers = np.linspace(-np.pi, np.pi, len(R), endpoint=False)
        width = 2*np.pi / max(1, len(R))
        ax.bar(centers, R, width=width, color='#1f77b4', edgecolor='k')
        ax.axhline(1.0, color='k', linestyle='--')
        ax.set_xlim(-np.pi, np.pi); ax.set_title(label); ax.set_xlabel('phase [rad]'); ax.set_ylabel('R_phi'); ax.grid(alpha=0.18)

    # panel d: median across windows with fitted cosine overlay
    ax_d = axes[3,0]
    if total_win == 0:
        median_ratios = np.ones(20)
        lower = np.zeros_like(median_ratios); upper = np.zeros_like(median_ratios)
    else:
        all_pr = window_results['coulomb_stress']['phase_ratios']
        median_ratios = np.nanmedian(all_pr, axis=0)
        lower = np.nanpercentile(all_pr, 16, axis=0)
        upper = np.nanpercentile(all_pr, 84, axis=0)
    centers = np.linspace(-np.pi, np.pi, len(median_ratios), endpoint=False)
    width = 2*np.pi / max(1, len(median_ratios))
    ax_d.bar(centers, median_ratios, width=width, color='#1f77b4', edgecolor='k')
    err_low = median_ratios - lower; err_high = upper - median_ratios
    ax_d.errorbar(centers, median_ratios, yerr=[err_low, err_high], fmt='none', ecolor='k', capsize=3)
    # overlay fitted cosine from results (if present)
    alpha_fit = coul.get('alpha', 0.0); phi0 = coul.get('phi_0', 0.0)
    if alpha_fit != 0.0:
        fitted = 1 + alpha_fit * np.cos(centers - phi0)
        ax_d.plot(centers, fitted, 'k-', lw=2)
    ax_d.set_xlim(-np.pi, np.pi); ax_d.set_title('(d)'); ax_d.set_xlabel('phase [rad]'); ax_d.set_ylabel('median R_phi'); ax_d.grid(alpha=0.18)

    # panel e: amplitude scatter vs time (Hilbert amplitude)
    ax_e = axes[0,1]; ax_e.scatter(catalog['time'], amps, s=6, color='k', alpha=0.6)
    ax_e.set_ylabel('Hilbert amplitude [Pa]'); ax_e.set_title('(e)'); ax_e.grid(alpha=0.2)

    # panels f,g: amplitude ratio PDFs
    ax_f = axes[1,1]; ax_g = axes[2,1]
    if total_win > 0:
        for ax, wi, label in zip([ax_f, ax_g], w_inds, ['(f)', '(g)']):
            R_amp = window_results['coulomb_stress']['amplitude_ratios'][wi]
            amps_w = window_results['coulomb_stress']['amplitudes'][wi]
            vamps = amps_w[~np.isnan(amps_w)]
            if len(vamps) > 0 and len(R_amp)>0:
                centers_a = np.linspace(vamps.min(), vamps.max(), len(R_amp))
                w = (centers_a[1]-centers_a[0]) if len(centers_a)>1 else 0.001
                ax.bar(centers_a, R_amp, width=w, color='#ff7f0e', edgecolor='k')
                ax.axhline(1.0, color='k', linestyle='--')
                ax.set_xlabel('amplitude [Pa]'); ax.set_ylabel('R_sigma'); ax.set_title(label); ax.grid(alpha=0.18)

    # panel h: median amplitude ratio
    ax_h = axes[3,1]
    all_amp_ratios = window_results['coulomb_stress']['amplitude_ratios']
    if isinstance(all_amp_ratios, np.ndarray) and all_amp_ratios.size>0:
        med_amp = np.nanmedian(all_amp_ratios, axis=0)
        first_amps = window_results['coulomb_stress']['amplitudes'][0]
        v = first_amps[~np.isnan(first_amps)]
        if len(v)>0:
            centers_a = np.linspace(v.min(), v.max(), len(med_amp))
            w = (centers_a[1]-centers_a[0]) if len(centers_a)>1 else 0.001
            ax_h.bar(centers_a, med_amp, width=w, color='#ff7f0e', edgecolor='k')
            low = np.nanpercentile(all_amp_ratios, 16, axis=0); up = np.nanpercentile(all_amp_ratios, 84, axis=0)
            ax_h.errorbar(centers_a, med_amp, yerr=[med_amp-low, up-med_amp], fmt='none', ecolor='k', capsize=3)
            gamma = coul.get('gamma', 0.0)
            if gamma != 0.0:
                ax_h.plot(centers_a, 1 + gamma*(centers_a - centers_a.mean()), 'k-')
            ax_h.set_xlabel('Stress amplitude [Pa]'); ax_h.set_ylabel('median R_sigma'); ax_h.set_title('(h)'); ax_h.grid(alpha=0.18)
    else:
        ax_h.text(0.5, 0.5, 'No amplitude ratio windows', ha='center')

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_figure_2.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_figure_3(results, window_results, catalog, output_prefix='beauce_2023', analyzer=None, n_boot=200):
    if analyzer is None:
        analyzer = TidalSensitivityAnalyzer()
    window_days = 90; overlap_percent = 0.9
    window_seconds = window_days * 24 * 3600
    step_seconds = int(window_seconds * (1 - overlap_percent))
    start_time = catalog['time'].min()

    total_win = 0
    if 'coulomb_stress' in window_results and isinstance(window_results['coulomb_stress']['phase_ratios'], np.ndarray):
        total_win = window_results['coulomb_stress']['phase_ratios'].shape[0]

    times = []; alphas = []; phi0s = []; gammas = []; a_sigma_ns = []; crit95s = []; pvals = []
    for wi in range(total_win):
        ws = start_time + pd.Timedelta(seconds=wi * step_seconds)
        times.append(ws + pd.Timedelta(seconds=window_seconds/2))
        R_phi = window_results['coulomb_stress']['phase_ratios'][wi]
        R_sig = window_results['coulomb_stress']['amplitude_ratios'][wi]
        amps = window_results['coulomb_stress']['amplitudes'][wi]

        a, phi0 = analyzer.fit_phase_modulation(R_phi)
        g, a_n = analyzer.fit_amplitude_modulation(R_sig, amps)
        alphas.append(a); phi0s.append(phi0); gammas.append(g); a_sigma_ns.append(a_n)

        # bootstrap via phase-randomization: need phases_ref and full ref indices -> approximate with window phases context
        # We use the phases array from window's per-event phases if present; else use overall ref phases (less ideal)
        # Build phases_ref and ref_indices for this bootstrap: we can reconstruct from window by taking unique bins (coarse approach)
        # Simpler: we will reuse the analyzer.bootstrap that expects phases_ref, eq_indices and ref_indices:
        # For the window, phases_ref is the full set of phases on reference grid in that window; we don't have it here easily,
        # so as a pragmatic approach we approximate by using the per-event phase samples and treat ref_indices as a local range.
        try:
            # build surrogate phases_ref by concatenating per-event phases in window and a grid
            phases_event = window_results['coulomb_stress']['phases'][wi]
            # build a simple ref grid: linspace between -np.pi and np.pi with bins equal to R_phi length
            phases_ref = np.linspace(-np.pi, np.pi, len(R_phi), endpoint=False)
            # eq_indices: map events to nearest phase bin (by np.digitize)
            eq_inds = np.digitize(phases_event, bins=np.linspace(-np.pi, np.pi, len(R_phi)+1)) - 1
            eq_inds = np.clip(eq_inds, 0, len(phases_ref)-1)
            ref_inds = np.arange(len(phases_ref))
            alpha_obs, p_val, crit95 = analyzer.bootstrap_alpha_phase_randomization(phases_ref, eq_inds, ref_inds, n_boot=n_boot, random_state=wi)
        except Exception:
            alpha_obs, p_val, crit95 = a, 1.0, 0.0
        crit95s.append(crit95); pvals.append(p_val)

    times = np.array(times)
    alphas = np.array(alphas)
    phi0s = np.array(phi0s)
    gammas = np.array(gammas)
    a_sigma_ns = np.array(a_sigma_ns)
    crit95s = np.array(crit95s)
    pvals = np.array(pvals)

    # example windows to show fits: choose first, mid, last if exist
    example_inds = [0, total_win//2, max(0, total_win-1)] if total_win>0 else []

    fig = plt.figure(figsize=(12,10))
    gs = fig.add_gridspec(3,3, width_ratios=[1.6,1,1], height_ratios=[1,1,0.9], hspace=0.35, wspace=0.4)
    ax_a = fig.add_subplot(gs[0:2,0])
    phase_plot_x = np.linspace(-np.pi, np.pi, 400)
    for i, wi in enumerate(example_inds):
        if wi >= total_win: continue
        R_phi = window_results['coulomb_stress']['phase_ratios'][wi]
        a_i, phi0_i = analyzer.fit_phase_modulation(R_phi)
        smooth_vals = 1.0 + a_i * np.cos(phase_plot_x - phi0_i)
        lw = 2.5 - 0.6 * i
        ax_a.plot(phase_plot_x, smooth_vals, label=f"{pd.to_datetime(times[wi]).date()}", linewidth=max(0.8, lw))
    ax_a.set_xlabel('phase [rad]'); ax_a.set_ylabel('Obs/Exp rate'); ax_a.set_title('(a) Periodic fits'); ax_a.set_xlim(-np.pi, np.pi); ax_a.grid(alpha=0.12); ax_a.legend(frameon=False)

    # panel b: phi0 time
    ax_b = fig.add_subplot(gs[0,1:3]); ax_b.plot(times, phi0s, 'o'); ax_b.set_ylim(-np.pi, np.pi); ax_b.set_title('(b) phi0 over time'); ax_b.grid(alpha=0.2)

    # panel c: alpha time with bootstrap 95% line
    ax_c = fig.add_subplot(gs[1,1:3]); ax_c.plot(times, alphas, '-o'); 
    if crit95s.size>0:
        median_crit = np.nanmedian(crit95s)
        ax_c.axhline(median_crit, color='k', linestyle='--', label='median 95% crit')
    ax_c.set_ylabel('alpha'); ax_c.set_title('(c) alpha over time'); ax_c.grid(alpha=0.16); ax_c.legend(frameon=False)

    # bottom-left (rate-state)
    ax_f = fig.add_subplot(gs[2,0])
    sigma_range = np.linspace(-750, 1500, 200)
    for wi in example_inds:
        if wi >= len(a_sigma_ns): continue
        a_n = a_sigma_ns[wi]
        if np.isfinite(a_n) and a_n>0:
            ax_f.plot(sigma_range, np.exp(sigma_range / a_n), label=f'{pd.to_datetime(times[wi]).date()}')
    ax_f.set_xlabel('CFS [Pa]'); ax_f.set_ylabel('Obs/Exp'); ax_f.set_title('(f) rate-state'); ax_f.grid(alpha=0.16); ax_f.legend(frameon=False)

    # bottom-middle linear fits
    ax_d = fig.add_subplot(gs[2,1])
    for wi in example_inds:
        if wi >= len(gammas): continue
        g = gammas[wi]
        ax_d.plot(sigma_range, 1 + g * sigma_range, label=f'{pd.to_datetime(times[wi]).date()}')
    ax_d.set_xlabel('CFS [Pa]'); ax_d.set_ylabel('Obs/Exp'); ax_d.set_title('(d) linear'); ax_d.grid(alpha=0.16); ax_d.legend(frameon=False)

    # bottom-right gamma time with percentiles
    ax_e = fig.add_subplot(gs[2,2])
    if gammas.size>0:
        ax_e.plot(times, gammas, '-o')
        p98,p95,p90 = np.nanpercentile(gammas, [98,95,90])
        ax_e.axhline(p98, linestyle='--'); ax_e.axhline(p95, linestyle='-.'); ax_e.axhline(p90, linestyle=':')
    ax_e.set_title('(e) gamma over time'); ax_e.set_ylabel('gamma [1/Pa]'); ax_e.grid(alpha=0.16)

    plt.tight_layout()
    fig.savefig(f'{output_prefix}_figure_3.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    # also save a_sigma_n zoom
    fig2, axg = plt.subplots(figsize=(6,2))
    if a_sigma_ns.size>0:
        med = np.nanmedian(a_sigma_ns); p5,p95 = np.nanpercentile(a_sigma_ns, [5,95])
        mask = (a_sigma_ns >= p5) & (a_sigma_ns <= p95)
        axg.plot(times[mask], a_sigma_ns[mask], 'o', color='brown', markersize=4)
        axg.plot(times[~mask], a_sigma_ns[~mask], 'o', color='gray', alpha=0.25, markersize=3)
        axg.axhline(med, linestyle='--')
    axg.set_title('(g) a_sigma_n over time (zoomed)'); axg.set_ylabel('aσn [Pa]'); axg.grid(alpha=0.2)
    fig2.savefig(f'{output_prefix}_figure_3_asigman.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_strong_modulation_summary(window_results, catalog, output_prefix='beauce_2023', analyzer=None):
    if analyzer is None:
        analyzer = TidalSensitivityAnalyzer()
    window_days = 90; overlap_percent = 0.9
    window_seconds = window_days * 24 * 3600
    step_seconds = int(window_seconds * (1 - overlap_percent))
    start_time = catalog['time'].min()

    phase_ratios = window_results['coulomb_stress']['phase_ratios']
    amps_list = window_results['coulomb_stress']['amplitudes']
    amp_ratios = window_results['coulomb_stress']['amplitude_ratios']
    if not isinstance(phase_ratios, np.ndarray) or phase_ratios.size == 0:
        print("No sliding-window data for summary.")
        return
    n_win = phase_ratios.shape[0]
    times=[]; alphas=[]; gammas=[]; counts=[]
    for wi in range(n_win):
        ws = start_time + pd.Timedelta(seconds=wi * step_seconds)
        times.append(ws + pd.Timedelta(seconds=window_seconds/2))
        R_phi = phase_ratios[wi]
        amps = amps_list[wi]
        R_sig = amp_ratios[wi] if (isinstance(amp_ratios, np.ndarray) and amp_ratios.shape[0]>wi) else np.array([])
        a, _ = analyzer.fit_phase_modulation(R_phi)
        g, _ = analyzer.fit_amplitude_modulation(R_sig, amps)
        alphas.append(a); gammas.append(g); counts.append(np.count_nonzero(~np.isnan(amps)))
    times=np.array(times); alphas=np.array(alphas); gammas=np.array(gammas); counts=np.array(counts)
    thr = np.nanpercentile(alphas, 95)

    fig, ax1 = plt.subplots(figsize=(11,4.5))
    ax1.plot(times, alphas, '-o', lw=1.5, label='alpha')
    strong = alphas >= thr
    for t,s in zip(times, strong):
        if s:
            ax1.axvspan(t - pd.Timedelta(days=window_days/2), t + pd.Timedelta(days=window_days/2), alpha=0.15, color='orange')
    if np.any(np.isfinite(gammas)):
        maxg = np.nanmax(np.abs(gammas)) if np.nanmax(np.abs(gammas))>0 else 1.0
        sizes = 20 + 180*(np.abs(gammas)/maxg)
        ax1.scatter(times, alphas, s=sizes, facecolor='none', edgecolor='k', linewidth=0.6, label='|gamma| scale')
    ax2 = ax1.twinx(); ax2.plot(times, counts, color='gray', alpha=0.6, label='events/window'); ax2.set_ylabel('events/window')
    ax1.axhline(thr, linestyle='--', color='k', label='95% alpha threshold')
    ax1.set_ylabel('alpha'); ax1.set_title('Strong modulation summary'); ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    fig.tight_layout(); fig.savefig(f'{output_prefix}_strong_modulation_summary.png', dpi=300, bbox_inches='tight'); plt.close(fig)

def create_alpha_events_single_plot(window_results, catalog, output_prefix='nm_tidal', analyzer=None, title_suffix=""):
    """Create a single subplot figure showing alpha values and events per window."""
    if analyzer is None:
        analyzer = TidalSensitivityAnalyzer()
    
    window_days = 90
    overlap_percent = 0.9
    window_seconds = window_days * 24 * 3600
    step_seconds = int(window_seconds * (1 - overlap_percent))
    start_time = catalog['time'].min()

    # Extract data from window results
    phase_ratios = window_results['coulomb_stress']['phase_ratios']
    amps_list = window_results['coulomb_stress']['amplitudes']
    
    if not isinstance(phase_ratios, np.ndarray) or phase_ratios.size == 0:
        print("No sliding-window data for alpha-events plot.")
        return
    
    n_win = phase_ratios.shape[0]
    times = []
    alphas = []
    counts = []
    
    for wi in range(n_win):
        ws = start_time + pd.Timedelta(seconds=wi * step_seconds)
        times.append(ws + pd.Timedelta(seconds=window_seconds/2))
        
        R_phi = phase_ratios[wi]
        amps = amps_list[wi]
        
        # Fit phase modulation to get alpha
        a, _ = analyzer.fit_phase_modulation(R_phi)
        alphas.append(a)
        counts.append(np.count_nonzero(~np.isnan(amps)))
    
    times = np.array(times)
    alphas = np.array(alphas)
    counts = np.array(counts)
    
    # Calculate threshold
    thr = np.nanpercentile(alphas, 95)
    
    # Create single subplot figure
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot alpha values
    line1 = ax1.plot(times, alphas, '-o', color='red', linewidth=2, markersize=4, 
                     label='Alpha (phase modulation)', alpha=0.8)
    ax1.axhline(thr, linestyle='--', color='red', linewidth=2, alpha=0.7,
                label=f'95% threshold ({thr:.3f})')
    
    # Highlight strong modulation periods
    strong = alphas >= thr
    for t, s in zip(times, strong):
        if s:
            ax1.axvspan(t - pd.Timedelta(days=window_days/2), 
                       t + pd.Timedelta(days=window_days/2), 
                       alpha=0.2, color='red', linewidth=0)
    
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Alpha (Phase Modulation Strength)', color='red', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for events per window
    ax2 = ax1.twinx()
    line2 = ax2.plot(times, counts, '-s', color='blue', linewidth=2, markersize=4,
                     label='Events per window', alpha=0.8)
    ax2.set_ylabel('Events per Window', color='blue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, 
              fancybox=True, shadow=True, fontsize=10)
    
    # Title
    title = f'Alpha Values and Events per Window{title_suffix}'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    
    # Improve layout
    fig.tight_layout()
    plt.savefig(f'{output_prefix}_alpha_events_single{title_suffix.replace(" ", "_").lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created single subplot alpha-events plot: {output_prefix}_alpha_events_single{title_suffix.replace(' ', '_').lower()}.png")


# -----------------------
# Beaucé et al. (2023) Style Figures
# -----------------------

def create_figure_1c(catalog, tidal_stress_func, output_prefix='nm_tidal'):
    """Create Figure 1c: Coulomb stress time series (Beaucé et al. 2023 style)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    start_time = catalog['time'].min()
    end_time = catalog['time'].max()
    regular_times = pd.date_range(start=start_time, end=end_time, freq='1h')
    regular_stresses = tidal_stress_func(regular_times)
    
    valid = ~np.isnan(regular_stresses['coulomb_stress'])
    if np.any(valid):
        ax.plot(regular_times[valid], regular_stresses['coulomb_stress'][valid], 
                color='lightgray', linewidth=2)
        
        # Add event markers
        try:
            from scipy import interpolate as _interp
            reg_secs = to_epoch_seconds(regular_times)
            reg_stress = np.asarray(regular_stresses['coulomb_stress'])
            curve_interp = _interp.interp1d(reg_secs, reg_stress, kind='cubic', 
                                          bounds_error=False, fill_value=np.nan)
            evt_secs = to_epoch_seconds(catalog['time'])
            evt_stress = curve_interp(evt_secs)
            vs = ~np.isnan(evt_stress)
            if np.any(vs):
                ax.scatter(catalog['time'][vs], evt_stress[vs], color='black', s=30, 
                          zorder=5, marker='o')
        except Exception as e:
            print(f"Event interpolation failed: {e}")
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Coulomb Stress (Pa)', fontsize=12)
    ax.set_title('Figure 1c: Coulomb Stress Time Series', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Format x-axis to show only years
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    fig.savefig(f'{output_prefix}_figure_1c.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created Figure 1c: {output_prefix}_figure_1c.png")

def create_figure1c_histogram(catalog, tidal_stress_func, output_prefix='nm_tidal'):
    """Create histogram of tidal Coulomb stress values at event times."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute stresses at event times
    event_stresses = tidal_stress_func(catalog['time'])
    
    valid = ~np.isnan(event_stresses['coulomb_stress'])
    if np.any(valid):
        coulomb_values = event_stresses['coulomb_stress'][valid]
        ax.hist(coulomb_values, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Coulomb Stress (Pa)')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram of Tidal Coulomb Stress at Events')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(f'{output_prefix}_figure1c_histogram.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created Figure 1c Histogram: {output_prefix}_figure1c_histogram.png")

def create_figure_2(results, window_results, catalog, output_prefix='nm_tidal', analyzer=None):
    """Create Figure 2: Tidal Sensitivity Analysis (Beaucé et al. 2023 style)."""
    if analyzer is None:
        analyzer = TidalSensitivityAnalyzer()
    
    # Extract data
    coul = results['coulomb_stress']
    evt_ref_idx = coul.get('evt_ref_idx', None)
    
    if evt_ref_idx is not None:
        phases = np.asarray(coul['phases'])[evt_ref_idx]
        amps = np.asarray(coul['amplitudes'])[evt_ref_idx]
    else:
        phases = coul['phases']
        amps = coul['amplitudes']
    
    # Create 4x2 subplot layout
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle('Figure 2: Tidal Sensitivity Analysis (Sliding Windows)', 
                fontsize=16, fontweight='bold')
    
    # Panel (a): Phase scatter over time
    ax_a = axes[0, 0]
    ax_a.scatter(catalog['time'], phases, s=6, color='k', alpha=0.6)
    ax_a.set_ylim(-np.pi, np.pi)
    ax_a.set_title('(a) Phase vs Time')
    ax_a.set_ylabel('Phase [rad]')
    ax_a.grid(alpha=0.2)
    ax_a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Get window information
    total_win = 0
    if 'coulomb_stress' in window_results and isinstance(window_results['coulomb_stress']['phase_ratios'], np.ndarray):
        total_win = window_results['coulomb_stress']['phase_ratios'].shape[0]
    
    w_inds = [0, min(total_win-1, total_win//2)] if total_win > 0 else []
    
    # Panels (b,c): Phase ratio PDFs
    for i, (ax, wi, label) in enumerate(zip([axes[1, 0], axes[2, 0]], w_inds, ['(b)', '(c)'])):
        if total_win > 0 and wi < len(window_results['coulomb_stress']['phase_ratios']):
            R = window_results['coulomb_stress']['phase_ratios'][wi]
            centers = np.linspace(-np.pi, np.pi, len(R), endpoint=False)
            width = 2*np.pi / max(1, len(R))
            ax.bar(centers, R, width=width, color='#1f77b4', edgecolor='k', alpha=0.7)
            ax.axhline(1.0, color='k', linestyle='--', linewidth=2)
            ax.set_xlim(-np.pi, np.pi); ax.set_title(label); ax.set_xlabel('phase [rad]'); ax.set_ylabel('R_phi'); ax.grid(alpha=0.18)
        else:
            ax.text(0.5, 0.5, f'{label} No data', ha='center', va='center', transform=ax.transAxes)
    
    # Panel (d): Median phase ratios
    ax_d = axes[3, 0]
    if total_win > 0:
        all_pr = window_results['coulomb_stress']['phase_ratios']
        median_ratios = np.nanmedian(all_pr, axis=0)
        lower = np.nanpercentile(all_pr, 16, axis=0)
        upper = np.nanpercentile(all_pr, 84, axis=0)
        
        centers = np.linspace(-np.pi, np.pi, len(median_ratios), endpoint=False)
        width = 2*np.pi / max(1, len(median_ratios))
        ax_d.bar(centers, median_ratios, width=width, color='#1f77b4', edgecolor='k', alpha=0.7)
        
        err_low = median_ratios - lower
        err_high = upper - median_ratios
        ax_d.errorbar(centers, median_ratios, yerr=[err_low, err_high], 
                     fmt='none', ecolor='k', capsize=3)
        
        # Add fitted cosine
        alpha_fit = coul.get('alpha', 0.0)
        phi0 = coul.get('phi_0', 0.0)
        if alpha_fit != 0.0:
            fitted = 1 + alpha_fit * np.cos(centers - phi0)
            ax_d.plot(centers, fitted, 'k-', lw=3, label='Fitted Cosine')
            ax_d.legend()
        
        ax_d.set_xlim(-np.pi, np.pi)
        ax_d.set_title('(d) Median Phase Ratios')
        ax_d.set_xlabel('Phase [rad]')
        ax_d.set_ylabel('Median R_phi')
        ax_d.grid(alpha=0.18)
    else:
        ax_d.text(0.5, 0.5, '(d) No window data', ha='center', va='center', transform=ax_d.transAxes)
    
    # Panel (e): Amplitude scatter
    ax_e = axes[0, 1]
    ax_e.scatter(catalog['time'], amps, s=6, color='k', alpha=0.6)
    ax_e.set_title('(e) Amplitude vs Time')
    ax_e.set_ylabel('Hilbert Amplitude [Pa]')
    ax_e.grid(alpha=0.2)
    ax_e.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Panels (f,g): Amplitude ratio PDFs
    for i, (ax, wi, label) in enumerate(zip([axes[1, 1], axes[2, 1]], w_inds, ['(f)', '(g)'])):
        if (total_win > 0 and wi < len(window_results['coulomb_stress']['amplitude_ratios']) and
            wi < len(window_results['coulomb_stress']['amplitudes'])):
            
            R_amp = window_results['coulomb_stress']['amplitude_ratios'][wi]
            amps_w = window_results['coulomb_stress']['amplitudes'][wi]
            vamps = amps_w[~np.isnan(amps_w)]
            
            if len(vamps) > 0 and len(R_amp) > 0:
                centers_a = np.linspace(vamps.min(), vamps.max(), len(R_amp))
                width = (centers_a[1] - centers_a[0]) if len(centers_a) > 1 else 0.001
                ax.bar(centers_a, R_amp, width=width, color='#ff7f0e', edgecolor='k', alpha=0.7)
                ax.axhline(1.0, color='k', linestyle='--', linewidth=2)
                ax.set_title(f'{label} Window {wi+1} Amplitude PDF')
                ax.set_xlabel('Amplitude [Pa]')
                ax.set_ylabel('R_sigma')
                ax.grid(alpha=0.18)
        else:
            ax.text(0.5, 0.5, f'{label} No data', ha='center', va='center', transform=ax.transAxes)
    
    # Panel (h): Median amplitude ratios
    ax_h = axes[3, 1]
    all_amp_ratios = window_results['coulomb_stress']['amplitude_ratios']
    
    if isinstance(all_amp_ratios, np.ndarray) and all_amp_ratios.size > 0 and total_win > 0:
        med_amp = np.nanmedian(all_amp_ratios, axis=0)
        first_amps = window_results['coulomb_stress']['amplitudes'][0]
        vamps = first_amps[~np.isnan(first_amps)]
        
        if len(vamps) > 0:
            centers_a = np.linspace(vamps.min(), vamps.max(), len(med_amp))
            width = (centers_a[1] - centers_a[0]) if len(centers_a) > 1 else 0.001
            ax_h.bar(centers_a, med_amp, width=width, color='#ff7f0e', edgecolor='k', alpha=0.7)
            
            low = np.nanpercentile(all_amp_ratios, 16, axis=0)
            up = np.nanpercentile(all_amp_ratios, 84, axis=0)
            ax_h.errorbar(centers_a, med_amp, yerr=[med_amp-low, up-med_amp], 
                         fmt='none', ecolor='k', capsize=3)
            
            # Add fitted line
            gamma = coul.get('gamma', 0.0)
            if gamma != 0.0:
                fitted_amp = 1 + gamma * (centers_a - centers_a.mean())
                ax_h.plot(centers_a, fitted_amp, 'k-', lw=3, label='Fitted Line')
                ax_h.legend()
            
            ax_h.set_title('(h) Median Amplitude Ratios')
            ax_h.set_xlabel('Stress Amplitude [Pa]')
            ax_h.set_ylabel('Median R_sigma')
            ax_h.grid(alpha=0.18)
    else:
        ax_h.text(0.5, 0.5, '(h) No amplitude data', ha='center', va='center', transform=ax_h.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_figure_2.png', dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    # Lightweight CLI runner placed at end of file so helper functions are available
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=str, required=True, help='Path to earthquake catalog CSV')
    p.add_argument('--spotl-output', type=str, required=True, help='Path to SPOTL strain CSV')
    p.add_argument('--grace-file', type=str, default=None, help='GRACE NetCDF file (optional)')
    p.add_argument('--projection', type=str, default='fault_plane', choices=['optimally_oriented','fault_plane'], help='Projection mode')
    p.add_argument('--strike', type=float, default=None, help='Fault strike (deg)')
    p.add_argument('--dip', type=float, default=None, help='Fault dip (deg)')
    p.add_argument('--rake', type=float, default=None, help='Fault rake (deg)')
    p.add_argument('--grace-lon-min', type=float, default=-92)
    p.add_argument('--grace-lon-max', type=float, default=-88)
    p.add_argument('--grace-lat-min', type=float, default=34)
    p.add_argument('--grace-lat-max', type=float, default=38)
    p.add_argument('--output-prefix', type=str, default='nm_tidal', help='Output prefix for plots')
    p.add_argument('--n-bootstrap', type=int, default=200, help='Bootstrap iterations')
    p.add_argument('--output-alpha-timeseries', type=str, default=None, help='CSV path to write alpha timeseries')

    args = p.parse_args()

    # Handle common typo in GRACE file extension
    import os

    # Reconstruct minimal args namespace expected by the original runner
    class _Args:
        pass
    runner_args = _Args()
    for key, val in vars(args).items():
        setattr(runner_args, key, val)

    # Load catalog and strain
    catalog = load_catalog_for_paper(runner_args.catalog)
    strain_df = load_spotl_strain_csv(runner_args.spotl_output)

    # Build tidal interpolator
    compute_tide = build_tidal_interpolator(strain_df,
                                            projection_mode='fault_plane' if runner_args.projection == 'fault_plane' else 'optimally_oriented',
                                            strike=runner_args.strike, dip=runner_args.dip, rake=runner_args.rake)

    original_compute_tide = compute_tide  # Save tidal-only version

    # If GRACE provided, load it and clip to catalog time range
    grace_callable = None
    grace_df = None
    if runner_args.grace_file:
        try:
            # Get catalog time range for clipping
            catalog_start = catalog['time'].min()
            catalog_end = catalog['time'].max()
            print(f"Clipping GRACE data to catalog time range: {catalog_start} to {catalog_end}")
            
            grace_callable, grace_df = load_grace_data(runner_args.grace_file,
                                                       runner_args.grace_lon_min, runner_args.grace_lon_max,
                                                       runner_args.grace_lat_min, runner_args.grace_lat_max,
                                                       strike=runner_args.strike, dip=runner_args.dip, rake=runner_args.rake,
                                                       start_time=catalog_start, end_time=catalog_end)
            print(f"Loaded GRACE data: {len(grace_df)} data points from {grace_df['time'].min()} to {grace_df['time'].max()}")
        except Exception as e:
            print(f"Warning: could not load GRACE file: {e}")

    analyzer = TidalSensitivityAnalyzer()

    # Compute phases and amplitudes for all events
    tidal_stresses = compute_tide(to_epoch_seconds(catalog['time']))
    if isinstance(tidal_stresses, dict):
        coulomb = tidal_stresses['coulomb_stress']
    else:
        coulomb = tidal_stresses
    phases, amplitudes = analyzer.compute_phase_and_amplitude(coulomb)

    # Compute tidal-only phases and amplitudes
    tidal_only_stresses = original_compute_tide(to_epoch_seconds(catalog['time']))
    if isinstance(tidal_only_stresses, dict):
        tidal_only_coulomb = tidal_only_stresses['coulomb_stress']
    else:
        tidal_only_coulomb = tidal_only_stresses
    tidal_only_phases, tidal_only_amplitudes = analyzer.compute_phase_and_amplitude(tidal_only_coulomb)

    # Compute sliding window results
    window_days = 90
    overlap_percent = 0.9
    window_seconds = window_days * 24 * 3600
    step_seconds = int(window_seconds * (1 - overlap_percent))
    start_time = catalog['time'].min()
    n_win = int((catalog['time'].max() - start_time).total_seconds() / step_seconds) + 1

    phase_ratios = []
    amplitude_ratios = []
    phases_list = []
    amplitudes_list = []

    for wi in range(n_win):
        ws = start_time + pd.Timedelta(seconds=wi * step_seconds)
        we = ws + pd.Timedelta(seconds=window_seconds)
        mask = (catalog['time'] >= ws) & (catalog['time'] < we)
        eq_indices = np.where(mask)[0]
        if len(eq_indices) < 10:  # min_events
            continue
        ref_indices = np.arange(len(phases))
        R_phi, R_sigma = analyzer.compute_ratio_distributions(eq_indices, ref_indices, phases, amplitudes)
        phase_ratios.append(R_phi)
        amplitude_ratios.append(R_sigma)
        phases_list.append(phases[eq_indices])
        amplitudes_list.append(amplitudes[eq_indices])

    window_results = {
        'coulomb_stress': {
            'phase_ratios': np.array(phase_ratios),
            'amplitude_ratios': np.array(amplitude_ratios),
            'phases': phases_list,
            'amplitudes': amplitudes_list
        }
    }

    # Compute tidal-only window results
    tidal_only_phase_ratios = []
    tidal_only_amplitude_ratios = []
    tidal_only_phases_list = []
    tidal_only_amplitudes_list = []
    for wi in range(n_win):
        ws = start_time + pd.Timedelta(seconds=wi * step_seconds)
        we = ws + pd.Timedelta(seconds=window_seconds)
        mask = (catalog['time'] >= ws) & (catalog['time'] < we)
        eq_indices = np.where(mask)[0]
        if len(eq_indices) < 10:
            continue
        ref_indices = np.arange(len(tidal_only_phases))
        R_phi, R_sigma = analyzer.compute_ratio_distributions(eq_indices, ref_indices, tidal_only_phases, tidal_only_amplitudes)
        tidal_only_phase_ratios.append(R_phi)
        tidal_only_amplitude_ratios.append(R_sigma)
        tidal_only_phases_list.append(tidal_only_phases[eq_indices])
        tidal_only_amplitudes_list.append(tidal_only_amplitudes[eq_indices])

    window_results_tidal_only = {
        'coulomb_stress': {
            'phase_ratios': np.array(tidal_only_phase_ratios),
            'amplitude_ratios': np.array(tidal_only_amplitude_ratios),
            'phases': tidal_only_phases_list,
            'amplitudes': tidal_only_amplitudes_list
        }
    }

    results = {'coulomb_stress': {'phases': phases, 'amplitudes': amplitudes}}

    # Create figures
    try:
        create_figure_1c(catalog, compute_tide, output_prefix=runner_args.output_prefix)
    except Exception as e:
        print(f"Could not create figure 1c: {e}")
    try:
        create_figure1c_histogram(catalog, compute_tide, output_prefix=runner_args.output_prefix)
    except Exception as e:
        print(f"Could not create figure 1c histogram: {e}")
    try:
        create_figure_2(results, window_results, catalog, output_prefix=runner_args.output_prefix)
    except Exception as e:
        print(f"Could not create figure 2: {e}")
    try:
        create_figure_3(results, window_results, catalog, output_prefix=runner_args.output_prefix, analyzer=analyzer, n_boot=min(runner_args.n_bootstrap, 200))
    except Exception as e:
        print(f"Could not create figure 3: {e}")
    try:
        create_alpha_events_single_plot(window_results, catalog, output_prefix=runner_args.output_prefix)
    except Exception as e:
        print(f"Could not create alpha events single plot: {e}")
    try:
        create_alpha_events_single_plot(window_results_tidal_only, catalog, output_prefix=runner_args.output_prefix + '_tidal_only', analyzer=analyzer, title_suffix=" (Tidal Only)")
    except Exception as e:
        print(f"Could not create tidal-only alpha events single plot: {e}")
    try:
        create_strong_modulation_summary(window_results, catalog, output_prefix=runner_args.output_prefix, analyzer=analyzer)
    except Exception as e:
        print(f"Could not create strong modulation summary: {e}")

    # Save alpha timeseries if requested
    if runner_args.output_alpha_timeseries:
        try:
            # Build alpha timeseries DataFrame from window results
            alpha_times = []
            alpha_coulomb_values = []
            
            # Create reference times at hourly intervals for the entire catalog
            ref_times = pd.date_range(start=catalog['time'].min(), end=catalog['time'].max(), freq='1h')
            ref_stresses = compute_tide(to_epoch_seconds(ref_times))
            if isinstance(ref_stresses, dict):
                ref_coulomb = ref_stresses['coulomb_stress']
            else:
                ref_coulomb = ref_stresses
            
            # Compute phases and amplitudes for the ENTIRE reference series once
            # (Hilbert transform should be computed on the full series, not windows)
            ref_phases_full, ref_amps_full = analyzer.compute_phase_and_amplitude(ref_coulomb)
            
            for wi in range(n_win):
                ws = start_time + pd.Timedelta(seconds=wi * step_seconds)
                we = ws + pd.Timedelta(seconds=window_seconds)
                window_center = ws + pd.Timedelta(seconds=window_seconds/2)
                
                # Get earthquakes in this window
                eq_mask = (catalog['time'] >= ws) & (catalog['time'] < we)
                eq_indices_global = np.where(eq_mask)[0]
                
                if len(eq_indices_global) < 10:
                    continue
                
                # Get reference times in this window
                ref_mask = (ref_times >= ws) & (ref_times < we)
                ref_inds_window = np.where(ref_mask)[0]
                
                if len(ref_inds_window) == 0:
                    continue
                
                # Extract window-specific phases and amplitudes from the full series
                phases_ref_window = ref_phases_full[ref_inds_window]
                amps_ref_window = ref_amps_full[ref_inds_window]
                
                # Map earthquakes to their nearest reference time within the window
                # This follows the same approach as the main sliding window analysis
                ref_secs = to_epoch_seconds(ref_times)
                evt_secs = to_epoch_seconds(catalog['time'])
                evt_ref_idx = np.searchsorted(ref_secs, evt_secs)
                evt_ref_idx = np.clip(evt_ref_idx, 0, len(ref_secs)-1)
                
                # Get reference indices for earthquakes in this window
                eq_ref_indices = evt_ref_idx[eq_indices_global]
                eq_in_window_mask = np.isin(eq_ref_indices, ref_inds_window)
                
                if np.sum(eq_in_window_mask) < 10:
                    continue
                
                # Map to local indices within the window
                eq_ref_in_window = eq_ref_indices[eq_in_window_mask]
                eq_indices_local = np.searchsorted(ref_inds_window, eq_ref_in_window)
                ref_indices_local = np.arange(len(ref_inds_window))
                
                # Compute ratio distribution and fit alpha
                R_phi, _ = analyzer.compute_ratio_distributions(eq_indices_local, ref_indices_local, 
                                                                 phases_ref_window, amps_ref_window)
                alpha_fit, _ = analyzer.fit_phase_modulation(R_phi)
                
                alpha_times.append(window_center)
                alpha_coulomb_values.append(alpha_fit)
            
            alpha_df = pd.DataFrame({
                'time': alpha_times,
                'alpha_coulomb': alpha_coulomb_values
            })
            
            alpha_df.to_csv(runner_args.output_alpha_timeseries, index=False)
            print(f"Saved alpha timeseries to: {runner_args.output_alpha_timeseries}")
            
            # Create alpha vs GRACE plot if GRACE is loaded
            if grace_df is not None:
                try:
                    alpha_times_dt = pd.to_datetime(alpha_df['time'])
                    
                    # Determine the correct column name for GRACE stress
                    grace_stress_col = 'dCFS' if 'dCFS' in grace_df.columns else 'load_pa'
                    grace_interp = pd.Series(grace_df[grace_stress_col].values, index=pd.to_datetime(grace_df['time']))
                    grace_at_alpha = grace_interp.reindex(alpha_times_dt, method='nearest').values
                    
                    fig, ax1 = plt.subplots(figsize=(12, 6))
                    ax1.plot(alpha_times_dt, alpha_df['alpha_coulomb'], 'b-', label='Alpha (phase modulation)', linewidth=2)
                    ax1.set_ylabel('Alpha', color='b', fontsize=12)
                    ax1.tick_params(axis='y', labelcolor='b')
                    
                    ax2 = ax1.twinx()
                    ax2.plot(alpha_times_dt, grace_at_alpha, 'r-', label='GRACE Coulomb Stress', linewidth=2)
                    ax2.set_ylabel('GRACE Coulomb Stress (Pa)', color='r', fontsize=12)
                    ax2.tick_params(axis='y', labelcolor='r')
                    
                    ax1.set_xlabel('Time', fontsize=12)
                    ax1.set_title('Alpha vs GRACE Coulomb Stress Comparison', fontsize=14, fontweight='bold')
                    ax1.grid(alpha=0.3)
                    fig.tight_layout()
                    plt.savefig(f"{runner_args.output_prefix}_alpha_grace_comparison.png", dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved alpha vs GRACE comparison plot: {runner_args.output_prefix}_alpha_grace_comparison.png")
                except Exception as e:
                    print(f"Could not create alpha vs GRACE plot: {e}")
        except Exception as e:
            print(f"Could not compute/save alpha timeseries: {e}")