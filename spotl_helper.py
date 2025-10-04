#!/usr/bin/env python3
"""
SPOTL Configuration Helper
Generates SPOTL input files and runs strain calculations for tidal analysis.
"""

import argparse
import os
import subprocess
from datetime import datetime, timedelta
import pandas as pd

def create_spotl_input(location_lon, location_lat, start_date, end_date, output_file, 
                      time_step=3600, constituents=None):
    """
    Create SPOTL input file for strain calculation.
    
    Parameters:
    -----------
    location_lon : float
        Longitude of calculation point (degrees)
    location_lat : float  
        Latitude of calculation point (degrees)
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    output_file : str
        Output CSV file name
    time_step : int
        Time step in seconds (default: 3600 = 1 hour)
    constituents : list
        List of tidal constituents (default: all major)
    """
    
    if constituents is None:
        # Major tidal constituents for solid Earth tides
        constituents = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'Mf', 'Mm']
    
    # Convert dates to year/day/hour format expected by ertid
    from datetime import datetime
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    start_year = start_dt.year
    start_day = start_dt.timetuple().tm_yday  # Day of year
    start_hour = 0
    
    end_year = end_dt.year
    end_day = end_dt.timetuple().tm_yday
    end_hour = 0
    
    time_step_hours = time_step / 3600.0
    
    # ertid input format based on working example
    # Format: start_year, start_day, start_hour, end_year, end_day, end_hour,
    # time_step_hours, 't' for theoretical, lat, lon, 
    # additional params: 0, 0, 3, 0.0, 90.0, 45.0, output files
    input_content = f"""{start_year}
{start_day}
{start_hour}
{end_year}
{end_day}
{end_hour}
{time_step_hours}
t
{location_lat:.6f}
{location_lon:.6f}
0
0
3
0.0
90.0
45.0
strain_N.dat
strain_E.dat
strain_NE.dat
"""
    
    input_filename = output_file.replace('.csv', '_input.txt')
    
    with open(input_filename, 'w') as f:
        f.write(input_content)
    
    print(f"SPOTL input file created: {input_filename}")
    return input_filename

def convert_dat_to_csv(output_csv):
    """
    Convert ertid .dat output files to CSV format.
    
    Parameters:
    -----------
    output_csv : str
        Desired CSV output filename
        
    Returns:
    --------
    bool
        True if conversion successful
    """
    try:
        import numpy as np
        
        # Read the .dat files produced by ertid (single column with strain values)
        strain_N = np.loadtxt('strain_N.dat')
        strain_E = np.loadtxt('strain_E.dat') 
        strain_NE = np.loadtxt('strain_NE.dat')
        
        # The .dat files only contain strain values, not time
        # Time must be reconstructed from start time and time step
        n_points = len(strain_N)
        
        # Ensure all arrays are the same length
        if not (len(strain_N) == len(strain_E) == len(strain_NE)):
            print("Warning: .dat files have different lengths")
            n_points = min(len(strain_N), len(strain_E), len(strain_NE))
            strain_N = strain_N[:n_points]
            strain_E = strain_E[:n_points]
            strain_NE = strain_NE[:n_points]
            
        # Parse the input file to get the actual start date and time step
        input_file = output_csv.replace('.csv', '_input.txt')
        start_year, start_day, time_step_hours = 2024, 1, 1.0  # defaults
        
        try:
            with open(input_file, 'r') as f:
                lines = f.readlines()
                start_year = int(lines[0].strip())
                start_day = int(lines[1].strip())
                # Skip start_hour (line 2), end info (lines 3-5)
                time_step_hours = float(lines[6].strip())
        except Exception as e:
            print(f"Warning: Could not parse input file: {e}, using defaults")
            
        # Convert day of year to actual date
        base_date = datetime(start_year, 1, 1) + timedelta(days=start_day - 1)
        
        # Generate timestamps based on time step
        timestamps = [base_date + timedelta(hours=i * time_step_hours) for i in range(n_points)]
        
        df = pd.DataFrame({
            'time': timestamps,
            'strain_N_nanostrain': strain_N,
            'strain_E_nanostrain': strain_E,
            'strain_NE_nanostrain': strain_NE
        })
        
        df.to_csv(output_csv, index=False)
        
        # Clean up .dat files
        import os
        for f in ['strain_N.dat', 'strain_E.dat', 'strain_NE.dat']:
            if os.path.exists(f):
                os.remove(f)
                
        return True
        
    except Exception as e:
        print(f"Error converting .dat files to CSV: {e}")
        return False

def run_spotl(spotl_executable, input_file, output_file=None, verbose=True):
    """
    Run SPOTL with the given input file.
    
    Parameters:
    -----------
    spotl_executable : str
        Path to SPOTL executable
    input_file : str
        SPOTL input file
    output_file : str
        Expected output file (for verification)
    verbose : bool
        Print detailed output
    """
    
    if not os.path.exists(spotl_executable):
        print(f"Error: SPOTL executable not found at {spotl_executable}")
        print("Please check the path or install SPOTL from:")
        print("http://holt.ess.washington.edu/spotl/")
        return False
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return False
    
    print(f"Running SPOTL with input file: {input_file}")
    
    try:
        # Read input file contents for piping to stdin
        with open(input_file, 'r') as f:
            input_contents = f.read()
        
        # Run SPOTL with input piped to stdin (ertid is interactive)
        cmd = [spotl_executable]
        if verbose:
            print(f"Command: {' '.join(cmd)}")
            print(f"Piping input from: {input_file}")
        
        result = subprocess.run(cmd, input=input_contents, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("✓ SPOTL completed successfully")
            
            # ertid produces .dat files, convert to CSV
            if convert_dat_to_csv(output_file):
                print(f"✓ Converted .dat files to CSV: {output_file}")
            else:
                print("✗ Failed to convert .dat files to CSV")
                return False
                
        else:
            print(f"✗ SPOTL failed with return code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ SPOTL timed out (>1 hour)")
        return False
    except Exception as e:
        print(f"✗ Error running SPOTL: {e}")
        return False
    
    # Verify output file was created
    if output_file and os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024*1024)  # MB
        print(f"✓ Output file created: {output_file} ({file_size:.1f} MB)")
        return True
    elif output_file:
        print(f"✗ Expected output file not found: {output_file}")
        return False
    else:
        print("✓ SPOTL run completed")
        return True

def validate_strain_output(csv_file):
    """
    Validate SPOTL strain output file format.
    
    Parameters:
    -----------
    csv_file : str
        Path to SPOTL output CSV file
    """
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        return False
    
    try:
        df = pd.read_csv(csv_file)
        
        print(f"\n=== STRAIN DATA VALIDATION ===")
        print(f"File: {csv_file}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for required columns
        required_patterns = ['time', 'strain']
        time_col = None
        strain_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if 'time' in col_lower or 'date' in col_lower:
                time_col = col
            if 'strain' in col_lower:
                strain_cols.append(col)
        
        if time_col is None:
            print("✗ No time column found")
            return False
        else:
            print(f"✓ Time column: {time_col}")
        
        if len(strain_cols) == 0:
            print("✗ No strain columns found")
            return False
        else:
            print(f"✓ Strain columns: {strain_cols}")
        
        # Check time range
        df['time_parsed'] = pd.to_datetime(df[time_col])
        time_span = df['time_parsed'].max() - df['time_parsed'].min()
        print(f"✓ Time range: {df['time_parsed'].min()} to {df['time_parsed'].max()}")
        print(f"✓ Duration: {time_span.days} days")
        
        # Check strain values
        for col in strain_cols:
            if df[col].dtype in ['float64', 'int64']:
                strain_min = df[col].min()
                strain_max = df[col].max()
                print(f"✓ {col}: range [{strain_min:.2e}, {strain_max:.2e}]")
        
        print("✓ Strain data validation completed")
        return True
        
    except Exception as e:
        print(f"✗ Error validating strain data: {e}")
        return False

def estimate_catalog_center(catalog_file):
    """
    Estimate the geographic center of an earthquake catalog.
    
    Parameters:
    -----------
    catalog_file : str
        Path to earthquake catalog CSV
        
    Returns:
    --------
    tuple : (longitude, latitude)
    """
    
    try:
        df = pd.read_csv(catalog_file)
        
        # Find longitude and latitude columns
        lon_col = None
        lat_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['longitude', 'lon', 'long']:
                lon_col = col
            elif col_lower in ['latitude', 'lat']:
                lat_col = col
        
        if lon_col is None or lat_col is None:
            print(f"Could not find longitude/latitude columns in {catalog_file}")
            return None, None
        
        # Calculate center
        center_lon = df[lon_col].median()
        center_lat = df[lat_col].median()
        
        print(f"Estimated catalog center: {center_lat:.4f}°N, {center_lon:.4f}°E")
        print(f"Longitude range: {df[lon_col].min():.4f} to {df[lon_col].max():.4f}")
        print(f"Latitude range: {df[lat_col].min():.4f} to {df[lat_col].max():.4f}")
        
        return center_lon, center_lat
        
    except Exception as e:
        print(f"Error estimating catalog center: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='SPOTL configuration and execution helper')
    parser.add_argument('--catalog', type=str, help='Earthquake catalog file (to estimate center)')
    parser.add_argument('--longitude', type=float, help='Longitude for SPOTL calculation')
    parser.add_argument('--latitude', type=float, help='Latitude for SPOTL calculation')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='strain_timeseries.csv', help='Output CSV filename')
    parser.add_argument('--spotl-executable', type=str, help='Path to SPOTL executable')
    parser.add_argument('--time-step', type=int, default=3600, help='Time step in seconds (default: 3600)')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing strain file')
    parser.add_argument('--create-input-only', action='store_true', help='Only create input file, don\'t run SPOTL')
    
    args = parser.parse_args()
    
    # Validation mode
    if args.validate_only:
        if os.path.exists(args.output):
            validate_strain_output(args.output)
        else:
            print(f"File not found: {args.output}")
        return
    
    # Determine location
    if args.longitude is not None and args.latitude is not None:
        lon, lat = args.longitude, args.latitude
    elif args.catalog:
        lon, lat = estimate_catalog_center(args.catalog)
        if lon is None:
            print("Could not determine location. Please specify --longitude and --latitude")
            return
    else:
        print("Please specify either --catalog file or --longitude/--latitude coordinates")
        return
    
    print(f"Using location: {lat:.4f}°N, {lon:.4f}°E")
    
    # Create SPOTL input file
    input_file = create_spotl_input(lon, lat, args.start_date, args.end_date, 
                                   args.output, args.time_step)
    
    if args.create_input_only:
        print(f"Input file created: {input_file}")
        print("To run SPOTL manually:")
        print(f"  {args.spotl_executable or 'spotl'} {input_file}")
        return
    
    # Run SPOTL
    if args.spotl_executable:
        success = run_spotl(args.spotl_executable, input_file, args.output)
        
        if success:
            print("\n=== NEXT STEPS ===")
            print("Run tidal sensitivity analysis:")
            print(f"python tidal_seasonal.py --catalog {args.catalog or 'CATALOG.csv'} --spotl-output {args.output} --projection fault_plane --strike X --dip Y --rake Z")
        
        # Validate output if successful
        if success and os.path.exists(args.output):
            validate_strain_output(args.output)
    else:
        print(f"Input file created: {input_file}")
        print("Please specify --spotl-executable to run SPOTL automatically")
        print("Or run manually with your SPOTL installation")

if __name__ == "__main__":
    main()
