"""
Module for preprocessing and cleaning data.
Extracted from: maize_yield_prediction.py
"""

import numpy as np
import pandas as pd
import h5py


COLS = [
    'Pedigree', 'Field-Location', 'Grain Yield [bu/A]',
    'Plant Height [cm]', 'Grain Moisture [%]', 'Silk DAP [days]',
    'Pollen DAP [days]', 'Ear Height [cm]'
]

WEA_COLS = [
    'Temperature [C]', 'Relative Humidity [%]',
    'Rainfall [mm]', 'Solar Radiation [W/m2]',
    'Wind Speed [m/s]', 'Photoperiod [hours]'
]


def clean_id(x):
    """
    Clean taxon ID from string format.
    Removes parentheses and extracts first part before colon.
    
    Args:
        x: Input taxon ID (can be None or NaN)
        
    Returns:
        str or None: Cleaned ID string
    """
    if pd.isna(x):
        return None
    return str(x).replace('(', '').replace(')', '').split(':')[0].strip()


def build_genotype_lookup(gen_path):
    """
    Build genotype lookup dictionaries from HDF5 file.
    
    Args:
        gen_path (str): Path to HDF5 genotype file
        
    Returns:
        tuple: (GENO_LOWER, TAXA_LOOKUP)
            - GENO_LOWER: Dict of lowercase genotype names to clean IDs
            - TAXA_LOOKUP: Dict of clean IDs to genotype HDF5 keys
    """
    with h5py.File(gen_path, 'r') as f:
        taxa_raw = [k for k in f['Taxa'].keys() if k != 'TaxaOrder']

    taxa_clean = [clean_id(t) for t in taxa_raw]
    GENO_LOWER = {g.lower(): g for g in taxa_clean if g}

    TAXA_LOOKUP = {}
    with h5py.File(gen_path, 'r') as f:
        for key in f['Genotypes'].keys():
            c = clean_id(key)
            if c and c not in TAXA_LOOKUP:
                TAXA_LOOKUP[c] = key

    print(f"Plant lines indexed: {len(GENO_LOWER):,}")
    print(f"Genotype entries: {len(TAXA_LOOKUP):,}")
    
    return GENO_LOWER, TAXA_LOOKUP


def match_id(x, GENO_LOWER):
    """
    Match parent name to genotype ID (case insensitive).
    
    Args:
        x: Input parent name
        GENO_LOWER (dict): Lookup dictionary for genotype matching
        
    Returns:
        str or None: Matched genotype ID
    """
    if pd.isna(x):
        return None
    cleaned = clean_id(str(x))
    if cleaned is None:
        return None
    return GENO_LOWER.get(cleaned.lower(), None)


def prepare_phenotype_data(df_phe, gen_path):
    """
    Prepare phenotype dataset for modeling.
    
    - Selects relevant columns
    - Drops missing yield values
    - Splits pedigree into parent components
    - Matches parents to genotype IDs
    - Filters valid rows
    - Encodes location
    
    Args:
        df_phe (pd.DataFrame): Raw phenotype data
        gen_path (str): Path to HDF5 genotype file (for building lookup)
        
    Returns:
        pd.DataFrame: Prepared phenotype data with parent matching
    """
    GENO_LOWER, TAXA_LOOKUP = build_genotype_lookup(gen_path)
    
    df = df_phe[COLS].dropna(subset=['Grain Yield [bu/A]']).copy()

    # Split pedigree into parents
    df[['Parent1', 'Parent2']] = df['Pedigree'].str.split('/', expand=True)

    # Match parents to genotype
    df['clean_p1'] = df['Parent1'].apply(lambda x: match_id(x, GENO_LOWER))
    df['clean_p2'] = df['Parent2'].apply(lambda x: match_id(x, GENO_LOWER))

    # Keep rows where both parents have valid data
    df = df[
        (df['Plant Height [cm]'] > 0) &
        (df['Silk DAP [days]'] > 0) &
        (df['Pollen DAP [days]'] > 0) &
        (df['Ear Height [cm]'] > 0)
    ].reset_index(drop=True)

    # Encode location
    df['location_code'] = pd.factorize(df['Field-Location'])[0]

    print('Phenotype prepared:')
    print(f'Rows: {df.shape[0]:,}')
    print(f"Hybrids: {df['Pedigree'].nunique():,}")
    print(f"Locations: {df['Field-Location'].nunique():,}")
    
    return df, GENO_LOWER, TAXA_LOOKUP


def process_weather_data(df_wea):
    """
    Process weather data: aggregate by location and time period.
    
    Computes:
    - Growing season (months 5-9) averages
    - Critical period (months 6-8) statistics
    
    Args:
        df_wea (pd.DataFrame): Raw weather data
        
    Returns:
        tuple: (wea_season, wea_crit)
            - wea_season: Growing season weather by location
            - wea_crit: Critical period weather by location
    """
    # Rename location column
    df_wea = df_wea.rename(columns={'Field Location': 'Field-Location'})

    # Convert to numeric
    df_wea['Month'] = pd.to_numeric(df_wea['Month'], errors='coerce')
    for col in WEA_COLS:
        if col in df_wea.columns:
            df_wea[col] = pd.to_numeric(df_wea[col], errors='coerce')

    # Growing season (months 5-9)
    w_season = df_wea[df_wea['Month'].between(5, 9)].copy()
    wea_season = w_season.groupby('Field-Location').agg({
        'Temperature [C]': 'mean',
        'Relative Humidity [%]': 'mean',
        'Rainfall [mm]': 'sum',
        'Solar Radiation [W/m2]': 'mean',
        'Wind Speed [m/s]': 'mean',
        'Photoperiod [hours]': 'mean'
    }).reset_index()

    # Critical period (months 6-8)
    w_crit = df_wea[df_wea['Month'].between(6, 8)].copy()
    wea_crit = w_crit.groupby('Field-Location').agg({
        'Temperature [C]': ['mean', 'max'],
        'Rainfall [mm]': 'sum',
        'Solar Radiation [W/m2]': 'mean',
        'Relative Humidity [%]': 'mean'
    }).reset_index()

    wea_crit.columns = [
        'Field-Location', 'Temp_mean_crit', 'Temp_max_crit',
        'Rain_crit', 'Solar_crit', 'Humid_crit'
    ]

    print('Weather processed:')
    print(f'Season locations: {wea_season.shape[0]}')
    print(f"Critical locations: {wea_crit.shape[0]}")

    return wea_season, wea_crit


def merge_weather_with_phenotype(df_pheno, wea_season, wea_crit):
    """
    Merge weather data with phenotype data.
    
    Fills missing values with column means.
    
    Args:
        df_pheno (pd.DataFrame): Phenotype data with location
        wea_season (pd.DataFrame): Growing season weather
        wea_crit (pd.DataFrame): Critical period weather
        
    Returns:
        pd.DataFrame: Merged data with weather features
    """
    df = df_pheno.merge(wea_season, on='Field-Location', how='left')
    df = df.merge(wea_crit, on='Field-Location', how='left')

    # Fill missing values with mean
    wea_all_cols = WEA_COLS + ['Temp_mean_crit', 'Temp_max_crit', 'Rain_crit',
                               'Solar_crit', 'Humid_crit']

    for col in wea_all_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    print('Weather merger')
    print(f'Final shape: {df.shape}')
    print(f"NaN count: {df[wea_all_cols].isna().sum().sum()}")

    return df
