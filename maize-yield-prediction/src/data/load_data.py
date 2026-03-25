"""
Module for loading datasets from files.
Extracted from: maize_yield_prediction.py
"""

import numpy as np
import pandas as pd
import h5py


def load_phenotype_data(phe_path, low_memory=False):
    """
    Load phenotype data from CSV file.
    
    Args:
        phe_path (str): Path to phenotype CSV file
        low_memory (bool): Whether to use low memory mode in pandas
        
    Returns:
        pd.DataFrame: Phenotype data
    """
    df_phe = pd.read_csv(phe_path, low_memory=low_memory)
    return df_phe


def load_weather_data(wea_path, low_memory=False):
    """
    Load weather data from CSV file.
    
    Args:
        wea_path (str): Path to weather CSV file
        low_memory (bool): Whether to use low memory mode in pandas
        
    Returns:
        pd.DataFrame: Weather data
    """
    df_wea = pd.read_csv(wea_path, low_memory=low_memory)
    return df_wea


def get_genotype_info(gen_path):
    """
    Get genotype information from HDF5 file.
    
    Args:
        gen_path (str): Path to HDF5 genotype file
        
    Returns:
        tuple: (n_taxa, n_snps) - number of plant lines and SNP markers
    """
    with h5py.File(gen_path, 'r') as f:
        n_taxa = len([k for k in f['Taxa'].keys() if k != 'TaxaOrder'])
        n_snps = f['Positions']['Positions'].shape[0]
    
    return n_taxa, n_snps


def print_dataset_info(df_phe, df_wea, n_taxa, n_snps):
    """
    Print summary statistics of loaded datasets.
    
    Args:
        df_phe (pd.DataFrame): Phenotype data
        df_wea (pd.DataFrame): Weather data
        n_taxa (int): Number of plant lines
        n_snps (int): Number of SNP markers
    """
    print('Loading datasets:')
    print('Phenotype:', df_phe.shape[0])
    print('Weather:', df_wea.shape[0])
    print('Plant Lines:', n_taxa)
    print('SNP Markers:', n_snps)
