"""
Module for building feature matrices from genomic and environmental data.
Extracted from: maize_yield_prediction.py
"""

import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


N_SNPS = 5000
N_COMP = 10


FEAT_COLS = [
    'Plant Height [cm]', 'Grain Moisture [%]', 'Silk DAP [days]',
    'Pollen DAP [days]', 'Ear Height [cm]', 'location_code',
    'Temperature [C]', 'Relative Humidity [%]', 'Rainfall [mm]', 'Solar Radiation [W/m2]',
    'Wind Speed [m/s]', 'Photoperiod [hours]', 'Temp_mean_crit', 'Temp_max_crit', 'Rain_crit',
    'Solar_crit', 'Humid_crit'
]


def load_snps_from_hdf5(gen_path, df_pheno, taxa_lookup, n_snps=N_SNPS):
    """
    Load SNP data from HDF5 file for all samples in phenotype dataframe.
    
    Args:
        gen_path (str): Path to HDF5 genotype file
        df_pheno (pd.DataFrame): Phenotype data with clean_p1 and clean_p2 columns
        taxa_lookup (dict): Mapping of clean IDs to HDF5 keys
        n_snps (int): Number of SNPs to load per parent
        
    Returns:
        tuple: (X_female, X_male, y, X_env)
            - X_female (np.array): Female parent SNP genotypes
            - X_male (np.array): Male parent SNP genotypes  
            - y (np.array): Grain yield values
            - X_env (np.array): Environmental features
    """
    X_female, X_male, y, X_env = [], [], [], []
    
    with h5py.File(gen_path, 'r') as f:
        total = len(df_pheno)
        for i, (_, row) in enumerate(df_pheno.iterrows()):
            if i % 5000 == 0:
                print(f"{i:,}/{total:,} rows processed")

            p1 = row['clean_p1']
            p2 = row['clean_p2']

            if p1 in taxa_lookup and p2 in taxa_lookup:
                # Load SNP genotypes
                snp1 = f['Genotypes'][taxa_lookup[p1]]['calls'][:n_snps].astype(np.float32)
                snp2 = f['Genotypes'][taxa_lookup[p2]]['calls'][:n_snps].astype(np.float32)
                
                # Replace missing values (-1) with 0
                snp1[snp1 == -1] = 0
                snp2[snp2 == -1] = 0

                X_female.append(snp1)
                X_male.append(snp2)
                y.append(row['Grain Yield [bu/A]'])
                
                # Extract environmental features
                X_env.append([float(row[c]) if pd.notna(row[c]) else 0.0 for c in FEAT_COLS])

    X_female = np.array(X_female, dtype=np.float32)
    X_male = np.array(X_male, dtype=np.float32)
    X_env = np.array(X_env, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print('SNPs data loaded')
    print(f"Samples: {len(y):,}")
    print(f"SNP shape: {np.concatenate([X_female, X_male], axis=1).shape}")
    print(f"Env shape: {X_env.shape}")

    return X_female, X_male, y, X_env


def build_feature_matrix(X_female, X_male, X_env, n_snps=N_SNPS, n_comp=N_COMP):
    """
    Build combined feature matrix using PCA on SNP data and environmental features.
    
    Process:
    1. Concatenate female and male SNP data
    2. Standardize SNP data
    3. Apply PCA to reduce dimensionality
    4. Combine PCA-transformed SNP data with environmental features
    
    Args:
        X_female (np.array): Female parent SNPs
        X_male (np.array): Male parent SNPs
        X_env (np.array): Environmental features
        n_snps (int): Number of SNPs per parent
        n_comp (int): Number of PCA components
        
    Returns:
        dict: Dictionary containing:
            - 'X_final': Combined feature matrix (SNP-PCA + environment)
            - 'X_snp': Concatenated SNP data
            - 'scaler_snp': Fitted StandardScaler for SNP data
            - 'pca': Fitted PCA model
            - 'scaler_final': Fitted StandardScaler for final features
    """
    # Concatenate SNP data
    X_snp = np.concatenate([X_female, X_male], axis=1)

    # Scale SNPs
    scaler_snp = StandardScaler()
    X_snp_sc = scaler_snp.fit_transform(X_snp)

    # Apply PCA
    pca = PCA(n_components=n_comp, random_state=42)
    X_snp_pca = pca.fit_transform(X_snp_sc)

    # Combine PCA and environment
    X_final = np.concatenate([X_snp_pca, X_env], axis=1)

    print('Features engineered')
    print(f"PCA variance explained: {pca.explained_variance_ratio_.round(3)}")
    print(f"Total variance: {pca.explained_variance_ratio_.sum():.3f}")
    print(f'SNP components: {n_comp}')
    print(f"Env features: {X_env.shape[1]}")
    print(f"Total features: {X_final.shape[1]}")

    return {
        'X_final': X_final,
        'X_snp': X_snp,
        'scaler_snp': scaler_snp,
        'pca': pca,
        'scaler_final': StandardScaler().fit(X_final)
    }


def get_feature_names(n_comp=N_COMP):
    """
    Get names of all features in final feature matrix.
    
    Args:
        n_comp (int): Number of PCA components
        
    Returns:
        list: Feature names (PCA_{1..n_comp} + environmental features)
    """
    pca_names = [f'PCA_{i+1}' for i in range(n_comp)]
    return pca_names + FEAT_COLS
