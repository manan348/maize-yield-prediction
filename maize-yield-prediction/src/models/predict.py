"""
Module for making yield predictions with trained model.
Extracted from: maize_yield_prediction.py
"""

import numpy as np
import pandas as pd
import h5py
from src.features.build_features import FEAT_COLS


def lookup_precomputed_prediction(df_preds, parent1, parent2, location):
    """
    Lookup a precomputed prediction for a parent pair and location.

    Supports swapped parent order fallback.

    Args:
        df_preds (pd.DataFrame): Dataframe with Female, Male, Location, Yield columns
        parent1 (str): First parent
        parent2 (str): Second parent
        location (str): Field location

    Returns:
        float or None: Predicted yield if found
    """
    res = df_preds[
        (df_preds["Female"] == parent1)
        & (df_preds["Male"] == parent2)
        & (df_preds["Location"] == location)
    ]

    if len(res) == 0:
        res = df_preds[
            (df_preds["Female"] == parent2)
            & (df_preds["Male"] == parent1)
            & (df_preds["Location"] == location)
        ]

    return round(float(res.iloc[0]["Yield"]), 2) if len(res) > 0 else None


def predict_yield_from_model(
    parent1,
    parent2,
    location,
    df,
    gen_path,
    taxa_lookup,
    geno_lower,
    rf,
    pca,
    scaler_snp,
    scaler_final,
    n_snps=5000,
    verbose=True,
):
    """
    Predict yield for a hybrid cross using genomic and environmental data.
    
    Args:
        parent1 (str): Female parent name
        parent2 (str): Male parent name
        location (str): Field location
        df (pd.DataFrame): Phenotype data containing location weather info
        gen_path (str): Path to HDF5 genotype file
        taxa_lookup (dict): Mapping of clean IDs to HDF5 keys
        rf: Trained Random Forest model
        pca: Fitted PCA transformer
        scaler_snp: Scaler for SNP data
        scaler_final: Scaler for final features
        n_snps (int): Number of SNPs to use
        verbose (bool): Whether to print prediction details
        
    Returns:
        float or None: Predicted yield in bu/A, or None if prediction failed
    """
    from src.data.preprocess import match_id
    
    # Match parent IDs
    p1 = match_id(parent1, geno_lower)
    p2 = match_id(parent2, geno_lower)

    if p1 is None:
        print(f'"{parent1}" not in genotype database')
        return None
    if p2 is None:
        print(f"'{parent2}' not in genotype database")
        return None
    if p1 not in taxa_lookup:
        print(f"'{parent1}' not in taxa lookup")
        return None
    if p2 not in taxa_lookup:
        print(f"'{parent2}' not in taxa lookup")
        return None

    # Get location data
    loc_data = df[df['Field-Location'] == location]
    if len(loc_data) == 0:
        print(f"Location '{location}' not found")
        print(f"Available: {sorted(df['Field-Location'].unique())}")
        return None

    # Load SNPs for parents
    with h5py.File(gen_path, 'r') as f:
        snp1 = f['Genotypes'][taxa_lookup[p1]]['calls'][:n_snps].astype(np.float32)
        snp2 = f['Genotypes'][taxa_lookup[p2]]['calls'][:n_snps].astype(np.float32)
        snp1[snp1 == -1] = 0
        snp2[snp2 == -1] = 0

    # Combine environment features
    env = loc_data[FEAT_COLS].mean().values
    
    # Build feature vector
    snp_c = np.concatenate([snp1, snp2]).reshape(1, -1)
    snp_sc = scaler_snp.transform(snp_c)
    snp_pca = pca.transform(snp_sc)
    x = np.concatenate([snp_pca, env.reshape(1, -1)], axis=1)

    # Make prediction
    x_sc = scaler_final.transform(x)
    pred = round(float(rf.predict(x_sc)[0]), 2)
    
    if verbose:
        cat = ('High' if pred >= 170 else
               'Medium' if pred >= 150 else 'Low')
        print(f"{parent1} × {parent2} @ {location}")
        print(f"Predicted: {pred} bu/A ({cat})")
    
    return pred


def predict_all_crosses_for_location(
    df,
    gen_path,
    taxa_lookup,
    geno_lower,
    rf,
    pca,
    scaler_snp,
    scaler_final,
    location,
    n_snps=5000,
):
    """
    Predict yields for all parent combinations at a given location.
    
    Args:
        df (pd.DataFrame): Phenotype data
        gen_path (str): Path to HDF5 genotype file
        taxa_lookup (dict): Mapping of clean IDs to HDF5 keys
        rf: Trained Random Forest model
        pca: Fitted PCA transformer
        scaler_snp: Scaler for SNP data
        scaler_final: Scaler for final features
        location (str): Field location
        n_snps (int): Number of SNPs to use
        
    Returns:
        pd.DataFrame: Predictions with columns [Female, Male, Location, Yield]
    """
    all_p1 = sorted(df['clean_p1'].dropna().unique())
    all_p2 = sorted(df['clean_p2'].dropna().unique())

    results = []
    total = len(all_p1) * len(all_p2)
    count = 0

    for p1 in all_p1:
        for p2 in all_p2:
            # Get original parent names (reverse lookup)
            rows_p1 = df[df['clean_p1'] == p1]['Parent1'].unique()
            rows_p2 = df[df['clean_p2'] == p2]['Parent2'].unique()
            
            if len(rows_p1) > 0 and len(rows_p2) > 0:
                parent1 = rows_p1[0]
                parent2 = rows_p2[0]
                
                pred = predict_yield_from_model(parent1, parent2, location, df, gen_path,
                                               taxa_lookup, geno_lower, rf, pca, scaler_snp,
                                               scaler_final, n_snps, verbose=False)
                if pred:
                    results.append({
                        'Female': p1,
                        'Male': p2,
                        'Location': location,
                        'Yield': pred
                    })
            
            count += 1
            if count % 10000 == 0:
                print(f"{count:,}/{total:,} done...")

    return pd.DataFrame(results)


def get_best_locations_for_cross(
    parent1,
    parent2,
    df,
    gen_path,
    taxa_lookup,
    geno_lower,
    rf,
    pca,
    scaler_snp,
    scaler_final,
    n_snps=5000,
):
    """
    Find best locations for a given hybrid cross.
    
    Args:
        parent1 (str): Female parent name
        parent2 (str): Male parent name
        df (pd.DataFrame): Phenotype data
        gen_path (str): Path to HDF5 genotype file
        taxa_lookup (dict): Mapping of clean IDs to HDF5 keys
        rf: Trained Random Forest model
        pca: Fitted PCA transformer
        scaler_snp: Scaler for SNP data
        scaler_final: Scaler for final features
        n_snps (int): Number of SNPs to use
        
    Returns:
        pd.DataFrame: Locations ranked by predicted yield
    """
    locations = sorted(df['Field-Location'].unique())
    results = []

    for loc in locations:
        pred = predict_yield_from_model(parent1, parent2, loc, df, gen_path, 
                                       taxa_lookup, geno_lower, rf, pca, scaler_snp,
                                       scaler_final, n_snps, verbose=False)
        if pred:
            results.append({'Location': loc, 'Yield': pred})

    return pd.DataFrame(results).sort_values('Yield', ascending=False).reset_index(drop=True)


def categorize_yield(yield_value, high_threshold=170, medium_threshold=150):
    """
    Categorize yield prediction as High/Medium/Low.
    
    Args:
        yield_value (float): Predicted yield in bu/A
        high_threshold (float): Threshold for High category
        medium_threshold (float): Threshold for Medium category
        
    Returns:
        str: Category ('High', 'Medium', or 'Low')
    """
    if yield_value >= high_threshold:
        return 'High'
    elif yield_value >= medium_threshold:
        return 'Medium'
    else:
        return 'Low'
