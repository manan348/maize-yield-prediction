"""
Module for training machine learning models.
Extracted from: maize_yield_prediction.py
"""

import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def train_test_data_split(X_final, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets with scaling.
    
    Args:
        X_final (np.array): Feature matrix
        y (np.array): Target values (yield)
        test_size (float): Proportion of data for testing
        random_state (int): Random seed
        
    Returns:
        dict: Dictionary containing:
            - 'X_train_sc': Scaled training features
            - 'X_test_sc': Scaled testing features
            - 'y_train': Training targets
            - 'y_test': Testing targets
            - 'scaler': Fitted StandardScaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=test_size, random_state=random_state
    )

    scaler_final = StandardScaler()
    X_train_sc = scaler_final.fit_transform(X_train)
    X_test_sc = scaler_final.transform(X_test)

    return {
        'X_train_sc': X_train_sc,
        'X_test_sc': X_test_sc,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler_final
    }


def train_random_forest(X_train_sc, y_train, n_estimators=200, max_depth=10, 
                       min_samples_leaf=5, random_state=42):
    """
    Train Random Forest regressor model.
    
    Args:
        X_train_sc (np.array): Scaled training features
        y_train (np.array): Training targets
        n_estimators (int): Number of trees
        max_depth (int): Maximum tree depth
        min_samples_leaf (int): Minimum samples per leaf
        random_state (int): Random seed
        
    Returns:
        RandomForestRegressor: Trained model
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train_sc, y_train)
    return rf


def evaluate_model(rf, X_train_sc, X_test_sc, y_train, y_test):
    """
    Evaluate model performance on training and test sets.
    
    Args:
        rf (RandomForestRegressor): Trained model
        X_train_sc (np.array): Scaled training features
        X_test_sc (np.array): Scaled testing features
        y_train (np.array): Training targets
        y_test (np.array): Testing targets
        
    Returns:
        dict: Dictionary containing R² scores and metrics
    """
    y_train_pred = rf.predict(X_train_sc)
    y_test_pred = rf.predict(X_test_sc)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print('Model trained')
    print(f'Train R²: {train_r2:.3f}')
    print(f'Test R²: {test_r2:.3f}')

    return {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }


def cross_validate_model(X_final, y, cv=5, n_estimators=200, max_depth=10,
                        min_samples_leaf=5, random_state=42):
    """
    Perform 5-fold cross-validation on the model.
    
    Args:
        X_final (np.array): Feature matrix
        y (np.array): Target values
        cv (int): Number of folds
        n_estimators (int): Number of trees
        max_depth (int): Maximum tree depth
        min_samples_leaf (int): Minimum samples per leaf
        random_state (int): Random seed
        
    Returns:
        np.array: Cross-validation R² scores for each fold
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    cv_scores = cross_val_score(pipe, X_final, y, cv=cv, scoring='r2')
    
    print(f"CV mean R²: {cv_scores.mean():.3f}")
    print(f'CV std: {cv_scores.std():.3f}')
    print(f'Individual folds: {cv_scores}')

    return cv_scores


def get_feature_importance(rf, feature_names):
    """
    Extract feature importance from trained model.
    
    Args:
        rf (RandomForestRegressor): Trained model
        feature_names (list): Names of features
        
    Returns:
        pd.DataFrame: Feature importance sorted by importance
    """
    importances = rf.feature_importances_
    
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return df_imp


def save_model_artifacts(rf, pca, scaler_snp, scaler_final, X_snp, X_env, 
                        X_final, y, df, save_path='./'):
    """
    Save all model artifacts and data to disk.
    
    Args:
        rf (RandomForestRegressor): Trained model
        pca (PCA): Fitted PCA transformer
        scaler_snp (StandardScaler): Scaler for SNP data
        scaler_final (StandardScaler): Scaler for final features
        X_snp (np.array): SNP data
        X_env (np.array): Environmental features
        X_final (np.array): Final feature matrix
        y (np.array): Target values
        df (pd.DataFrame): Original dataframe
        save_path (str): Directory to save files
    """
    np.save(save_path + 'X_snp.npy', X_snp)
    np.save(save_path + 'X_env.npy', X_env)
    np.save(save_path + 'X_final.npy', X_final)
    np.save(save_path + 'y.npy', y)
    df.to_csv(save_path + 'df_raw.csv', index=False)

    pickle.dump(rf, open(save_path + 'best_model.pkl', 'wb'))
    pickle.dump(pca, open(save_path + 'pca.pkl', 'wb'))
    pickle.dump(scaler_snp, open(save_path + 'scaler_snp.pkl', 'wb'))
    pickle.dump(scaler_final, open(save_path + 'scaler_final.pkl', 'wb'))
    joblib.dump(rf, save_path + 'model.joblib')

    print('Model saved to disk')
    print(f'Samples: {len(y):,}')
    print(f"Features: {X_final.shape[1]}")


def load_saved_model(model_path='./model.joblib'):
    """
    Load a trained model from joblib file.

    Args:
        model_path (str): Path to model.joblib file

    Returns:
        object: Loaded model
    """
    return joblib.load(model_path)


def load_model_artifacts(save_path='./'):
    """
    Load all model artifacts from disk.
    
    Args:
        save_path (str): Directory containing saved files
        
    Returns:
        dict: Dictionary containing all loaded artifacts
    """
    X_snp = np.load(save_path + 'X_snp.npy')
    X_env = np.load(save_path + 'X_env.npy')
    X_final = np.load(save_path + 'X_final.npy')
    y = np.load(save_path + 'y.npy')
    df = pd.read_csv(save_path + 'df_raw.csv')

    rf = pickle.load(open(save_path + 'best_model.pkl', 'rb'))
    pca = pickle.load(open(save_path + 'pca.pkl', 'rb'))
    scaler_snp = pickle.load(open(save_path + 'scaler_snp.pkl', 'rb'))
    scaler_final = pickle.load(open(save_path + 'scaler_final.pkl', 'rb'))

    print('Everything loaded')
    print(f'Samples: {len(y):,}')
    print(f'Features: {X_final.shape[1]}')

    return {
        'X_snp': X_snp,
        'X_env': X_env,
        'X_final': X_final,
        'y': y,
        'df': df,
        'rf': rf,
        'pca': pca,
        'scaler_snp': scaler_snp,
        'scaler_final': scaler_final
    }
