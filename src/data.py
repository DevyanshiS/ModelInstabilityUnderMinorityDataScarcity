import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pandas.api.types import is_object_dtype, is_categorical_dtype, is_string_dtype
import os

def load_arff_robust(filepath, target_col):
    """
    Loads an ARFF file, processes byte strings, and handles categorical encoding.
    """
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    
    # Decode byte strings to normal strings
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    
    # Categorical encoding for features
    for col in X.columns:
        if (
            is_object_dtype(X[col].dtype)
            or is_categorical_dtype(X[col].dtype)
            or is_string_dtype(X[col].dtype)
        ):
            if X[col].nunique() <= 2:
                X[col] = LabelEncoder().fit_transform(X[col])
            else:
                X = pd.get_dummies(X, columns=[col], drop_first=True)
                
    # Label encode the target variable
    y = LabelEncoder().fit_transform(y)
    
    return X.astype(float).values, y

def load_datasets(data_dir='Datasets', subsample_creditcard=True, creditcard_frac=0.1, random_state=42):
    """
    Loads all required datasets for the experiment.
    
    Parameters:
    - data_dir: Directory where the datasets are located.
    - subsample_creditcard: Whether to subsample the highly imbalanced and large credit card dataset.
    - creditcard_frac: Fraction of credit card dataset to retain if subsampling.
    - random_state: Seed for reproducibility.
    
    Returns:
    - dict: A dictionary with dataset names as keys and (X, y) tuples as values.
    """
    datasets = {}
    
    print("Loading Diabetes dataset...")
    df_diabetes = pd.read_csv(os.path.join(data_dir, 'diabetes.csv'))
    X_diab = df_diabetes.drop(columns=['Outcome']).astype(float).values
    y_diab = df_diabetes['Outcome'].values
    datasets['diabetes'] = (X_diab, y_diab)
    
    print("Loading Sick dataset...")
    X_sick, y_sick = load_arff_robust(os.path.join(data_dir, 'dataset_38_sick.arff'), target_col='Class')
    datasets['sick'] = (X_sick, y_sick)
    
    print("Loading Mammography dataset...")
    X_mammo, y_mammo = load_arff_robust(os.path.join(data_dir, 'phpn1jVwe.arff'), target_col='class')
    datasets['mammography'] = (X_mammo, y_mammo)
    
    print("Loading Credit Card dataset...")
    df_cc = pd.read_csv(os.path.join(data_dir, 'creditcard.csv'))
    
    if subsample_creditcard:
        # Stratified subsampling to preserve class distribution
        print(f"Subsampling Credit Card dataset to {creditcard_frac*100}%...")
        _, df_cc = train_test_split(
            df_cc,
            train_size=creditcard_frac,
            stratify=df_cc['Class'],
            random_state=random_state,
        )
        
    X_cc = df_cc.drop(columns=['Class', 'Time']).astype(float).values
    y_cc = df_cc['Class'].values
    datasets['creditcard'] = (X_cc, y_cc)
    
    # IMPORTANT: do NOT fit preprocessing here.
    # Imputation/scaling is handled inside fold-level evaluation to prevent leakage.
    print("Loaded raw datasets. Fold-level preprocessing is applied during evaluation...")
    for name, (X, y) in datasets.items():
        X_clean = np.nan_to_num(X)
        datasets[name] = (X_clean, y)
        print(f"[{name}] Shape: {X_clean.shape}, Minority Class Ratio: {np.mean(y == 1):.4f}")
        
    return datasets

if __name__ == "__main__":
    # Test data loading
    d = load_datasets(subsample_creditcard=True)
    print("Data loading script executed successfully.")
