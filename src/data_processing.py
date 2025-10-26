"""
Data loading and preprocessing module for Life Expectancy ML Project.
"""

import kagglehub
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import joblib
import json
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config


def download_dataset() -> pd.DataFrame:
    """
    Download the Life Expectancy dataset from Kaggle.
    
    Returns:
        DataFrame with raw data
    """
    print("📥 Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download(config.DATASET_NAME)
    print(f"✅ Dataset downloaded to: {path}")
    
    csv_file = Path(path) / config.CSV_FILENAME
    df = pd.read_csv(csv_file)
    
    # Save raw data
    raw_data_path = config.RAW_DATA_DIR / config.CSV_FILENAME
    df.to_csv(raw_data_path, index=False)
    print(f"💾 Raw data saved to: {raw_data_path}")
    
    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    Perform basic exploratory data analysis.
    
    Args:
        df: Input DataFrame
    """
    print("\n" + "="*60)
    print("📊 DATASET OVERVIEW")
    print("="*60)
    
    print(f"\n🔢 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    print("\n📋 Columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n🎯 Target variable: {config.TARGET_COLUMN}")
    print(f"   Range: {df[config.TARGET_COLUMN].min():.1f} - {df[config.TARGET_COLUMN].max():.1f} years")
    print(f"   Mean: {df[config.TARGET_COLUMN].mean():.1f} years")
    print(f"   Median: {df[config.TARGET_COLUMN].median():.1f} years")
    
    print("\n❌ Missing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing': missing[missing > 0],
        'Percentage': missing_pct[missing > 0]
    }).sort_values('Percentage', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string())
    else:
        print("   No missing values!")
    
    print(f"\n🌍 Countries: {df['Country'].nunique()}")
    print(f"📅 Years: {df['Year'].min()} - {df['Year'].max()}")
    print(f"🏥 Status distribution:\n{df['Status'].value_counts().to_string()}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and outliers.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "="*60)
    print("🧹 CLEANING DATA")
    print("="*60)
    
    df_clean = df.copy()
    
    # Remove rows with missing target variable
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=[config.TARGET_COLUMN])
    print(f"✅ Removed {initial_rows - len(df_clean)} rows with missing target variable")
    
    # Handle missing values in numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != config.TARGET_COLUMN and df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            n_missing = df_clean[col].isnull().sum()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"✅ Filled {n_missing} missing values in '{col}' with median: {median_val:.2f}")
    
    # Handle missing values in categorical columns
    for col in config.CATEGORICAL_FEATURES:
        if col in df_clean.columns and df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()[0]
            n_missing = df_clean[col].isnull().sum()
            df_clean[col] = df_clean[col].fillna(mode_val)
            print(f"✅ Filled {n_missing} missing values in '{col}' with mode: {mode_val}")
    
    print(f"\n✅ Cleaned data shape: {df_clean.shape}")
    
    return df_clean


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with encoded features
    """
    print("\n" + "="*60)
    print("🔄 ENCODING FEATURES")
    print("="*60)
    
    df_encoded = df.copy()
    
    # One-hot encode Status (Developed vs Developing)
    if 'Status' in df_encoded.columns:
        df_encoded = pd.get_dummies(df_encoded, columns=['Status'], prefix='Status', drop_first=True)
        print(f"✅ One-hot encoded 'Status' column")
    
    # Drop Country column (too many categories, will use aggregate features if needed)
    if 'Country' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['Country'])
        print(f"✅ Dropped 'Country' column (categorical with {df['Country'].nunique()} unique values)")
    
    return df_encoded


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("\n" + "="*60)
    print("✂️ SPLITTING DATA")
    print("="*60)
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=config.TEST_RATIO,
        random_state=config.RANDOM_STATE,
        shuffle=True
    )
    
    # Second split: train vs val
    val_ratio_adjusted = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=config.RANDOM_STATE,
        shuffle=True
    )
    
    print(f"✅ Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"✅ Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"✅ Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def create_preprocessing_pipeline(train_df: pd.DataFrame) -> StandardScaler:
    """
    Create and fit preprocessing pipeline on training data.
    
    Args:
        train_df: Training DataFrame
        
    Returns:
        Fitted StandardScaler
    """
    print("\n" + "="*60)
    print("⚙️ CREATING PREPROCESSING PIPELINE")
    print("="*60)
    
    # Separate features and target
    X_train = train_df.drop(columns=[config.TARGET_COLUMN])
    
    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    print(f"✅ StandardScaler fitted on {X_train.shape[1]} features")
    
    # Save scaler
    scaler_path = config.PREPROCESSING_DIR / config.SCALER_FILENAME
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler saved to: {scaler_path}")
    
    # Save feature names
    feature_names = X_train.columns.tolist()
    features_path = config.PREPROCESSING_DIR / config.FEATURE_NAMES_FILENAME
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"💾 Feature names saved to: {features_path}")
    
    return scaler


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """
    Save processed data to CSV files.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
    """
    print("\n" + "="*60)
    print("💾 SAVING PROCESSED DATA")
    print("="*60)
    
    train_path = config.PROCESSED_DATA_DIR / config.TRAIN_DATA_FILENAME
    val_path = config.PROCESSED_DATA_DIR / config.VAL_DATA_FILENAME
    test_path = config.PROCESSED_DATA_DIR / config.TEST_DATA_FILENAME
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✅ Train data saved to: {train_path}")
    print(f"✅ Validation data saved to: {val_path}")
    print(f"✅ Test data saved to: {test_path}")


def main():
    """Main data processing pipeline."""
    print("\n" + "="*60)
    print("🚀 LIFE EXPECTANCY DATA PROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Download data
    df = download_dataset()
    
    # Step 2: Explore data
    explore_data(df)
    
    # Step 3: Clean data
    df_clean = clean_data(df)
    
    # Step 4: Encode features
    df_encoded = encode_features(df_clean)
    
    # Step 5: Split data
    train_df, val_df, test_df = split_data(df_encoded)
    
    # Step 6: Create preprocessing pipeline
    create_preprocessing_pipeline(train_df)
    
    # Step 7: Save processed data
    save_processed_data(train_df, val_df, test_df)
    
    print("\n" + "="*60)
    print("✅ DATA PROCESSING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

