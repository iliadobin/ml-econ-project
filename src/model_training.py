"""
Model training module for Life Expectancy ML Project.
Trains baseline and advanced models, performs hyperparameter tuning, and saves the best model.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from typing import Dict, Any, Tuple, List
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load processed train, validation, and test data.
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_path = config.PROCESSED_DATA_DIR / config.TRAIN_DATA_FILENAME
    val_path = config.PROCESSED_DATA_DIR / config.VAL_DATA_FILENAME
    test_path = config.PROCESSED_DATA_DIR / config.TEST_DATA_FILENAME
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✅ Loaded train data: {train_df.shape}")
    print(f"✅ Loaded validation data: {val_df.shape}")
    print(f"✅ Loaded test data: {test_df.shape}")
    
    return train_df, val_df, test_df


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Separate features and target variable.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (X, y)
    """
    X = df.drop(columns=[config.TARGET_COLUMN]).values
    y = df[config.TARGET_COLUMN].values
    return X, y


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray, scaler: Any = None) -> Dict[str, float]:
    """
    Evaluate model and return metrics.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target values
        scaler: Optional scaler to transform features
        
    Returns:
        Dictionary of metrics (MAE, RMSE, R²)
    """
    if scaler is not None:
        X = scaler.transform(X)
    
    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def train_baseline_model(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray,
    scaler: Any
) -> Tuple[Any, Dict[str, float]]:
    """
    Train baseline Linear Regression model with cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        scaler: Fitted scaler for feature transformation
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print("\n" + "="*60)
    print("📊 TRAINING BASELINE MODEL: Linear Regression")
    print("="*60)
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    start_time = time.time()
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    print(f"⏱️  Training time: {training_time:.2f}s")
    
    # Cross-validation on training set
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, 
        cv=config.CV_FOLDS, 
        scoring='neg_mean_absolute_error'
    )
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"🔄 Cross-validation MAE: {cv_mae:.3f} (±{cv_std:.3f})")
    
    # Evaluate on train and validation sets
    train_metrics = evaluate_model(model, X_train_scaled, y_train)
    val_metrics = evaluate_model(model, X_val_scaled, y_val)
    
    print(f"\n📈 Training Metrics:")
    print(f"   MAE:  {train_metrics['mae']:.3f}")
    print(f"   RMSE: {train_metrics['rmse']:.3f}")
    print(f"   R²:   {train_metrics['r2']:.3f}")
    
    print(f"\n📉 Validation Metrics:")
    print(f"   MAE:  {val_metrics['mae']:.3f}")
    print(f"   RMSE: {val_metrics['rmse']:.3f}")
    print(f"   R²:   {val_metrics['r2']:.3f}")
    
    metrics = {
        'model_name': 'Linear Regression',
        'training_time': training_time,
        'cv_mae': cv_mae,
        'cv_std': cv_std,
        'train_mae': train_metrics['mae'],
        'train_rmse': train_metrics['rmse'],
        'train_r2': train_metrics['r2'],
        'val_mae': val_metrics['mae'],
        'val_rmse': val_metrics['rmse'],
        'val_r2': val_metrics['r2']
    }
    
    return model, metrics


def train_gradient_boosting_model(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray,
    scaler: Any
) -> Tuple[Any, Dict[str, float]]:
    """
    Train Gradient Boosting model with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        scaler: Fitted scaler for feature transformation
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print("\n" + "="*60)
    print("🌳 TRAINING GRADIENT BOOSTING MODEL")
    print("="*60)
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 1.0]
    }
    
    print(f"🔍 Hyperparameter search space: {sum([len(v) for v in param_grid.values()])} parameters")
    
    # Randomized search for efficiency
    start_time = time.time()
    model = GradientBoostingRegressor(random_state=config.RANDOM_STATE)
    
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=20,  # Try 20 random combinations
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    best_model = random_search.best_estimator_
    
    print(f"\n⏱️  Training time: {training_time:.2f}s")
    print(f"🏆 Best parameters: {random_search.best_params_}")
    print(f"🔄 Best CV MAE: {-random_search.best_score_:.3f}")
    
    # Evaluate on train and validation sets
    train_metrics = evaluate_model(best_model, X_train_scaled, y_train)
    val_metrics = evaluate_model(best_model, X_val_scaled, y_val)
    
    print(f"\n📈 Training Metrics:")
    print(f"   MAE:  {train_metrics['mae']:.3f}")
    print(f"   RMSE: {train_metrics['rmse']:.3f}")
    print(f"   R²:   {train_metrics['r2']:.3f}")
    
    print(f"\n📉 Validation Metrics:")
    print(f"   MAE:  {val_metrics['mae']:.3f}")
    print(f"   RMSE: {val_metrics['rmse']:.3f}")
    print(f"   R²:   {val_metrics['r2']:.3f}")
    
    metrics = {
        'model_name': 'Gradient Boosting',
        'training_time': training_time,
        'best_params': random_search.best_params_,
        'cv_mae': -random_search.best_score_,
        'train_mae': train_metrics['mae'],
        'train_rmse': train_metrics['rmse'],
        'train_r2': train_metrics['r2'],
        'val_mae': val_metrics['mae'],
        'val_rmse': val_metrics['rmse'],
        'val_r2': val_metrics['r2']
    }
    
    return best_model, metrics


def train_xgboost_model(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray,
    scaler: Any
) -> Tuple[Any, Dict[str, float]]:
    """
    Train XGBoost model with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        scaler: Fitted scaler for feature transformation
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print("\n" + "="*60)
    print("🚀 TRAINING XGBOOST MODEL")
    print("="*60)
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    print(f"🔍 Hyperparameter search space: {sum([len(v) for v in param_grid.values()])} parameters")
    
    # Randomized search for efficiency
    start_time = time.time()
    model = XGBRegressor(
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=25,  # Try 25 random combinations
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    best_model = random_search.best_estimator_
    
    print(f"\n⏱️  Training time: {training_time:.2f}s")
    print(f"🏆 Best parameters: {random_search.best_params_}")
    print(f"🔄 Best CV MAE: {-random_search.best_score_:.3f}")
    
    # Evaluate on train and validation sets
    train_metrics = evaluate_model(best_model, X_train_scaled, y_train)
    val_metrics = evaluate_model(best_model, X_val_scaled, y_val)
    
    print(f"\n📈 Training Metrics:")
    print(f"   MAE:  {train_metrics['mae']:.3f}")
    print(f"   RMSE: {train_metrics['rmse']:.3f}")
    print(f"   R²:   {train_metrics['r2']:.3f}")
    
    print(f"\n📉 Validation Metrics:")
    print(f"   MAE:  {val_metrics['mae']:.3f}")
    print(f"   RMSE: {val_metrics['rmse']:.3f}")
    print(f"   R²:   {val_metrics['r2']:.3f}")
    
    metrics = {
        'model_name': 'XGBoost',
        'training_time': training_time,
        'best_params': random_search.best_params_,
        'cv_mae': -random_search.best_score_,
        'train_mae': train_metrics['mae'],
        'train_rmse': train_metrics['rmse'],
        'train_r2': train_metrics['r2'],
        'val_mae': val_metrics['mae'],
        'val_rmse': val_metrics['rmse'],
        'val_r2': val_metrics['r2']
    }
    
    return best_model, metrics


def compare_models(all_metrics: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare all trained models.
    
    Args:
        all_metrics: List of metrics dictionaries
        
    Returns:
        DataFrame with comparison results
    """
    comparison_df = pd.DataFrame(all_metrics)
    
    # Sort by validation MAE (lower is better)
    comparison_df = comparison_df.sort_values('val_mae')
    
    return comparison_df


def save_best_model(
    model: Any, 
    metrics: Dict[str, float], 
    scaler: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> None:
    """
    Save the best performing model and its metrics.
    
    Args:
        model: Best trained model
        metrics: Model metrics
        scaler: Fitted scaler
        X_test: Test features
        y_test: Test target
    """
    print("\n" + "="*60)
    print("💾 SAVING BEST MODEL")
    print("="*60)
    
    # Evaluate on test set
    X_test_scaled = scaler.transform(X_test)
    test_metrics = evaluate_model(model, X_test_scaled, y_test)
    
    print(f"\n🧪 Test Set Performance:")
    print(f"   MAE:  {test_metrics['mae']:.3f}")
    print(f"   RMSE: {test_metrics['rmse']:.3f}")
    print(f"   R²:   {test_metrics['r2']:.3f}")
    
    # Add test metrics to metrics dict
    metrics['test_mae'] = test_metrics['mae']
    metrics['test_rmse'] = test_metrics['rmse']
    metrics['test_r2'] = test_metrics['r2']
    
    # Save model
    model_path = config.TRAINED_MODELS_DIR / config.BEST_MODEL_FILENAME
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to: {model_path}")
    
    # Save metrics
    metrics_path = config.TRAINED_MODELS_DIR / config.MODEL_METRICS_FILENAME
    
    # Convert numpy types to native Python types for JSON serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = float(value)
        elif isinstance(value, dict):
            metrics_serializable[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                        for k, v in value.items()}
        else:
            metrics_serializable[key] = value
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"✅ Metrics saved to: {metrics_path}")


def main():
    """Main model training pipeline."""
    print("\n" + "="*60)
    print("🚀 LIFE EXPECTANCY MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Load data
    print("\n📂 Loading processed data...")
    train_df, val_df, test_df = load_processed_data()
    
    # Step 2: Prepare data
    X_train, y_train = prepare_data(train_df)
    X_val, y_val = prepare_data(val_df)
    X_test, y_test = prepare_data(test_df)
    
    # Step 3: Load scaler
    scaler_path = config.PREPROCESSING_DIR / config.SCALER_FILENAME
    scaler = joblib.load(scaler_path)
    print(f"✅ Loaded scaler from: {scaler_path}")
    
    # Step 4: Train models
    all_metrics = []
    all_models = []
    
    # Train baseline
    lr_model, lr_metrics = train_baseline_model(X_train, y_train, X_val, y_val, scaler)
    all_metrics.append(lr_metrics)
    all_models.append(('Linear Regression', lr_model))
    
    # Train Gradient Boosting
    gb_model, gb_metrics = train_gradient_boosting_model(X_train, y_train, X_val, y_val, scaler)
    all_metrics.append(gb_metrics)
    all_models.append(('Gradient Boosting', gb_model))
    
    # Train XGBoost
    xgb_model, xgb_metrics = train_xgboost_model(X_train, y_train, X_val, y_val, scaler)
    all_metrics.append(xgb_metrics)
    all_models.append(('XGBoost', xgb_model))
    
    # Step 5: Compare models
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON")
    print("="*60)
    
    comparison_df = compare_models(all_metrics)
    print("\n" + comparison_df[['model_name', 'val_mae', 'val_rmse', 'val_r2']].to_string(index=False))
    
    # Step 6: Select and save best model
    best_idx = comparison_df.index[0]
    best_model_name = all_metrics[best_idx]['model_name']
    best_model = all_models[best_idx][1]
    best_metrics = all_metrics[best_idx]
    
    print(f"\n🏆 Best model: {best_model_name}")
    print(f"   Validation MAE: {best_metrics['val_mae']:.3f}")
    
    save_best_model(best_model, best_metrics, scaler, X_test, y_test)
    
    print("\n" + "="*60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

