"""
Utility functions for the Life Expectancy ML Project.
Includes policy application, simulation, and validation functions.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config


def load_model_and_preprocessing() -> Tuple[Any, Any, List[str]]:
    """
    Load the trained model, preprocessing pipeline, and feature names.
    
    Returns:
        Tuple of (model, scaler, feature_names)
    """
    model_path = config.TRAINED_MODELS_DIR / config.BEST_MODEL_FILENAME
    scaler_path = config.PREPROCESSING_DIR / config.SCALER_FILENAME
    features_path = config.PREPROCESSING_DIR / config.FEATURE_NAMES_FILENAME
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
    
    return model, scaler, feature_names


def apply_policy_changes(
    original_features: Dict[str, float],
    policy_changes: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Apply policy changes to original features.
    
    Args:
        original_features: Dictionary of current feature values
        policy_changes: List of dictionaries with 'feature', 'delta' keys
        
    Returns:
        Dictionary of modified feature values
    """
    modified_features = original_features.copy()
    
    for change in policy_changes:
        feature = change['feature']
        delta = change['delta']
        
        if feature in modified_features:
            new_value = modified_features[feature] + delta
            # Ensure non-negative values for features that shouldn't be negative
            if feature in ['GDP', 'Population', 'percentage expenditure', 'Schooling']:
                new_value = max(0, new_value)
            
            modified_features[feature] = new_value
    
    return modified_features


def simulate_uplift(
    original_features: Dict[str, float],
    modified_features: Dict[str, float],
    model: Any,
    scaler: Any,
    feature_names: List[str]
) -> Tuple[float, float, float]:
    """
    Simulate the uplift in life expectancy after applying policy changes.
    
    Args:
        original_features: Original feature values
        modified_features: Modified feature values after policy application
        model: Trained ML model
        scaler: Fitted preprocessing scaler
        feature_names: List of feature names in correct order
        
    Returns:
        Tuple of (original_prediction, new_prediction, uplift)
    """
    # Prepare data arrays
    original_array = np.array([[original_features.get(f, 0) for f in feature_names]])
    modified_array = np.array([[modified_features.get(f, 0) for f in feature_names]])
    
    # Scale features
    original_scaled = scaler.transform(original_array)
    modified_scaled = scaler.transform(modified_array)
    
    # Make predictions
    original_pred = model.predict(original_scaled)[0]
    new_pred = model.predict(modified_scaled)[0]
    
    uplift = new_pred - original_pred
    
    return original_pred, new_pred, uplift


def validate_policy_feasibility(
    policy_changes: List[Dict[str, Any]],
    reference_data: pd.DataFrame,
    original_features: Dict[str, float]
) -> Dict[str, Any]:
    """
    Validate whether policy changes are feasible based on historical data.
    
    Args:
        policy_changes: List of proposed policy changes
        reference_data: Historical dataset for reference
        original_features: Current feature values
        
    Returns:
        Dictionary with validation results and warnings
    """
    validation_result = {
        'is_feasible': True,
        'warnings': [],
        'recommendations': []
    }
    
    for change in policy_changes:
        feature = change['feature']
        delta = change['delta']
        
        if feature not in reference_data.columns:
            continue
        
        # Get current and proposed values
        current_value = original_features.get(feature, 0)
        proposed_value = current_value + delta
        
        # Calculate percentiles in reference data
        percentile_5 = reference_data[feature].quantile(0.05)
        percentile_95 = reference_data[feature].quantile(0.95)
        
        # Check if proposed value is within reasonable range
        if proposed_value < percentile_5:
            validation_result['warnings'].append(
                f"{feature}: Proposed value {proposed_value:.2f} is below 5th percentile ({percentile_5:.2f})"
            )
        elif proposed_value > percentile_95:
            validation_result['warnings'].append(
                f"{feature}: Proposed value {proposed_value:.2f} is above 95th percentile ({percentile_95:.2f})"
            )
    
    if len(validation_result['warnings']) > 3:
        validation_result['is_feasible'] = False
        validation_result['recommendations'].append(
            "Too many extreme changes proposed. Consider more gradual policy adjustments."
        )
    
    return validation_result


def format_feature_name(feature_name: str) -> str:
    """
    Format feature name for display (handle spaces and make readable).
    
    Args:
        feature_name: Raw feature name
        
    Returns:
        Formatted feature name
    """
    # Remove extra spaces and make title case
    return feature_name.strip().replace('  ', ' ').title()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

