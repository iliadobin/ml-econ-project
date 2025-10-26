"""
Model interpretation module using SHAP for Life Expectancy ML Project.
Provides global and local explanations for model predictions.
"""

import shap
import pandas as pd
import numpy as np
import joblib
import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config


def load_model_and_data() -> Tuple[Any, Any, pd.DataFrame, List[str]]:
    """
    Load trained model, scaler, data, and feature names.
    
    Returns:
        Tuple of (model, scaler, data_df, feature_names)
    """
    print("📂 Loading model and data...")
    
    # Load model
    model_path = config.TRAINED_MODELS_DIR / config.BEST_MODEL_FILENAME
    model = joblib.load(model_path)
    print(f"✅ Loaded model from: {model_path}")
    
    # Load scaler
    scaler_path = config.PREPROCESSING_DIR / config.SCALER_FILENAME
    scaler = joblib.load(scaler_path)
    print(f"✅ Loaded scaler from: {scaler_path}")
    
    # Load feature names
    features_path = config.PREPROCESSING_DIR / config.FEATURE_NAMES_FILENAME
    with open(features_path, 'r') as f:
        feature_names = json.load(f)
    print(f"✅ Loaded {len(feature_names)} feature names")
    
    # Load training data (for background samples)
    train_path = config.PROCESSED_DATA_DIR / config.TRAIN_DATA_FILENAME
    train_df = pd.read_csv(train_path)
    print(f"✅ Loaded training data: {train_df.shape}")
    
    return model, scaler, train_df, feature_names


def create_shap_explainer(
    model: Any, 
    X_background: np.ndarray
) -> shap.Explainer:
    """
    Create SHAP explainer for the model.
    
    Args:
        model: Trained model
        X_background: Background dataset for SHAP
        
    Returns:
        SHAP Explainer object
    """
    print("\n🔍 Creating SHAP explainer...")
    
    # Use TreeExplainer for tree-based models (XGBoost, GradientBoosting)
    try:
        # For XGBoost compatibility, use model.predict as callable
        explainer = shap.Explainer(model.predict, X_background)
        print("✅ Created Explainer with model.predict (compatible mode)")
    except Exception as e:
        print(f"⚠️  Error creating explainer: {e}")
        # Last resort: use KernelExplainer (slower but works)
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_background, 100))
        print("✅ Created KernelExplainer (fallback mode)")
    
    return explainer


def compute_global_importance(
    explainer: shap.Explainer,
    X: np.ndarray,
    feature_names: List[str],
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Compute global feature importance using SHAP.
    
    Args:
        explainer: SHAP explainer
        X: Feature matrix
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        Dictionary with SHAP values and feature importance
    """
    print("\n📊 Computing global feature importance...")
    
    # Compute SHAP values
    shap_values_obj = explainer(X)
    
    # Extract values array
    if hasattr(shap_values_obj, 'values'):
        shap_values = shap_values_obj.values
    else:
        shap_values = shap_values_obj
    
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    
    print(f"✅ Computed SHAP values for {len(feature_names)} features")
    print(f"\n🏆 Top {top_n} Most Important Features:")
    print(importance_df.head(top_n).to_string(index=False))
    
    # Get top features
    top_features = importance_df.head(top_n)['feature'].tolist()
    
    return {
        'shap_values': shap_values,
        'shap_values_obj': shap_values_obj,
        'importance_df': importance_df,
        'top_features': top_features,
        'mean_abs_shap': mean_abs_shap
    }


def create_shap_visualizations(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    max_display: int = 15
) -> None:
    """
    Create and save SHAP visualizations.
    
    Args:
        shap_values: SHAP values
        X: Feature matrix
        feature_names: List of feature names
        max_display: Maximum number of features to display
    """
    print("\n📈 Creating SHAP visualizations...")
    
    # Create visualizations directory
    viz_dir = config.MODEL_DIR / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Summary Plot (Bar) - Feature Importance
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig(viz_dir / "shap_importance_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: shap_importance_bar.png")
    
    # 2. Summary Plot (Beeswarm) - Feature Impact
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        X, 
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig(viz_dir / "shap_summary_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: shap_summary_beeswarm.png")
    
    print(f"\n📁 Visualizations saved to: {viz_dir}")


def compute_local_explanation(
    explainer: shap.Explainer,
    X_instance: np.ndarray,
    feature_names: List[str],
    instance_name: str = "Sample"
) -> Dict[str, Any]:
    """
    Compute SHAP values for a single instance (local explanation).
    
    Args:
        explainer: SHAP explainer
        X_instance: Single instance feature vector
        feature_names: List of feature names
        instance_name: Name/identifier for the instance
        
    Returns:
        Dictionary with local SHAP values and explanation
    """
    # Compute SHAP values for single instance
    shap_values_obj = explainer(X_instance)
    
    # Extract values
    if hasattr(shap_values_obj, 'values'):
        shap_values_instance = shap_values_obj.values
    else:
        shap_values_instance = shap_values_obj
    
    # Create explanation dataframe
    explanation_df = pd.DataFrame({
        'feature': feature_names,
        'value': X_instance[0],
        'shap_value': shap_values_instance[0]
    })
    explanation_df['abs_shap'] = explanation_df['shap_value'].abs()
    explanation_df = explanation_df.sort_values('abs_shap', ascending=False)
    
    return {
        'shap_values': shap_values_instance,
        'explanation_df': explanation_df,
        'instance_name': instance_name
    }


def create_local_visualization(
    explainer: shap.Explainer,
    X_instance: np.ndarray,
    feature_names: List[str],
    instance_name: str,
    expected_value: float
) -> None:
    """
    Create waterfall plot for a single instance.
    
    Args:
        explainer: SHAP explainer
        X_instance: Single instance
        feature_names: Feature names
        instance_name: Instance identifier
        expected_value: Base value (average prediction)
    """
    viz_dir = config.MODEL_DIR / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    shap_values_obj = explainer(X_instance)
    
    # Extract values
    if hasattr(shap_values_obj, 'values'):
        shap_values_instance = shap_values_obj.values[0]
    else:
        shap_values_instance = shap_values_obj[0]
    
    # Create waterfall plot
    plt.figure(figsize=(10, 8))
    
    # Create Explanation object for waterfall plot
    explanation = shap.Explanation(
        values=shap_values_instance,
        base_values=expected_value,
        data=X_instance[0],
        feature_names=feature_names
    )
    
    shap.waterfall_plot(explanation, max_display=15, show=False)
    plt.title(f"SHAP Waterfall Plot: {instance_name}")
    plt.tight_layout()
    
    filename = f"shap_waterfall_{instance_name.replace(' ', '_')}.png"
    plt.savefig(viz_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {filename}")


def save_shap_results(
    shap_values: np.ndarray,
    importance_df: pd.DataFrame,
    top_features: List[str],
    feature_names: List[str]
) -> None:
    """
    Save SHAP values and importance results.
    
    Args:
        shap_values: SHAP values array
        importance_df: Feature importance dataframe
        top_features: List of top feature names
        feature_names: All feature names
    """
    print("\n💾 Saving SHAP results...")
    
    # Save SHAP values
    shap_values_path = config.MODEL_DIR / config.SHAP_VALUES_FILENAME
    joblib.dump(shap_values, shap_values_path)
    print(f"✅ SHAP values saved to: {shap_values_path}")
    
    # Save top features
    top_features_path = config.MODEL_DIR / config.TOP_FEATURES_FILENAME
    top_features_data = {
        'top_features': top_features,
        'importance': importance_df.head(len(top_features)).to_dict('records')
    }
    with open(top_features_path, 'w') as f:
        json.dump(top_features_data, f, indent=2)
    print(f"✅ Top features saved to: {top_features_path}")
    
    # Save full importance as CSV
    importance_csv_path = config.MODEL_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_csv_path, index=False)
    print(f"✅ Feature importance saved to: {importance_csv_path}")


def analyze_feature_interactions(
    explainer: shap.Explainer,
    X: np.ndarray,
    feature_names: List[str],
    top_features: List[str]
) -> None:
    """
    Analyze and visualize feature interactions.
    
    Args:
        explainer: SHAP explainer
        X: Feature matrix
        feature_names: List of feature names
        top_features: List of top features to analyze
    """
    print("\n🔗 Analyzing feature interactions...")
    
    viz_dir = config.MODEL_DIR / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Get SHAP values
    try:
        shap_values_obj = explainer(X)
        
        # Extract values
        if hasattr(shap_values_obj, 'values'):
            shap_values = shap_values_obj.values
        else:
            shap_values = shap_values_obj
        
        # Create dependence plots for top features
        for i, feature in enumerate(top_features[:5]):  # Top 5 features
            plt.figure(figsize=(10, 6))
            
            feature_idx = feature_names.index(feature)
            shap.dependence_plot(
                feature_idx,
                shap_values,
                X,
                feature_names=feature_names,
                show=False
            )
            
            plt.title(f"SHAP Dependence Plot: {feature}")
            plt.tight_layout()
            
            filename = f"shap_dependence_{feature.replace(' ', '_').replace('/', '_')}.png"
            plt.savefig(viz_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved dependence plot: {feature}")
    
    except Exception as e:
        print(f"⚠️  Could not create interaction plots: {e}")


def main():
    """Main SHAP interpretation pipeline."""
    print("\n" + "="*60)
    print("🔍 LIFE EXPECTANCY MODEL INTERPRETATION (SHAP)")
    print("="*60)
    
    # Step 1: Load model and data
    model, scaler, train_df, feature_names = load_model_and_data()
    
    # Step 2: Prepare data
    X_train = train_df.drop(columns=[config.TARGET_COLUMN]).values
    X_train_scaled = scaler.transform(X_train)
    
    # Use a sample for background (for efficiency)
    sample_size = min(1000, len(X_train_scaled))
    indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
    X_background = X_train_scaled[indices]
    
    print(f"📊 Using {sample_size} samples for SHAP background")
    
    # Step 3: Create SHAP explainer
    explainer = create_shap_explainer(model, X_background)
    
    # Step 4: Compute global importance
    # Use subset for computation (faster)
    sample_for_shap = min(500, len(X_train_scaled))
    X_for_shap = X_train_scaled[:sample_for_shap]
    
    shap_results = compute_global_importance(
        explainer, 
        X_for_shap, 
        feature_names,
        top_n=10
    )
    
    # Step 5: Create visualizations
    create_shap_visualizations(
        shap_results['shap_values'],
        X_for_shap,
        feature_names,
        max_display=15
    )
    
    # Step 6: Analyze feature interactions
    analyze_feature_interactions(
        explainer,
        X_for_shap[:100],  # Use small sample for interactions
        feature_names,
        shap_results['top_features']
    )
    
    # Step 7: Create example local explanations
    print("\n📍 Creating example local explanations...")
    
    # Get expected value (base prediction) - use mean of training data
    expected_value = train_df[config.TARGET_COLUMN].mean()
    print(f"📊 Using base value (expected): {expected_value:.2f}")
    
    # Example 1: High life expectancy country
    high_idx = train_df[config.TARGET_COLUMN].idxmax()
    X_high = X_train_scaled[high_idx:high_idx+1]
    create_local_visualization(
        explainer, 
        X_high, 
        feature_names, 
        "High_Life_Expectancy",
        expected_value
    )
    
    # Example 2: Low life expectancy country
    low_idx = train_df[config.TARGET_COLUMN].idxmin()
    X_low = X_train_scaled[low_idx:low_idx+1]
    create_local_visualization(
        explainer, 
        X_low, 
        feature_names, 
        "Low_Life_Expectancy",
        expected_value
    )
    
    # Step 8: Save results
    save_shap_results(
        shap_results['shap_values'],
        shap_results['importance_df'],
        shap_results['top_features'],
        feature_names
    )
    
    print("\n" + "="*60)
    print("✅ MODEL INTERPRETATION COMPLETE!")
    print("="*60)
    
    print("\n📊 Summary:")
    print(f"   - SHAP values computed for {len(feature_names)} features")
    print(f"   - Top 10 features identified and saved")
    print(f"   - Visualizations created in: {config.MODEL_DIR / 'visualizations'}")
    print(f"   - Results saved to: {config.MODEL_DIR}")


if __name__ == "__main__":
    main()

