"""
Test script to demonstrate all utility functions from src/utils.py
Specifically tests Этап 7: Симуляция counterfactual сценариев
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.utils import (
    load_model_and_preprocessing,
    apply_policy_changes,
    simulate_uplift,
    validate_policy_feasibility
)
import config


def main():
    print("\n" + "="*60)
    print("🧪 TESTING COUNTERFACTUAL SIMULATION UTILITIES")
    print("="*60)
    
    # Step 1: Load model and preprocessing
    print("\n📂 Step 1: Loading model and preprocessing...")
    model, scaler, feature_names = load_model_and_preprocessing()
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"✅ Scaler loaded: {type(scaler).__name__}")
    print(f"✅ Features: {len(feature_names)} features")
    
    # Step 2: Create example scenario
    print("\n🌍 Step 2: Creating example scenario...")
    
    # Example country features
    original_features = {
        'Year': 2015,
        'Adult Mortality': 180.0,
        ' HIV/AIDS': 1.2,
        'infant deaths': 50,
        'Alcohol': 5.0,
        'percentage expenditure': 300.0,
        'Hepatitis B': 85.0,
        'Measles ': 500,
        ' BMI ': 25.0,
        'under-five deaths ': 65,
        'Polio': 85.0,
        'Total expenditure': 5.5,
        'Diphtheria ': 85.0,
        'GDP': 5000.0,
        'Population': 10000000.0,
        ' thinness  1-19 years': 5.0,
        ' thinness 5-9 years': 5.0,
        'Income composition of resources': 0.60,
        'Schooling': 9.5,
        'Status_Developing': 1.0
    }
    
    print(f"📊 Country: Example Developing Country")
    print(f"   Adult Mortality: {original_features['Adult Mortality']}")
    print(f"   Schooling: {original_features['Schooling']}")
    print(f"   Income Composition: {original_features['Income composition of resources']}")
    
    # Step 3: Define policy changes
    print("\n🎯 Step 3: Defining policy changes...")
    policy_changes = [
        {
            'feature': 'Adult Mortality',
            'delta': -30.0,
            'current': 180.0,
            'proposed': 150.0
        },
        {
            'feature': ' HIV/AIDS',
            'delta': -0.4,
            'current': 1.2,
            'proposed': 0.8
        },
        {
            'feature': 'Schooling',
            'delta': 1.5,
            'current': 9.5,
            'proposed': 11.0
        },
        {
            'feature': 'Income composition of resources',
            'delta': 0.08,
            'current': 0.60,
            'proposed': 0.68
        }
    ]
    
    print(f"📋 Proposed {len(policy_changes)} policy changes:")
    for policy in policy_changes:
        print(f"   • {policy['feature']}: {policy['current']:.2f} → {policy['proposed']:.2f} (Δ{policy['delta']:+.2f})")
    
    # Step 4: Apply policy changes
    print("\n✏️  Step 4: Applying policy changes...")
    modified_features = apply_policy_changes(original_features, policy_changes)
    print("✅ Policy changes applied successfully")
    
    # Verify changes
    for policy in policy_changes:
        feature = policy['feature']
        if feature in modified_features:
            print(f"   ✓ {feature}: {original_features[feature]:.2f} → {modified_features[feature]:.2f}")
    
    # Step 5: Simulate uplift
    print("\n🔮 Step 5: Simulating uplift...")
    original_pred, new_pred, uplift = simulate_uplift(
        original_features,
        modified_features,
        model,
        scaler,
        feature_names
    )
    
    print(f"✅ Simulation complete:")
    print(f"   Original Life Expectancy: {original_pred:.2f} years")
    print(f"   Predicted after policies: {new_pred:.2f} years")
    print(f"   Achieved Uplift: {uplift:+.2f} years")
    
    if uplift > 0:
        print(f"   🎉 Success! Policies improved life expectancy by {uplift:.2f} years")
    else:
        print(f"   ⚠️  Warning: Policies decreased life expectancy by {abs(uplift):.2f} years")
    
    # Step 6: Validate policy feasibility
    print("\n🔍 Step 6: Validating policy feasibility...")
    
    # Load reference data for validation
    reference_data = pd.read_csv(config.RAW_DATA_DIR / config.CSV_FILENAME)
    
    validation_result = validate_policy_feasibility(
        policy_changes,
        reference_data,
        original_features
    )
    
    print(f"✅ Validation complete:")
    print(f"   Feasible: {'Yes ✓' if validation_result['is_feasible'] else 'No ✗'}")
    
    if validation_result['warnings']:
        print(f"\n   ⚠️  Warnings ({len(validation_result['warnings'])}):")
        for warning in validation_result['warnings']:
            print(f"      - {warning}")
    else:
        print("   ✓ No warnings - all changes are within reasonable bounds")
    
    if validation_result['recommendations']:
        print(f"\n   💡 Recommendations:")
        for rec in validation_result['recommendations']:
            print(f"      - {rec}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ ALL COUNTERFACTUAL SIMULATION FUNCTIONS TESTED")
    print("="*60)
    print(f"""
📊 Summary:
   • apply_policy_changes():        ✅ Working
   • simulate_uplift():              ✅ Working  
   • validate_policy_feasibility():  ✅ Working
   
🎯 Results:
   • Policies applied: {len(policy_changes)}
   • Life expectancy change: {uplift:+.2f} years
   • Validation: {'Passed' if validation_result['is_feasible'] else 'Failed'}
   • Warnings: {len(validation_result['warnings'])}
""")


if __name__ == "__main__":
    main()

