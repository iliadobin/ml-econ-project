"""
Demo script for Policy Generator (mock mode without API key).
Shows how the system would work with LLM integration.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def generate_mock_policies(
    country: str,
    current_features: Dict[str, float],
    target_uplift: float,
    budget_pct_gdp: float,
    top_features: List[str],
    current_life_expectancy: float
) -> Dict[str, Any]:
    """
    Generate mock policy recommendations (for demonstration without API key).
    
    Args:
        country: Country name
        current_features: Current feature values
        target_uplift: Target increase in life expectancy (years)
        budget_pct_gdp: Available budget as % of GDP
        top_features: List of most important features (from SHAP)
        current_life_expectancy: Current life expectancy value
        
    Returns:
        Dictionary with mock policy recommendations
    """
    print(f"\n🎭 MOCK MODE: Generating policies for {country}...")
    print(f"   Target uplift: +{target_uplift} years")
    print(f"   Budget: {budget_pct_gdp}% of GDP")
    
    # Create realistic mock policies based on top features
    policies = []
    
    # Adult Mortality (if in top features and value is high)
    if "Adult Mortality" in current_features and current_features["Adult Mortality"] > 100:
        current_val = current_features["Adult Mortality"]
        reduction = min(30, current_val * 0.20)  # 20% reduction
        policies.append({
            "feature": "Adult Mortality",
            "current": current_val,
            "proposed": current_val - reduction,
            "delta": -reduction,
            "rationale": "Reduce adult mortality through improved healthcare access, preventive care, and emergency services"
        })
    
    # HIV/AIDS (if in top features)
    if " HIV/AIDS" in current_features and current_features[" HIV/AIDS"] > 0.3:
        current_val = current_features[" HIV/AIDS"]
        reduction = min(0.3, current_val * 0.40)  # 40% reduction
        policies.append({
            "feature": " HIV/AIDS",
            "current": current_val,
            "proposed": current_val - reduction,
            "delta": -reduction,
            "rationale": "Expand HIV/AIDS prevention programs, improve access to antiretroviral therapy"
        })
    
    # Income Composition
    if "Income composition of resources" in current_features:
        current_val = current_features["Income composition of resources"]
        if current_val < 0.75:
            increase = min(0.10, (0.75 - current_val) * 0.5)
            policies.append({
                "feature": "Income composition of resources",
                "current": current_val,
                "proposed": current_val + increase,
                "delta": increase,
                "rationale": "Improve income distribution and economic opportunities through job creation and social programs"
            })
    
    # Schooling
    if "Schooling" in current_features:
        current_val = current_features["Schooling"]
        if current_val < 13:
            increase = min(1.5, 13 - current_val)
            policies.append({
                "feature": "Schooling",
                "current": current_val,
                "proposed": current_val + increase,
                "delta": increase,
                "rationale": "Increase average years of schooling through education reforms and accessibility programs"
            })
    
    # Under-five deaths
    if "under-five deaths " in current_features and current_features["under-five deaths "] > 20:
        current_val = current_features["under-five deaths "]
        reduction = min(current_val * 0.30, 50)
        policies.append({
            "feature": "under-five deaths ",
            "current": current_val,
            "proposed": current_val - reduction,
            "delta": -reduction,
            "rationale": "Reduce child mortality through vaccination programs and maternal health initiatives"
        })
    
    return {
        "policies": policies[:5],  # Return max 5 policies
        "estimated_cost_pct_gdp": budget_pct_gdp * 0.8,  # Use 80% of budget
        "implementation_timeline": "5-10 years",
        "key_challenges": "Political will, infrastructure development, behavior change, funding sustainability"
    }


def generate_mock_explanation(
    country: str,
    policies: List[Dict[str, Any]],
    current_life_expectancy: float,
    predicted_new_life_expectancy: float,
    target_uplift: float
) -> str:
    """
    Generate mock natural language explanation.
    
    Args:
        country: Country name
        policies: List of policy changes
        current_life_expectancy: Current life expectancy
        predicted_new_life_expectancy: Predicted life expectancy after policies
        target_uplift: Target uplift requested
        
    Returns:
        Natural language explanation
    """
    achieved_uplift = predicted_new_life_expectancy - current_life_expectancy
    
    explanation = f"""These policy recommendations for {country} represent a comprehensive, evidence-based approach to increasing life expectancy from {current_life_expectancy:.1f} to {predicted_new_life_expectancy:.1f} years (an increase of {achieved_uplift:.1f} years). """
    
    if len(policies) >= 2:
        explanation += f"The strategy focuses on {len(policies)} key interventions: "
        policy_names = [p['feature'] for p in policies[:3]]
        explanation += ", ".join(policy_names)
        if len(policies) > 3:
            explanation += ", and others"
        explanation += ". "
    
    explanation += """These interventions work synergistically - improvements in healthcare reduce mortality, while education and income gains create better health-seeking behaviors and living conditions. The proposed changes are based on achievements observed in countries that have successfully improved life expectancy.

The implementation will require sustained political commitment, adequate funding, and strong institutional capacity. Key challenges include building infrastructure, training healthcare workers, and ensuring equitable access across all population segments. """
    
    if achieved_uplift >= target_uplift * 0.9:
        explanation += f"The model predicts that these changes are likely to achieve the target increase of {target_uplift:.1f} years."
    else:
        explanation += f"While these changes will substantially improve life expectancy, achieving the full target of {target_uplift:.1f} years may require additional long-term investments beyond the current scope."
    
    return explanation


def main():
    """Demonstrate policy generation in mock mode."""
    print("\n" + "="*60)
    print("🎭 POLICY GENERATOR - MOCK DEMONSTRATION")
    print("="*60)
    print("\nThis demonstrates how the LLM policy generator would work.")
    print("For real LLM-generated policies, add your API key to .env file.")
    
    # Example data
    country = "Example Country"
    current_features = {
        "Adult Mortality": 180.0,
        " HIV/AIDS": 1.2,
        "Income composition of resources": 0.60,
        "Schooling": 9.5,
        "under-five deaths ": 45.0,
        " BMI ": 22.0,
        "Year": 2015
    }
    
    top_features = [
        "Adult Mortality",
        " HIV/AIDS",
        "Income composition of resources",
        "under-five deaths ",
        "Schooling"
    ]
    
    current_le = 65.0
    target_uplift = 5.0
    budget = 3.5
    
    # Generate policies
    policies_result = generate_mock_policies(
        country,
        current_features,
        target_uplift,
        budget,
        top_features,
        current_le
    )
    
    print(f"\n✅ Generated {len(policies_result['policies'])} mock policies:")
    print(json.dumps(policies_result, indent=2))
    
    # Simulate prediction after policies
    predicted_le = current_le + 4.2  # Mock uplift
    
    # Generate explanation
    explanation = generate_mock_explanation(
        country,
        policies_result['policies'],
        current_le,
        predicted_le,
        target_uplift
    )
    
    print("\n💬 Mock Explanation:")
    print(explanation)
    
    print("\n" + "="*60)
    print("✅ MOCK DEMONSTRATION COMPLETE!")
    print("="*60)
    print("\n💡 To use real LLM:")
    print("   1. Add your OpenAI API key to .env file")
    print("   2. Run: python src/llm_policy_generator.py")


if __name__ == "__main__":
    main()

