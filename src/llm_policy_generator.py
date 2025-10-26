"""
LLM-based policy generator for Life Expectancy ML Project.
Generates counterfactual policy recommendations using GPT-4 or Claude.
"""

import os
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Import LLM clients (with fallback if not installed)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI library not installed. Install with: pip install openai")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️  Anthropic library not installed. Install with: pip install anthropic")


class PolicyGenerator:
    """
    Generate counterfactual policies using LLM to increase life expectancy.
    """
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize LLM client based on configuration.
        
        Args:
            provider: Optional LLM provider override ('openai' or 'anthropic')
        """
        self.provider = provider or config.LLM_PROVIDER
        self.model = config.LLM_MODEL
        self.client = None
        
        # Initialize client
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise RuntimeError("OpenAI library not installed")
            if not config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in environment")
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
            print(f"✅ Initialized OpenAI client with model: {self.model}")
            
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise RuntimeError("Anthropic library not installed")
            if not config.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not set in environment")
            self.client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
            print(f"✅ Initialized Anthropic client with model: {self.model}")
            
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def generate_policies(
        self,
        country: str,
        current_features: Dict[str, float],
        target_uplift: float,
        budget_pct_gdp: float,
        top_features: List[str],
        current_life_expectancy: float
    ) -> Dict[str, Any]:
        """
        Generate policy recommendations to achieve target uplift.
        
        Args:
            country: Country name
            current_features: Current feature values
            target_uplift: Target increase in life expectancy (years)
            budget_pct_gdp: Available budget as % of GDP
            top_features: List of most important features (from SHAP)
            current_life_expectancy: Current life expectancy value
            
        Returns:
            Dictionary with policy recommendations
        """
        print(f"\n🤖 Generating policies for {country}...")
        print(f"   Target uplift: +{target_uplift} years")
        print(f"   Budget: {budget_pct_gdp}% of GDP")
        
        # Create prompt
        prompt = self._create_policy_prompt(
            country,
            current_features,
            target_uplift,
            budget_pct_gdp,
            top_features,
            current_life_expectancy
        )
        
        # Call LLM
        response_text = self._call_llm(prompt)
        
        # Parse response
        policies = self._parse_llm_response(response_text)
        
        print(f"✅ Generated {len(policies.get('policies', []))} policy recommendations")
        
        return policies
    
    def explain_policies(
        self,
        country: str,
        policies: List[Dict[str, Any]],
        current_life_expectancy: float,
        predicted_new_life_expectancy: float,
        target_uplift: float
    ) -> str:
        """
        Generate natural language explanation of the policies.
        
        Args:
            country: Country name
            policies: List of policy changes
            current_life_expectancy: Current life expectancy
            predicted_new_life_expectancy: Predicted life expectancy after policies
            target_uplift: Target uplift requested
            
        Returns:
            Natural language explanation
        """
        print(f"\n📝 Generating explanation for {country} policies...")
        
        achieved_uplift = predicted_new_life_expectancy - current_life_expectancy
        
        prompt = f"""You are an expert policy advisor explaining health and social policy recommendations.

Country: {country}
Current Life Expectancy: {current_life_expectancy:.1f} years
Predicted New Life Expectancy: {predicted_new_life_expectancy:.1f} years
Achieved Uplift: {achieved_uplift:.1f} years
Target Uplift: {target_uplift:.1f} years

Proposed Policy Changes:
"""
        
        for i, policy in enumerate(policies, 1):
            feature = policy['feature']
            current = policy['current']
            proposed = policy['proposed']
            delta = policy['delta']
            prompt += f"\n{i}. {feature}:"
            prompt += f"\n   Current: {current:.2f}"
            prompt += f"\n   Proposed: {proposed:.2f}"
            prompt += f"\n   Change: {delta:+.2f}"
        
        prompt += """

Please provide a clear, concise explanation (2-3 paragraphs) that:
1. Explains the rationale behind these policy recommendations
2. Describes how these changes work together to improve life expectancy
3. Discusses the feasibility and potential challenges
4. Comments on whether the target uplift is likely to be achieved

Write in an accessible style for policymakers and the general public."""
        
        explanation = self._call_llm(prompt)
        
        print("✅ Generated explanation")
        
        return explanation
    
    def _create_policy_prompt(
        self,
        country: str,
        current_features: Dict[str, float],
        target_uplift: float,
        budget_pct_gdp: float,
        top_features: List[str],
        current_life_expectancy: float
    ) -> str:
        """
        Create prompt for LLM policy generation.
        
        Args:
            country: Country name
            current_features: Current feature values
            target_uplift: Target uplift in years
            budget_pct_gdp: Budget as % of GDP
            top_features: Most important features from SHAP
            current_life_expectancy: Current life expectancy
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert health policy advisor with deep knowledge of public health, economics, and social policy. Your task is to recommend specific, actionable policy changes to increase life expectancy.

**Context:**
Country: {country}
Current Life Expectancy: {current_life_expectancy:.1f} years
Target Increase: +{target_uplift} years
Available Budget: {budget_pct_gdp}% of GDP

**Current Indicators (features that influence life expectancy):**
"""
        
        # Add current feature values (focus on top features)
        for feature in top_features[:10]:
            if feature in current_features:
                value = current_features[feature]
                prompt += f"\n- {feature}: {value:.2f}"
        
        prompt += f"""

**Most Important Features (by SHAP importance):**
These features have the strongest impact on life expectancy predictions:
"""
        for i, feature in enumerate(top_features[:5], 1):
            prompt += f"\n{i}. {feature}"
        
        prompt += f"""

**Your Task:**
Generate 3-5 specific policy recommendations that would increase life expectancy by approximately {target_uplift} years. For each recommendation, specify:
1. Which feature/indicator to change
2. The current value
3. The proposed new value (realistic and achievable)
4. The change amount (delta)

**Guidelines:**
- Focus on the most impactful features (Adult Mortality, HIV/AIDS, Income Composition, Schooling, etc.)
- Propose realistic changes based on what successful countries have achieved
- Consider budget constraints ({budget_pct_gdp}% of GDP)
- Changes should be evidence-based and feasible within 5-10 years
- Avoid extreme changes that would be impossible to implement

**Output Format (JSON):**
Return ONLY a valid JSON object (no other text) in this exact format:

{{
  "policies": [
    {{
      "feature": "Adult Mortality",
      "current": 150.0,
      "proposed": 120.0,
      "delta": -30.0,
      "rationale": "Reduce adult mortality through improved healthcare access"
    }},
    {{
      "feature": "Schooling",
      "current": 10.5,
      "proposed": 12.0,
      "delta": 1.5,
      "rationale": "Increase average years of schooling through education reforms"
    }}
  ],
  "estimated_cost_pct_gdp": {budget_pct_gdp},
  "implementation_timeline": "5-10 years",
  "key_challenges": "Political will, infrastructure development, behavior change"
}}

Generate the recommendations now:"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM API with the given prompt.
        Includes fallback logic for unavailable models.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response text
        """
        if self.provider == "openai":
            # List of models to try in order (from best to most accessible)
            models_to_try = [
                self.model,  # User's preferred model
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo"
            ]
            
            # Remove duplicates while preserving order
            models_to_try = list(dict.fromkeys(models_to_try))
            
            for model in models_to_try:
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are an expert health policy advisor. Always respond with valid JSON when requested."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=config.LLM_TEMPERATURE,
                        max_tokens=config.LLM_MAX_TOKENS
                    )
                    
                    if model != self.model:
                        print(f"ℹ️  Using fallback model: {model}")
                    
                    return response.choices[0].message.content
                    
                except Exception as e:
                    error_msg = str(e)
                    if "model_not_found" in error_msg or "does not exist" in error_msg:
                        print(f"⚠️  Model {model} not available, trying next...")
                        continue
                    elif "insufficient_quota" in error_msg or "429" in error_msg:
                        print(f"⚠️  Quota exceeded for model {model}")
                        if model != models_to_try[-1]:
                            print(f"   Trying next model...")
                            continue
                        else:
                            print("\n" + "="*60)
                            print("❌ OpenAI API quota exceeded")
                            print("="*60)
                            print("\n💡 Solutions:")
                            print("   1. Use Mock Mode (recommended for demos)")
                            print("      → Uncheck 'Use Real LLM' in Streamlit sidebar")
                            print("   2. Add billing details to your OpenAI account")
                            print("   3. Use a different API key")
                            print("\n📖 For Mock Mode demo, run:")
                            print("   python src/policy_demo.py")
                            print("="*60)
                            raise
                    else:
                        # Other error, re-raise
                        print(f"❌ Error with model {model}: {e}")
                        raise
            
            # If all models failed
            raise RuntimeError("All OpenAI models failed. Check your API key and account access.")
                
        elif self.provider == "anthropic":
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=config.LLM_MAX_TOKENS,
                    temperature=config.LLM_TEMPERATURE,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            except Exception as e:
                print(f"❌ Error calling Anthropic: {e}")
                raise
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured format.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Parsed dictionary with policy recommendations
        """
        # Try to extract JSON from response
        # Sometimes LLM adds extra text before/after JSON
        
        # Look for JSON block
        json_match = re.search(r'\{[\s\S]*\}', response)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                policies = json.loads(json_str)
                
                # Validate structure
                if 'policies' not in policies:
                    raise ValueError("Response missing 'policies' key")
                
                # Ensure all required fields are present
                for policy in policies['policies']:
                    required = ['feature', 'current', 'proposed', 'delta']
                    for field in required:
                        if field not in policy:
                            raise ValueError(f"Policy missing required field: {field}")
                
                return policies
                
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
                print(f"Response: {response[:500]}...")
                raise
        else:
            print(f"❌ No JSON found in response")
            print(f"Response: {response[:500]}...")
            raise ValueError("Could not extract JSON from LLM response")


def test_policy_generator():
    """Test the policy generator with mock data."""
    print("\n" + "="*60)
    print("🧪 TESTING POLICY GENERATOR")
    print("="*60)
    
    # Check if API key is available
    if not config.OPENAI_API_KEY and not config.ANTHROPIC_API_KEY:
        print("\n⚠️  No API keys found in environment.")
        print("To test the LLM policy generator:")
        print("1. Copy .env.example to .env")
        print("2. Add your OpenAI or Anthropic API key")
        print("3. Run this script again")
        return
    
    try:
        # Initialize generator
        generator = PolicyGenerator()
        
        # Mock data
        country = "Example Country"
        current_features = {
            "Adult Mortality": 150.0,
            "HIV/AIDS": 0.5,
            "Income composition of resources": 0.65,
            "Schooling": 10.5,
            "BMI": 25.0,
            "GDP": 5000.0,
            "percentage expenditure": 300.0,
            "Total expenditure": 5.5
        }
        
        top_features = [
            "Adult Mortality",
            "HIV/AIDS",
            "Income composition of resources",
            "Schooling",
            "BMI"
        ]
        
        # Generate policies
        policies = generator.generate_policies(
            country=country,
            current_features=current_features,
            target_uplift=5.0,
            budget_pct_gdp=3.0,
            top_features=top_features,
            current_life_expectancy=68.0
        )
        
        print("\n📋 Generated Policies:")
        print(json.dumps(policies, indent=2))
        
        # Generate explanation
        explanation = generator.explain_policies(
            country=country,
            policies=policies['policies'],
            current_life_expectancy=68.0,
            predicted_new_life_expectancy=72.5,
            target_uplift=5.0
        )
        
        print("\n💬 Explanation:")
        print(explanation)
        
        print("\n" + "="*60)
        print("✅ POLICY GENERATOR TEST COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    test_policy_generator()


if __name__ == "__main__":
    main()

