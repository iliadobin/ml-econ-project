"""
Streamlit web application for Life Expectancy Forecasting + Policy Generator.
Interactive demo with ML model predictions and LLM-generated policy recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import json
import joblib
from pathlib import Path
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from src.llm_policy_generator import PolicyGenerator
from src.policy_demo import generate_mock_policies, generate_mock_explanation
from src.utils import apply_policy_changes, simulate_uplift


@st.cache_resource
def load_model_and_data():
    """Load model, scaler, data, and feature metadata."""
    try:
        # Load model
        model = joblib.load(config.TRAINED_MODELS_DIR / config.BEST_MODEL_FILENAME)
        
        # Load scaler
        scaler = joblib.load(config.PREPROCESSING_DIR / config.SCALER_FILENAME)
        
        # Load feature names
        with open(config.PREPROCESSING_DIR / config.FEATURE_NAMES_FILENAME, 'r') as f:
            feature_names = json.load(f)
        
        # Load metrics
        with open(config.TRAINED_MODELS_DIR / config.MODEL_METRICS_FILENAME, 'r') as f:
            metrics = json.load(f)
        
        # Load top features from SHAP
        with open(config.MODEL_DIR / config.TOP_FEATURES_FILENAME, 'r') as f:
            top_features_data = json.load(f)
        
        # Load raw data (for country selection)
        raw_data = pd.read_csv(config.RAW_DATA_DIR / config.CSV_FILENAME)
        
        return model, scaler, feature_names, metrics, top_features_data, raw_data
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None, None


def get_country_data(raw_data, country, year):
    """Get feature data for a specific country and year."""
    data = raw_data[(raw_data['Country'] == country) & (raw_data['Year'] == year)]
    
    if len(data) == 0:
        # Try to get the latest available year
        country_data = raw_data[raw_data['Country'] == country]
        if len(country_data) > 0:
            latest_year = country_data['Year'].max()
            data = country_data[country_data['Year'] == latest_year]
            st.warning(f"Data for {year} not available. Using {latest_year} instead.")
    
    return data.iloc[0] if len(data) > 0 else None


def prepare_features_for_prediction(country_data, feature_names):
    """Prepare feature dictionary for prediction."""
    features = {}
    
    for feature in feature_names:
        if feature == 'Status_Developing':
            # Handle encoded status
            features[feature] = 1.0 if country_data.get('Status') == 'Developing' else 0.0
        else:
            # Get feature value from raw data
            features[feature] = country_data.get(feature, 0.0)
    
    return features


def create_comparison_chart(current_le, predicted_le, target_le):
    """Create comparison bar chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Current', 'Predicted', 'Target'],
        y=[current_le, predicted_le, target_le],
        marker_color=['#3498db', '#2ecc71', '#e74c3c'],
        text=[f'{current_le:.1f}', f'{predicted_le:.1f}', f'{target_le:.1f}'],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Life Expectancy Comparison',
        yaxis_title='Years',
        yaxis_range=[0, max(current_le, predicted_le, target_le) * 1.2],
        height=400
    )
    
    return fig


def create_policy_impact_chart(policies):
    """Create chart showing policy impacts."""
    df = pd.DataFrame(policies)
    df['abs_delta'] = df['delta'].abs()
    df = df.sort_values('abs_delta', ascending=True)
    
    fig = go.Figure()
    
    colors = ['#e74c3c' if d < 0 else '#2ecc71' for d in df['delta']]
    
    fig.add_trace(go.Bar(
        y=df['feature'],
        x=df['delta'],
        orientation='h',
        marker_color=colors,
        text=[f'{d:+.2f}' for d in df['delta']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Proposed Policy Changes',
        xaxis_title='Change Amount (Δ)',
        yaxis_title='Feature',
        height=max(300, len(policies) * 60)
    )
    
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Life Expectancy Forecasting + Policy Generator",
        page_icon="🌍",
        layout="wide"
    )
    
    # Header
    st.title("🌍 Life Expectancy Forecasting + Policy Generator")
    st.markdown("### ML in Economics - Interactive Demo")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading model and data..."):
        model, scaler, feature_names, metrics, top_features_data, raw_data = load_model_and_data()
    
    if model is None:
        st.error("⚠️ Could not load model. Please run model training first.")
        st.code("python src/data_processing.py\npython src/model_training.py\npython src/model_interpretation.py")
        return
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model info
    with st.sidebar.expander("📊 Model Performance", expanded=True):
        st.metric("Model", metrics['model_name'])
        col1, col2 = st.columns(2)
        col1.metric("Test MAE", f"{metrics['test_mae']:.2f} years")
        col2.metric("Test R²", f"{metrics['test_r2']:.3f}")
    
    # Country selection
    countries = sorted(raw_data['Country'].unique())
    selected_country = st.sidebar.selectbox(
        "🌍 Select Country",
        countries,
        index=countries.index('United States of America') if 'United States of America' in countries else 0
    )
    
    # Year selection
    available_years = sorted(raw_data['Year'].unique())
    selected_year = st.sidebar.selectbox(
        "📅 Year",
        available_years,
        index=len(available_years) - 1  # Default to latest year
    )
    
    # Policy parameters
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Policy Generation")
    
    target_uplift = st.sidebar.slider(
        "Target Life Expectancy Increase (years)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5
    )
    
    budget_pct_gdp = st.sidebar.slider(
        "Available Budget (% of GDP)",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.5
    )
    
    # LLM mode selection
    use_real_llm = st.sidebar.checkbox(
        "🤖 Use Real LLM (requires API key)",
        value=False,
        help="Enable to use OpenAI/Anthropic API for policy generation"
    )
    
    # Main content
    # Get country data
    country_data = get_country_data(raw_data, selected_country, selected_year)
    
    if country_data is None:
        st.error(f"No data available for {selected_country}")
        return
    
    # Prepare features
    current_features = prepare_features_for_prediction(country_data, feature_names)
    current_life_expectancy = country_data.get('Life expectancy ', 0.0)
    
    # Make prediction
    feature_array = np.array([[current_features[f] for f in feature_names]])
    feature_array_scaled = scaler.transform(feature_array)
    predicted_le = model.predict(feature_array_scaled)[0]
    
    # Section 1: Current Status
    st.header("📊 Current Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Actual Life Expectancy",
            f"{current_life_expectancy:.1f} years"
        )
    
    with col2:
        st.metric(
            "Model Prediction",
            f"{predicted_le:.1f} years",
            delta=f"{predicted_le - current_life_expectancy:+.1f}"
        )
    
    with col3:
        st.metric(
            "Target",
            f"{current_life_expectancy + target_uplift:.1f} years",
            delta=f"+{target_uplift:.1f}"
        )
    
    with col4:
        status = country_data.get('Status', 'Unknown')
        st.metric("Development Status", status)
    
    # Section 2: Key Indicators
    st.markdown("---")
    st.header("🔑 Key Indicators")
    
    top_features = top_features_data['top_features'][:6]
    
    cols = st.columns(3)
    for i, feature in enumerate(top_features):
        with cols[i % 3]:
            value = current_features.get(feature, 0.0)
            st.metric(feature, f"{value:.2f}")
    
    # Section 3: Policy Generation
    st.markdown("---")
    st.header("🤖 Policy Recommendations")
    
    if st.button("🚀 Generate Policy Recommendations", type="primary", use_container_width=True):
        with st.spinner("Generating policies..."):
            try:
                if use_real_llm:
                    # Use real LLM
                    generator = PolicyGenerator()
                    policies_result = generator.generate_policies(
                        country=selected_country,
                        current_features=current_features,
                        target_uplift=target_uplift,
                        budget_pct_gdp=budget_pct_gdp,
                        top_features=top_features,
                        current_life_expectancy=current_life_expectancy
                    )
                else:
                    # Use mock generator
                    st.info("🎭 Using Mock Mode (enable 'Use Real LLM' in sidebar for GPT-4)")
                    policies_result = generate_mock_policies(
                        country=selected_country,
                        current_features=current_features,
                        target_uplift=target_uplift,
                        budget_pct_gdp=budget_pct_gdp,
                        top_features=top_features,
                        current_life_expectancy=current_life_expectancy
                    )
                
                # Store in session state
                st.session_state['policies'] = policies_result
                st.session_state['current_features'] = current_features
                st.session_state['current_le'] = current_life_expectancy
                
            except Exception as e:
                st.error(f"Error generating policies: {e}")
                st.info("💡 Tip: Make sure your API key is set in .env file")
    
    # Display policies if generated
    if 'policies' in st.session_state:
        policies_result = st.session_state['policies']
        policies = policies_result['policies']
        
        st.success(f"✅ Generated {len(policies)} policy recommendations")
        
        # Display policies table
        st.subheader("📋 Proposed Changes")
        
        policies_df = pd.DataFrame([{
            'Feature': p['feature'],
            'Current': f"{p['current']:.2f}",
            'Proposed': f"{p['proposed']:.2f}",
            'Change': f"{p['delta']:+.2f}",
            'Rationale': p.get('rationale', '')
        } for p in policies])
        
        st.dataframe(policies_df, use_container_width=True, hide_index=True)
        
        # Policy impact chart
        st.plotly_chart(create_policy_impact_chart(policies), use_container_width=True)
        
        # Implementation details
        with st.expander("📅 Implementation Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Estimated Cost:**", f"{policies_result.get('estimated_cost_pct_gdp', 0):.1f}% of GDP")
                st.write("**Timeline:**", policies_result.get('implementation_timeline', 'N/A'))
            with col2:
                st.write("**Key Challenges:**")
                st.write(policies_result.get('key_challenges', 'N/A'))
        
        # Section 4: Simulation
        st.markdown("---")
        st.header("🔮 Counterfactual Simulation")
        
        if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Simulating policy impact..."):
                # Apply policy changes
                modified_features = apply_policy_changes(current_features, policies)
                
                # Make new prediction
                modified_array = np.array([[modified_features[f] for f in feature_names]])
                modified_scaled = scaler.transform(modified_array)
                new_prediction = model.predict(modified_scaled)[0]
                
                achieved_uplift = new_prediction - current_life_expectancy
                
                # Store in session state
                st.session_state['simulation_result'] = {
                    'new_prediction': new_prediction,
                    'achieved_uplift': achieved_uplift
                }
        
        # Display simulation results
        if 'simulation_result' in st.session_state:
            sim_result = st.session_state['simulation_result']
            new_prediction = sim_result['new_prediction']
            achieved_uplift = sim_result['achieved_uplift']
            
            st.success("✅ Simulation Complete")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Current Life Expectancy",
                    f"{current_life_expectancy:.1f} years"
                )
            
            with col2:
                st.metric(
                    "Predicted After Policies",
                    f"{new_prediction:.1f} years",
                    delta=f"+{achieved_uplift:.1f}"
                )
            
            with col3:
                target_le = current_life_expectancy + target_uplift
                achievement_pct = (achieved_uplift / target_uplift) * 100
                st.metric(
                    "Target Achievement",
                    f"{achievement_pct:.0f}%",
                    delta=f"{achieved_uplift:.1f} / {target_uplift:.1f}"
                )
            
            # Comparison chart
            st.plotly_chart(
                create_comparison_chart(
                    current_life_expectancy,
                    new_prediction,
                    current_life_expectancy + target_uplift
                ),
                use_container_width=True
            )
            
            # Explanation
            st.subheader("💬 Policy Explanation")
            
            if st.button("Generate Explanation"):
                with st.spinner("Generating explanation..."):
                    try:
                        if use_real_llm and 'policies' in st.session_state:
                            generator = PolicyGenerator()
                            explanation = generator.explain_policies(
                                country=selected_country,
                                policies=policies,
                                current_life_expectancy=current_life_expectancy,
                                predicted_new_life_expectancy=new_prediction,
                                target_uplift=target_uplift
                            )
                        else:
                            explanation = generate_mock_explanation(
                                country=selected_country,
                                policies=policies,
                                current_life_expectancy=current_life_expectancy,
                                predicted_new_life_expectancy=new_prediction,
                                target_uplift=target_uplift
                            )
                        
                        st.info(explanation)
                        
                    except Exception as e:
                        st.error(f"Error generating explanation: {e}")
    
    # Section 5: Feature Importance
    st.markdown("---")
    st.header("📈 Feature Importance (SHAP)")
    
    with st.expander("View SHAP Analysis", expanded=False):
        viz_dir = config.MODEL_DIR / "visualizations"
        
        if viz_dir.exists():
            # Show SHAP importance bar
            bar_plot = viz_dir / "shap_importance_bar.png"
            if bar_plot.exists():
                st.subheader("Global Feature Importance")
                st.image(str(bar_plot), use_container_width=True)
            
            # Show SHAP summary
            summary_plot = viz_dir / "shap_summary_beeswarm.png"
            if summary_plot.exists():
                st.subheader("Feature Impact Distribution")
                st.image(str(summary_plot), use_container_width=True)
        else:
            st.warning("SHAP visualizations not found. Run model_interpretation.py first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>🎓 ML in Economics - Life Expectancy Forecasting Project</p>
    <p>Powered by XGBoost ML Model + GPT-4 Policy Generator</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

