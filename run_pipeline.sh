#!/bin/bash
# Complete pipeline execution script for Life Expectancy ML Project

echo "=================================="
echo "🚀 LIFE EXPECTANCY ML PIPELINE"
echo "=================================="
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Step 1: Data Processing
echo "📊 Step 1/3: Data Processing..."
python src/data_processing.py
if [ $? -ne 0 ]; then
    echo "❌ Data processing failed!"
    exit 1
fi
echo "✅ Data processing complete"
echo ""

# Step 2: Model Training
echo "🤖 Step 2/3: Model Training..."
python src/model_training.py
if [ $? -ne 0 ]; then
    echo "❌ Model training failed!"
    exit 1
fi
echo "✅ Model training complete"
echo ""

# Step 3: Model Interpretation
echo "🔍 Step 3/3: Model Interpretation (SHAP)..."
python src/model_interpretation.py
if [ $? -ne 0 ]; then
    echo "❌ Model interpretation failed!"
    exit 1
fi
echo "✅ Model interpretation complete"
echo ""

echo "=================================="
echo "✅ PIPELINE COMPLETE!"
echo "=================================="
echo ""
echo "🌐 To launch the web application:"
echo "   streamlit run app/app.py"
echo ""
echo "📊 Results:"
echo "   - Model: model/trained_models/best_model.pkl"
echo "   - SHAP plots: model/visualizations/"
echo "   - Processed data: data/processed/"
echo ""

