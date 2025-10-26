"""
Configuration file for the Life Expectancy ML Project.
Centralizes all paths, constants, and settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "model"
TRAINED_MODELS_DIR = MODEL_DIR / "trained_models"
PREPROCESSING_DIR = MODEL_DIR / "preprocessing"

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAINED_MODELS_DIR, PREPROCESSING_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data settings
DATASET_NAME = "kumarajarshi/life-expectancy-who"
CSV_FILENAME = "Life Expectancy Data.csv"
TARGET_COLUMN = "Life expectancy "  # Note: original dataset has trailing space
RANDOM_STATE = 42

# Train/Val/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model settings
MODEL_METRICS = ["mae", "rmse", "r2"]
CV_FOLDS = 5

# Feature engineering
CATEGORICAL_FEATURES = ["Country", "Status"]
NUMERIC_FEATURES = [
    "Year", "Adult Mortality", "infant deaths", "Alcohol",
    "percentage expenditure", "Hepatitis B", "Measles", " BMI ",
    "under-five deaths ", "Polio", "Total expenditure", "Diphtheria ",
    " HIV/AIDS", "GDP", "Population", " thinness  1-19 years",
    " thinness 5-9 years", "Income composition of resources", "Schooling"
]

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
# Updated model names (gpt-4-turbo-preview is deprecated)
# Options: gpt-4o, gpt-4-turbo, gpt-4, gpt-3.5-turbo
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")  # Most capable model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# LLM generation parameters
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2000

# Policy generation constraints
MIN_UPLIFT = 1.0  # years
MAX_UPLIFT = 10.0  # years
MIN_BUDGET = 0.5  # % of GDP
MAX_BUDGET = 10.0  # % of GDP

# File names for saved models and data
BEST_MODEL_FILENAME = "best_model.pkl"
SCALER_FILENAME = "scaler.pkl"
FEATURE_NAMES_FILENAME = "feature_names.json"
MODEL_METRICS_FILENAME = "model_metrics.json"
SHAP_VALUES_FILENAME = "shap_values.pkl"
TOP_FEATURES_FILENAME = "top_features.json"

# Processed data files
TRAIN_DATA_FILENAME = "train_data.csv"
VAL_DATA_FILENAME = "val_data.csv"
TEST_DATA_FILENAME = "test_data.csv"

