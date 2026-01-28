import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# API URLs
SOIL_API_URL = "https://rest.isric.org/soilgrids/v2.0/properties"
ELEVATION_API_URL = "https://api.open-elevation.com/api/v1/lookup"
WEATHER_API_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/era5"
WEATHER_API_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Groq AI Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# OpenAI Configuration (Alternative)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# AGMARKNET Configuration (Real Market Data)
AGMARKNET_API_KEY = os.getenv("AGMARKNET_API_KEY", "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b")

# Default values when APIs fail
DEFAULT_VALUES = {
    'soil': {
        'ph': 6.5,
        'nitrogen': 5.0,
        'organic_carbon': 1.5,
        'sand_content': 40.0,
        'clay_content': 25.0,
        'silt_content': 35.0,
        'cec': 15.0,
        'bulk_density': 1.4
    },
    'weather': {
        'total_rainfall': 1000.0,
        'avg_temp_max': 32.0,
        'avg_temp_min': 20.0,
        'avg_humidity': 70.0,
        'avg_wind_speed': 10.0
    },
    'elevation': 100.0
}

# ============================================================================
# REMOVED: CROP_PRICES dictionary
# ============================================================================
# Crop prices are now fetched from real market data sources via:
# services/real_market_data_service.py
#
# This ensures farmers get accurate, up-to-date prices from:
# - AGMARKNET (Government of India Agricultural Marketing API)
# - State APMC (Agricultural Produce Market Committee) data
# - Real-time market trends
#
# To get crop prices in your code, use:
#   from services.real_market_data_service import get_real_market_price
#   price_data = get_real_market_price(crop_type, state, district)
#
# Example response:
#   {
#       'current': 2200,
#       'min': 1900,
#       'max': 2600,
#       'average': 2250,
#       'trend': 'increasing',
#       'source': 'AGMARKNET (Government of India)',
#       'last_updated': '2026-01-28T10:30:00'
#   }
# ============================================================================

# State-wise agricultural zones (for future enhancements)
AGRICULTURAL_ZONES = {
    'odisha': {
        'major_crops': ['rice', 'groundnut', 'turmeric'],
        'climate': 'tropical',
        'avg_annual_rainfall': 1482  # mm
    },
    'andhra pradesh': {
        'major_crops': ['rice', 'cotton', 'groundnut'],
        'climate': 'tropical',
        'avg_annual_rainfall': 940
    },
    'telangana': {
        'major_crops': ['rice', 'cotton', 'maize'],
        'climate': 'tropical',
        'avg_annual_rainfall': 906
    },
    'punjab': {
        'major_crops': ['wheat', 'rice', 'maize'],
        'climate': 'subtropical',
        'avg_annual_rainfall': 649
    },
    'haryana': {
        'major_crops': ['wheat', 'rice', 'sugarcane'],
        'climate': 'subtropical',
        'avg_annual_rainfall': 617
    },
    'maharashtra': {
        'major_crops': ['sugarcane', 'cotton', 'soybean'],
        'climate': 'tropical',
        'avg_annual_rainfall': 1200
    }
}

# Crop-specific optimal conditions (for XAI service)
CROP_OPTIMAL_CONDITIONS = {
    'rice': {
        'temperature_range': (20, 35),  # Celsius
        'rainfall_range': (1000, 2000),  # mm
        'ph_range': (5.5, 7.0),
        'soil_type': 'clayey loam',
        'growing_season': 'kharif'
    },
    'wheat': {
        'temperature_range': (12, 25),
        'rainfall_range': (450, 650),
        'ph_range': (6.0, 7.5),
        'soil_type': 'loamy',
        'growing_season': 'rabi'
    },
    'cotton': {
        'temperature_range': (21, 35),
        'rainfall_range': (500, 1000),
        'ph_range': (6.0, 7.5),
        'soil_type': 'black soil',
        'growing_season': 'kharif'
    },
    'maize': {
        'temperature_range': (21, 30),
        'rainfall_range': (500, 800),
        'ph_range': (5.8, 7.0),
        'soil_type': 'well-drained loam',
        'growing_season': 'kharif/rabi'
    },
    'sugarcane': {
        'temperature_range': (20, 35),
        'rainfall_range': (1500, 2500),
        'ph_range': (6.0, 7.5),
        'soil_type': 'loamy',
        'growing_season': 'year-round'
    },
    'soybean': {
        'temperature_range': (20, 30),
        'rainfall_range': (450, 700),
        'ph_range': (6.0, 7.0),
        'soil_type': 'well-drained loam',
        'growing_season': 'kharif'
    },
    'groundnut': {
        'temperature_range': (22, 30),
        'rainfall_range': (500, 1000),
        'ph_range': (6.0, 7.0),
        'soil_type': 'sandy loam',
        'growing_season': 'kharif/summer'
    },
    'turmeric': {
        'temperature_range': (20, 30),
        'rainfall_range': (1500, 2250),
        'ph_range': (5.5, 7.5),
        'soil_type': 'well-drained loam',
        'growing_season': 'kharif'
    }
}

# API Rate Limiting Configuration
API_RATE_LIMITS = {
    'soilgrids': {
        'calls_per_minute': 5,
        'retry_after': 12  # seconds
    },
    'open_meteo': {
        'calls_per_minute': 100,
        'retry_after': 1
    },
    'agmarknet': {
        'calls_per_day': 1000,
        'cache_duration': 86400  # 24 hours in seconds
    }
}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Model Configuration
MODEL_PATH = "models/crop_yield_model.joblib"
MODEL_VERSION = "1.0"

# Feature Engineering Configuration
FEATURE_COLUMNS = [
    # Soil features
    'ph', 'nitrogen', 'organic_carbon', 'sand_content', 'clay_content', 
    'silt_content', 'cec', 'bulk_density',
    # Weather features
    'total_rainfall', 'avg_temp_max', 'avg_temp_min', 'avg_humidity', 'avg_wind_speed',
    # Location features
    'elevation', 'latitude', 'longitude',
    # Input features
    'area_hectares', 'fertilizer', 'pesticide'
]

# XAI Configuration
XAI_CONFIG = {
    'min_feature_contributions': 5,  # Show top 5 contributing features
    'confidence_thresholds': {
        'high': 80.0,
        'moderate': 60.0,
        'low': 0.0
    },
    'show_optimization_opportunities': True,
    'show_limiting_factors': True
}

# Market Data Configuration
MARKET_DATA_CONFIG = {
    'cache_duration_hours': 24,
    'fallback_to_historical': True,
    'require_data_attribution': True,
    'min_data_freshness_hours': 48
}

# Frontend Configuration
FRONTEND_CONFIG = {
    'supported_languages': ['en', 'hi'],
    'default_language': 'en',
    'show_confidence_scores': True,
    'show_data_sources': True,
    'enable_geolocation': True
}

# System Constants
MINIMUM_TRAINING_SAMPLES = 10
PREDICTION_BATCH_SIZE = 100
DATABASE_TIMEOUT_SECONDS = 30

# Feature Validation Ranges
FEATURE_VALIDATION_RANGES = {
    'ph': (3.5, 9.0),
    'nitrogen': (0.1, 10.0),
    'organic_carbon': (0.1, 5.0),
    'sand_content': (0, 100),
    'clay_content': (0, 100),
    'silt_content': (0, 100),
    'cec': (1, 50),
    'bulk_density': (0.8, 2.0),
    'total_rainfall': (0, 5000),
    'avg_temp_max': (-10, 50),
    'avg_temp_min': (-20, 40),
    'avg_humidity': (10, 100),
    'avg_wind_speed': (0, 50),
    'elevation': (-500, 9000),
    'latitude': (-90, 90),
    'longitude': (-180, 180),
    'area_hectares': (0.01, 100000),
    'fertilizer': (0, 1000),
    'pesticide': (0, 100)
}

# Error Messages
ERROR_MESSAGES = {
    'model_not_ready': 'ML model is not ready. Please wait for system initialization.',
    'no_data': 'No training data found. Please load dataset first.',
    'invalid_input': 'Invalid input parameters provided.',
    'api_failure': 'External API call failed. Using fallback values.',
    'database_error': 'Database operation failed.',
    'prediction_error': 'Error during yield prediction.'
}

# Success Messages
SUCCESS_MESSAGES = {
    'prediction_complete': 'Yield prediction completed successfully.',
    'model_trained': 'ML model trained successfully.',
    'data_loaded': 'Dataset loaded successfully.',
    'market_data_fetched': 'Real market data retrieved successfully.'
}