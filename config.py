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

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

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

# Crop price estimates (INR per quintal) for value calculation
CROP_PRICES = {
    'rice': 2000,
    'wheat': 2200,
    'maize': 1800,
    'cotton': 5500,
    'sugarcane': 300,
    'groundnut': 5000,
    'soybean': 4000,
    'turmeric': 8000,
    'onion': 1500,
    'tomato': 2500,
    'chilli': 12000,
    'potato': 1200,
    'default': 2000
}