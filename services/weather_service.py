import requests
import numpy as np
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from config import WEATHER_API_ARCHIVE_URL, WEATHER_API_FORECAST_URL, DEFAULT_VALUES
from db.mongo_connector import get_db

logger = logging.getLogger(__name__)

class WeatherService:
    def __init__(self):
        self.db = get_db()
        self.weather_collection = self.db.weather_cache
    
    def fetch_historical_weather(self, lat: float, lon: float, year: int) -> Dict[str, float]:
        """Fetch historical weather data for a specific year"""
        try:
            # Check cache first
            cache_key = f"weather_{round(lat, 2)}_{round(lon, 2)}_{year}"
            cached = self.weather_collection.find_one({'_id': cache_key})
            
            if cached:
                logger.info(f"Retrieved weather data from cache for {year}")
                return {k: v for k, v in cached.items() if k not in ['_id', 'fetch_timestamp']}
            
            # Fetch from API
            logger.info(f"Fetching weather data from API for {lat}, {lon}, year {year}")
            weather_data = self._fetch_from_api(lat, lon, year)
            
            # Cache the result
            try:
                cache_record = weather_data.copy()
                cache_record['_id'] = cache_key
                cache_record['fetch_timestamp'] = datetime.now()
                self.weather_collection.insert_one(cache_record)
                logger.info(f"Cached weather data for {year}")
            except Exception as e:
                logger.warning(f"Failed to cache weather data: {e}")
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Error getting weather data: {e}")
            return self._get_default_weather_data()
    
    def _fetch_from_api(self, lat: float, lon: float, year: int) -> Dict[str, float]:
        """Fetch weather data from Open-Meteo API"""
        try:
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            params = {
                'latitude': lat,
                'longitude': lon,
                'start_date': start_date,
                'end_date': end_date,
                'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean,windspeed_10m_max',
                'timezone': 'Asia/Kolkata'
            }
            
            response = requests.get(
                WEATHER_API_ARCHIVE_URL,
                params=params,
                timeout=60,
                headers={'User-Agent': 'CropPredict/1.0'}
            )
            
            if response.status_code == 200:
                data = response.json()
                daily = data.get('daily', {})
                
                # Process the data
                weather_summary = self._process_weather_data(daily)
                
                # Validate the processed data
                weather_summary = self._validate_weather_data(weather_summary)
                
                logger.info(f"Successfully processed weather data for {year}")
                return weather_summary
            else:
                logger.warning(f"Weather API returned status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API request failed: {e}")
        except Exception as e:
            logger.error(f"Error processing weather data: {e}")
        
        return self._get_default_weather_data()
    
    def _process_weather_data(self, daily_data: Dict) -> Dict[str, float]:
        """Process daily weather data into summary statistics"""
        try:
            # Extract arrays with fallback values
            temp_max = daily_data.get('temperature_2m_max', [])
            temp_min = daily_data.get('temperature_2m_min', [])
            precipitation = daily_data.get('precipitation_sum', [])
            humidity = daily_data.get('relative_humidity_2m_mean', [])
            windspeed = daily_data.get('windspeed_10m_max', [])
            
            # Filter out None values and calculate statistics
            def safe_mean(values, default=0):
                filtered = [v for v in values if v is not None]
                return np.mean(filtered) if filtered else default
            
            def safe_sum(values, default=0):
                filtered = [v for v in values if v is not None]
                return np.sum(filtered) if filtered else default
            
            weather_summary = {
                'total_rainfall': safe_sum(precipitation, DEFAULT_VALUES['weather']['total_rainfall']),
                'avg_temp_max': safe_mean(temp_max, DEFAULT_VALUES['weather']['avg_temp_max']),
                'avg_temp_min': safe_mean(temp_min, DEFAULT_VALUES['weather']['avg_temp_min']),
                'avg_humidity': safe_mean(humidity, DEFAULT_VALUES['weather']['avg_humidity']),
                'avg_wind_speed': safe_mean(windspeed, DEFAULT_VALUES['weather']['avg_wind_speed'])
            }
            
            return weather_summary
            
        except Exception as e:
            logger.error(f"Error processing weather data arrays: {e}")
            return self._get_default_weather_data()
    
    def _validate_weather_data(self, weather_data: Dict[str, float]) -> Dict[str, float]:
        """Validate weather data ranges"""
        constraints = {
            'total_rainfall': (0, 5000),      # 0-5000mm annual rainfall
            'avg_temp_max': (-10, 50),        # -10°C to 50°C
            'avg_temp_min': (-20, 40),        # -20°C to 40°C
            'avg_humidity': (10, 100),        # 10-100%
            'avg_wind_speed': (0, 50)         # 0-50 km/h
        }
        
        validated = {}
        for param, value in weather_data.items():
            if param in constraints:
                min_val, max_val = constraints[param]
                if min_val <= value <= max_val:
                    validated[param] = value
                else:
                    logger.warning(f"Weather parameter {param} value {value} outside valid range [{min_val}, {max_val}]")
                    validated[param] = DEFAULT_VALUES['weather'].get(param, value)
            else:
                validated[param] = value
        
        return validated
    
    def _get_default_weather_data(self) -> Dict[str, float]:
        """Return default weather values when API fails"""
        logger.info("Using default weather values")
        return DEFAULT_VALUES['weather'].copy()
    
    def get_current_weather_forecast(self, lat: float, lon: float, days: int = 7) -> Dict:
        """Get current weather forecast (optional feature)"""
        try:
            params = {
                'latitude': lat,
                'longitude': lon,
                'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum',
                'forecast_days': days
            }
            
            response = requests.get(
                WEATHER_API_FORECAST_URL,
                params=params,
                timeout=30,
                headers={'User-Agent': 'CropPredict/1.0'}
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            logger.error(f"Error fetching weather forecast: {e}")
        
        return {}

# Global service instance
weather_service = WeatherService()

def fetch_historical_weather(lat: float, lon: float, year: int) -> Dict[str, float]:
    """Public function to fetch historical weather data"""
    return weather_service.fetch_historical_weather(lat, lon, year)