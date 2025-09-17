import requests
import logging
from typing import Optional
from config import ELEVATION_API_URL, DEFAULT_VALUES
from db.mongo_connector import get_db

logger = logging.getLogger(__name__)

class ElevationService:
    def __init__(self):
        self.db = get_db()
        self.elevation_collection = self.db.elevation_cache
    
    def get_elevation(self, lat: float, lon: float) -> float:
        """Get elevation data for given coordinates with caching"""
        try:
            # Round coordinates for caching
            lat_rounded = round(lat, 3)
            lon_rounded = round(lon, 3)
            cache_key = f"elevation_{lat_rounded}_{lon_rounded}"
            
            # Check cache first
            cached = self.elevation_collection.find_one({'_id': cache_key})
            if cached:
                logger.info(f"Retrieved elevation from cache for {lat_rounded}, {lon_rounded}")
                return cached['elevation']
            
            # Fetch from API
            logger.info(f"Fetching elevation from API for {lat}, {lon}")
            elevation = self._fetch_from_api(lat, lon)
            
            # Cache the result
            try:
                cache_record = {
                    '_id': cache_key,
                    'elevation': elevation,
                    'latitude': lat_rounded,
                    'longitude': lon_rounded
                }
                self.elevation_collection.insert_one(cache_record)
                logger.info(f"Cached elevation data for {lat_rounded}, {lon_rounded}")
            except Exception as e:
                logger.warning(f"Failed to cache elevation data: {e}")
            
            return elevation
            
        except Exception as e:
            logger.error(f"Error getting elevation for {lat}, {lon}: {e}")
            return DEFAULT_VALUES['elevation']
    
    def _fetch_from_api(self, lat: float, lon: float) -> float:
        """Fetch elevation from Open-Elevation API"""
        try:
            params = {
                'locations': f"{lat},{lon}"
            }
            
            response = requests.get(
                ELEVATION_API_URL,
                params=params,
                timeout=30,
                headers={'User-Agent': 'CropPredict/1.0'}
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                if results and len(results) > 0:
                    elevation = results[0].get('elevation')
                    if elevation is not None:
                        # Ensure reasonable elevation values
                        if -500 <= elevation <= 9000:  # Valid elevation range
                            logger.debug(f"Fetched elevation: {elevation}m")
                            return float(elevation)
                        else:
                            logger.warning(f"Elevation value {elevation} seems unrealistic")
                            
            logger.warning(f"No valid elevation data received from API")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
        except (ValueError, KeyError) as e:
            logger.error(f"Error parsing elevation response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching elevation: {e}")
        
        # Return default value if API fails
        return DEFAULT_VALUES['elevation']

# Global service instance
elevation_service = ElevationService()

def get_elevation(lat: float, lon: float) -> float:
    """Public function to get elevation data"""
    return elevation_service.get_elevation(lat, lon)