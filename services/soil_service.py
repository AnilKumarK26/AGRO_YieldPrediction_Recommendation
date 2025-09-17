# import requests
# import logging
# import time
# from typing import Dict, Optional
# from config import SOIL_API_URL, DEFAULT_VALUES
# from db.mongo_connector import get_db

# logger = logging.getLogger(__name__)

# class SoilService:
#     def __init__(self):
#         self.db = get_db()
#         self.soil_collection = self.db.soil_cache
#         self.properties_mapping = {
#             'phh2o': 'ph',
#             'nitrogen': 'nitrogen', 
#             'soc': 'organic_carbon',
#             'sand': 'sand_content',
#             'clay': 'clay_content',
#             'silt': 'silt_content',
#             'cec': 'cec',
#             'bdod': 'bulk_density'
#         }
        
#         # Conversion factors for API values
#         self.conversion_factors = {
#             'phh2o': 10,
#             'nitrogen': 100,
#             'soc': 10,
#             'sand': 10,
#             'clay': 10,
#             'silt': 10,
#             'cec': 10,
#             'bdod': 100
#         }
        
#         # State-wise default soil values for Indian states
#         self.state_soil_defaults = {
#             'odisha': {
#                 'ph': 6.2, 'nitrogen': 4.5, 'organic_carbon': 1.3,
#                 'sand_content': 45.0, 'clay_content': 28.0, 'silt_content': 27.0,
#                 'cec': 16.0, 'bulk_density': 1.35
#             },
#             'andhra pradesh': {
#                 'ph': 7.1, 'nitrogen': 3.8, 'organic_carbon': 1.1,
#                 'sand_content': 52.0, 'clay_content': 22.0, 'silt_content': 26.0,
#                 'cec': 14.0, 'bulk_density': 1.42
#             },
#             'telangana': {
#                 'ph': 6.8, 'nitrogen': 4.2, 'organic_carbon': 1.2,
#                 'sand_content': 48.0, 'clay_content': 25.0, 'silt_content': 27.0,
#                 'cec': 15.0, 'bulk_density': 1.38
#             }
#         }
        
#         # API retry configuration
#         self.max_retries = 2
#         self.timeout = 15  # Reduced timeout
#         self.rate_limit_delay = 1.0  # Increased delay between requests
    
#     def get_soil_data(self, lat: float, lon: float, state: str = None) -> Dict[str, float]:
#         """Get soil data for given coordinates with caching and fallbacks"""
#         try:
#             # Round coordinates for caching
#             lat_rounded = round(lat, 2)  # Reduced precision for better caching
#             lon_rounded = round(lon, 2)
#             cache_key = f"soil_{lat_rounded}_{lon_rounded}"
            
#             # Check cache first
#             cached = self.soil_collection.find_one({'_id': cache_key})
#             if cached:
#                 logger.debug(f"Retrieved soil data from cache for {lat_rounded}, {lon_rounded}")
#                 return {k: v for k, v in cached.items() if k != '_id'}
            
#             # Try to fetch from API with timeout and retries
#             logger.info(f"Fetching soil data from API for {lat}, {lon}")
#             try:
#                 soil_data = self._fetch_from_api_with_retry(lat, lon)
#                 if soil_data:
#                     processed_data = self._process_soil_data(soil_data)
#                     self._cache_soil_data(cache_key, processed_data)
#                     return processed_data
#             except Exception as e:
#                 logger.warning(f"API fetch failed for {lat}, {lon}: {e}")
            
#             # Fallback to state-based defaults if available
#             if state and state.lower() in self.state_soil_defaults:
#                 logger.info(f"Using state-based soil defaults for {state}")
#                 return self.state_soil_defaults[state.lower()].copy()
            
#             # Final fallback to global defaults
#             logger.info("Using global default soil values")
#             return self._get_default_soil_data()
            
#         except Exception as e:
#             logger.error(f"Error getting soil data for {lat}, {lon}: {e}")
#             return self._get_default_soil_data()
    
#     def _fetch_from_api_with_retry(self, lat: float, lon: float) -> Dict[str, float]:
#         """Fetch soil data with retry logic and timeout handling"""
#         soil_data = {}
        
#         for prop_api, prop_name in self.properties_mapping.items():
#             success = False
            
#             for attempt in range(self.max_retries):
#                 try:
#                     params = {
#                         'lon': lon,
#                         'lat': lat,
#                         'property': prop_api,
#                         'depth': '0-30cm',
#                         'value': 'mean'
#                     }
                    
#                     response = requests.get(
#                         SOIL_API_URL, 
#                         params=params, 
#                         timeout=self.timeout,
#                         headers={'User-Agent': 'CropPredict/1.0'}
#                     )
                    
#                     if response.status_code == 200:
#                         data = response.json()
#                         mean_value = self._extract_soil_value(data, prop_api)
                        
#                         if mean_value is not None:
#                             soil_data[prop_api] = mean_value
#                             logger.debug(f"Fetched {prop_api}: {mean_value}")
#                             success = True
#                             break
#                     else:
#                         logger.warning(f"API returned status {response.status_code} for {prop_api}")
                        
#                 except requests.exceptions.Timeout:
#                     logger.warning(f"Timeout for {prop_api} (attempt {attempt + 1}/{self.max_retries})")
#                     if attempt < self.max_retries - 1:
#                         time.sleep(2 ** attempt)  # Exponential backoff
                        
#                 except requests.exceptions.RequestException as e:
#                     logger.warning(f"Request error for {prop_api}: {e}")
#                     break
            
#             if not success:
#                 logger.warning(f"Failed to fetch {prop_api} after {self.max_retries} attempts")
            
#             # Rate limiting
#             time.sleep(self.rate_limit_delay)
        
#         return soil_data
    
#     def _extract_soil_value(self, data: Dict, prop_api: str) -> Optional[float]:
#         """Extract soil value from API response"""
#         try:
#             properties = data.get('properties', {})
#             if prop_api in properties:
#                 layers = properties[prop_api].get('layers', [])
#                 if layers and len(layers) > 0:
#                     depths = layers[0].get('depths', [])
#                     if depths and len(depths) > 0:
#                         values = depths[0].get('values', {})
#                         mean_value = values.get('mean')
#                         if mean_value is not None:
#                             return float(mean_value)
#         except Exception as e:
#             logger.warning(f"Error extracting value for {prop_api}: {e}")
        
#         return None
    
#     def _fetch_from_api(self, lat: float, lon: float) -> Dict[str, float]:
#         """Original API fetch method - kept for compatibility"""
#         return self._fetch_from_api_with_retry(lat, lon)
    
#     def _process_soil_data(self, raw_data: Dict[str, float]) -> Dict[str, float]:
#         """Process and normalize soil data from API"""
#         processed = {}
        
#         for prop_api, prop_name in self.properties_mapping.items():
#             if prop_api in raw_data:
#                 # Apply conversion factor
#                 conversion_factor = self.conversion_factors.get(prop_api, 1)
#                 processed[prop_name] = raw_data[prop_api] / conversion_factor
#             else:
#                 # Use default value
#                 processed[prop_name] = DEFAULT_VALUES['soil'].get(prop_name, 0)
        
#         # Ensure realistic ranges
#         processed = self._validate_soil_values(processed)
        
#         return processed
    
#     def _validate_soil_values(self, soil_data: Dict[str, float]) -> Dict[str, float]:
#         """Validate and constrain soil values to realistic ranges"""
#         constraints = {
#             'ph': (3.5, 9.0),
#             'nitrogen': (0.1, 10.0),
#             'organic_carbon': (0.1, 5.0),
#             'sand_content': (0, 100),
#             'clay_content': (0, 100),
#             'silt_content': (0, 100),
#             'cec': (1, 50),
#             'bulk_density': (0.8, 2.0)
#         }
        
#         validated = {}
#         for prop, value in soil_data.items():
#             if prop in constraints:
#                 min_val, max_val = constraints[prop]
#                 validated[prop] = max(min_val, min(max_val, value))
#             else:
#                 validated[prop] = value
        
#         return validated
    
#     def _cache_soil_data(self, cache_key: str, processed_data: Dict[str, float]):
#         """Cache soil data with error handling"""
#         try:
#             cache_record = processed_data.copy()
#             cache_record['_id'] = cache_key
#             cache_record['fetch_timestamp'] = time.time()
            
#             self.soil_collection.update_one(
#                 {'_id': cache_key},
#                 {'$set': cache_record},
#                 upsert=True
#             )
#             logger.debug(f"Cached soil data for {cache_key}")
#         except Exception as e:
#             logger.warning(f"Failed to cache soil data: {e}")
    
#     def _get_default_soil_data(self) -> Dict[str, float]:
#         """Return default soil values when API fails"""
#         return DEFAULT_VALUES['soil'].copy()
    
#     def bulk_prefetch_soil_data(self, coordinates: list[tuple], states: list[str] = None):
#         """Prefetch soil data for multiple coordinates to improve performance"""
#         logger.info(f"Prefetching soil data for {len(coordinates)} locations")
        
#         for i, (lat, lon) in enumerate(coordinates):
#             try:
#                 state = states[i] if states and i < len(states) else None
#                 self.get_soil_data(lat, lon, state)
                
#                 if (i + 1) % 10 == 0:
#                     logger.info(f"Prefetched soil data for {i + 1}/{len(coordinates)} locations")
                    
#             except Exception as e:
#                 logger.warning(f"Error prefetching soil data for {lat}, {lon}: {e}")

# # Global service instance
# soil_service = SoilService()

# def get_soil_data(lat: float, lon: float, state: str = None) -> Dict[str, float]:
#     """Public function to get soil data with state fallback"""
#     return soil_service.get_soil_data(lat, lon, state)

import requests
import logging
import time
from typing import Dict, Optional
from config import DEFAULT_VALUES
from db.mongo_connector import get_db

logger = logging.getLogger(__name__)

class SoilService:
    def __init__(self):
        self.db = get_db()
        self.soil_collection = self.db.soil_cache
        
        # FIXED: Correct SoilGrids API endpoint
        self.SOIL_API_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        
        # Property mapping for SoilGrids API
        self.properties_mapping = {
            'phh2o': 'ph',
            'nitrogen': 'nitrogen', 
            'soc': 'organic_carbon',
            'sand': 'sand_content',
            'clay': 'clay_content',
            'silt': 'silt_content',
            'cec': 'cec',
            'bdod': 'bulk_density'
        }
        
        # Conversion factors for API values (SoilGrids uses specific units)
        self.conversion_factors = {
            'phh2o': 10,      # pH * 10
            'nitrogen': 100,   # g/kg * 100 
            'soc': 10,        # dg/kg * 10
            'sand': 10,       # g/kg * 10
            'clay': 10,       # g/kg * 10
            'silt': 10,       # g/kg * 10
            'cec': 10,        # cmol(+)/kg * 10
            'bdod': 100       # cg/cm³ * 100
        }
        
        # State-wise default soil values for Indian states
        self.state_soil_defaults = {
            'odisha': {
                'ph': 6.2, 'nitrogen': 4.5, 'organic_carbon': 1.3,
                'sand_content': 45.0, 'clay_content': 28.0, 'silt_content': 27.0,
                'cec': 16.0, 'bulk_density': 1.35
            },
            'andhra pradesh': {
                'ph': 7.1, 'nitrogen': 3.8, 'organic_carbon': 1.1,
                'sand_content': 52.0, 'clay_content': 22.0, 'silt_content': 26.0,
                'cec': 14.0, 'bulk_density': 1.42
            },
            'telangana': {
                'ph': 6.8, 'nitrogen': 4.2, 'organic_carbon': 1.2,
                'sand_content': 48.0, 'clay_content': 25.0, 'silt_content': 27.0,
                'cec': 15.0, 'bulk_density': 1.38
            },
            'punjab': {
                'ph': 7.5, 'nitrogen': 5.2, 'organic_carbon': 1.4,
                'sand_content': 35.0, 'clay_content': 32.0, 'silt_content': 33.0,
                'cec': 18.0, 'bulk_density': 1.30
            },
            'haryana': {
                'ph': 7.8, 'nitrogen': 4.8, 'organic_carbon': 1.2,
                'sand_content': 40.0, 'clay_content': 30.0, 'silt_content': 30.0,
                'cec': 17.0, 'bulk_density': 1.35
            },
            'uttar pradesh': {
                'ph': 7.2, 'nitrogen': 4.5, 'organic_carbon': 1.3,
                'sand_content': 42.0, 'clay_content': 28.0, 'silt_content': 30.0,
                'cec': 16.5, 'bulk_density': 1.38
            }
        }
        
        # API retry configuration
        self.max_retries = 2
        self.timeout = 20
        self.rate_limit_delay = 12.0  # SoilGrids has 5 calls per minute limit
    
    def get_soil_data(self, lat: float, lon: float, state: str = None) -> Dict[str, float]:
        """Get soil data for given coordinates with caching and fallbacks"""
        try:
            # Round coordinates for caching
            lat_rounded = round(lat, 2)
            lon_rounded = round(lon, 2)
            cache_key = f"soil_{lat_rounded}_{lon_rounded}"
            
            # Check cache first
            cached = self.soil_collection.find_one({'_id': cache_key})
            if cached:
                logger.info(f"Retrieved soil data from cache for {lat_rounded}, {lon_rounded}")
                return {k: v for k, v in cached.items() if k != '_id' and k != 'fetch_timestamp'}
            
            # Try to fetch from API with the new endpoint
            logger.info(f"Fetching soil data from SoilGrids API for {lat}, {lon}")
            try:
                soil_data = self._fetch_from_soilgrids_api(lat, lon)
                if soil_data and len(soil_data) > 0:
                    processed_data = self._process_soil_data(soil_data)
                    self._cache_soil_data(cache_key, processed_data)
                    logger.info(f"Successfully fetched soil data from API: pH={processed_data.get('ph', 'N/A')}")
                    return processed_data
            except Exception as e:
                logger.warning(f"SoilGrids API fetch failed for {lat}, {lon}: {e}")
            
            # Fallback to state-based defaults if available
            if state and state.lower() in self.state_soil_defaults:
                logger.info(f"Using state-based soil defaults for {state}")
                return self.state_soil_defaults[state.lower()].copy()
            
            # Final fallback to global defaults
            logger.info("Using global default soil values")
            return self._get_default_soil_data()
            
        except Exception as e:
            logger.error(f"Error getting soil data for {lat}, {lon}: {e}")
            return self._get_default_soil_data()
    
    def _fetch_from_soilgrids_api(self, lat: float, lon: float) -> Dict[str, float]:
        """
        Fetch soil data from SoilGrids API using the correct endpoint
        NEW METHOD: Uses the properties/query endpoint with all properties at once
        """
        try:
            # SoilGrids properties query - fetch all properties in one request
            properties = list(self.properties_mapping.keys())
            
            params = {
                'lon': lon,
                'lat': lat,
                'property': properties,  # Multiple properties
                'depth': '0-30cm',       # Surface layer
                'value': 'mean'          # Mean value
            }
            
            headers = {
                'User-Agent': 'CropPredict/1.0 (Agricultural Application)',
                'Accept': 'application/json'
            }
            
            logger.info(f"Making SoilGrids API request for {len(properties)} properties")
            
            response = requests.get(
                self.SOIL_API_URL,
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info("SoilGrids API request successful")
                return self._extract_all_soil_values(data)
            else:
                logger.warning(f"SoilGrids API returned status {response.status_code}: {response.text[:200]}")
                return {}
                
        except requests.exceptions.Timeout:
            logger.warning("SoilGrids API request timed out")
            return {}
        except requests.exceptions.RequestException as e:
            logger.warning(f"SoilGrids API request failed: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error fetching from SoilGrids API: {e}")
            return {}
    
    def _extract_all_soil_values(self, data: Dict) -> Dict[str, float]:
        """Extract all soil values from SoilGrids API response"""
        soil_values = {}
        
        try:
            properties = data.get('properties', {})
            
            for prop_api, prop_name in self.properties_mapping.items():
                if prop_api in properties:
                    layers = properties[prop_api].get('layers', [])
                    if layers and len(layers) > 0:
                        depths = layers[0].get('depths', [])
                        if depths and len(depths) > 0:
                            values = depths[0].get('values', {})
                            mean_value = values.get('mean')
                            if mean_value is not None:
                                soil_values[prop_api] = float(mean_value)
                                logger.debug(f"Extracted {prop_api}: {mean_value}")
            
            logger.info(f"Successfully extracted {len(soil_values)} soil properties")
            return soil_values
            
        except Exception as e:
            logger.error(f"Error extracting soil values from API response: {e}")
            return {}
    
    def _process_soil_data(self, raw_data: Dict[str, float]) -> Dict[str, float]:
        """Process and normalize soil data from API"""
        processed = {}
        
        for prop_api, prop_name in self.properties_mapping.items():
            if prop_api in raw_data:
                # Apply conversion factor
                conversion_factor = self.conversion_factors.get(prop_api, 1)
                processed[prop_name] = raw_data[prop_api] / conversion_factor
            else:
                # Use default value
                processed[prop_name] = DEFAULT_VALUES['soil'].get(prop_name, 0)
        
        # Ensure realistic ranges
        processed = self._validate_soil_values(processed)
        
        return processed
    
    def _validate_soil_values(self, soil_data: Dict[str, float]) -> Dict[str, float]:
        """Validate and constrain soil values to realistic ranges"""
        constraints = {
            'ph': (3.5, 9.0),
            'nitrogen': (0.1, 10.0),
            'organic_carbon': (0.1, 5.0),
            'sand_content': (0, 100),
            'clay_content': (0, 100),
            'silt_content': (0, 100),
            'cec': (1, 50),
            'bulk_density': (0.8, 2.0)
        }
        
        validated = {}
        for prop, value in soil_data.items():
            if prop in constraints:
                min_val, max_val = constraints[prop]
                validated[prop] = max(min_val, min(max_val, value))
            else:
                validated[prop] = value
        
        # Ensure sand + clay + silt = 100% (approximately)
        if all(k in validated for k in ['sand_content', 'clay_content', 'silt_content']):
            total = validated['sand_content'] + validated['clay_content'] + validated['silt_content']
            if total > 0 and abs(total - 100) > 20:  # If total is way off
                # Normalize to 100%
                factor = 100 / total
                validated['sand_content'] *= factor
                validated['clay_content'] *= factor
                validated['silt_content'] *= factor
        
        return validated
    
    def _cache_soil_data(self, cache_key: str, processed_data: Dict[str, float]):
        """Cache soil data with error handling"""
        try:
            cache_record = processed_data.copy()
            cache_record['_id'] = cache_key
            cache_record['fetch_timestamp'] = time.time()
            
            self.soil_collection.update_one(
                {'_id': cache_key},
                {'$set': cache_record},
                upsert=True
            )
            logger.debug(f"Cached soil data for {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache soil data: {e}")
    
    def _get_default_soil_data(self) -> Dict[str, float]:
        """Return default soil values when API fails"""
        return DEFAULT_VALUES['soil'].copy()
    
    def test_api_connection(self, lat: float = 15.81, lon: float = 78.02) -> bool:
        """Test if SoilGrids API is accessible"""
        try:
            response = requests.get(
                self.SOIL_API_URL,
                params={
                    'lon': lon,
                    'lat': lat,
                    'property': ['phh2o'],
                    'depth': '0-30cm',
                    'value': 'mean'
                },
                timeout=10
            )
            
            success = response.status_code == 200
            logger.info(f"SoilGrids API connection test: {'SUCCESS' if success else 'FAILED'}")
            if not success:
                logger.warning(f"API response: {response.status_code} - {response.text[:200]}")
            
            return success
            
        except Exception as e:
            logger.error(f"SoilGrids API connection test failed: {e}")
            return False
    
    def get_cached_locations(self) -> list:
        """Get list of cached soil data locations"""
        try:
            cached = list(self.soil_collection.find({}, {'_id': 1}))
            return [doc['_id'] for doc in cached]
        except Exception as e:
            logger.error(f"Error retrieving cached locations: {e}")
            return []
    
    def clear_cache(self) -> bool:
        """Clear all cached soil data"""
        try:
            result = self.soil_collection.delete_many({})
            logger.info(f"Cleared {result.deleted_count} cached soil records")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False


# Global service instance
soil_service = SoilService()

def get_soil_data(lat: float, lon: float, state: str = None) -> Dict[str, float]:
    """Public function to get soil data with state fallback"""
    return soil_service.get_soil_data(lat, lon, state)

def test_soil_api():
    """Test function to check if soil API is working"""
    return soil_service.test_api_connection()

# # Example usage and testing
# if __name__ == "__main__":
#     # Test the API connection
#     print("Testing SoilGrids API connection...")
#     if test_soil_api():
#         print("✅ API is working!")
        
#         # Test with Visakhapatnam coordinates
#         test_lat, test_lon = 17.6868, 83.2185
#         print(f"\nFetching soil data for Visakhapatnam ({test_lat}, {test_lon})...")
        
#         soil_data = get_soil_data(test_lat, test_lon, "andhra pradesh")
#         print("Soil Data Retrieved:")
#         for key, value in soil_data.items():
#             print(f"  {key}: {value}")
#     else:
#         print("❌ API is not accessible - will use fallback values")
        
#         # Test fallback
#         soil_data = get_soil_data(17.6868, 83.2185, "andhra pradesh")
#         print("Fallback Soil Data:")
#         for key, value in soil_data.items():
#             print(f"  {key}: {value}")