import pandas as pd
import logging
from typing import Dict, Tuple, List, Set
from datetime import datetime
from db.mongo_connector import get_db
from services.soil_service import get_soil_data
from services.elevation_service import get_elevation
from services.weather_service import fetch_historical_weather

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.db = get_db()
        self.historical_data_collection = self.db.historical_data
        
        # State coordinates mapping (using state capitals/major cities)
        self.state_coordinates = {
            'odisha': (20.2961, 85.8245),
            'andhra pradesh': (15.9129, 79.7400),
            'telangana': (17.3850, 78.4867),
            'west bengal': (22.5726, 88.3639),
            'uttar pradesh': (26.8467, 80.9462),
            'tamil nadu': (11.1271, 78.6569),
            'rajasthan': (27.0238, 74.2179),
            'punjab': (31.1471, 75.3412),
            'maharashtra': (19.7515, 75.7139),
            'madhya pradesh': (23.2599, 77.4126),
            'karnataka': (15.3173, 75.7139),
            'kerala': (10.8505, 76.2711),
            'jharkhand': (23.6102, 85.2799),
            'haryana': (29.0588, 76.0856),
            'gujarat': (23.0225, 72.5714),
            'chhattisgarh': (21.2787, 81.8661),
            'bihar': (25.0961, 85.3131),
            'assam': (26.2006, 92.9376),
        }
        
        # Crop name standardization mapping
        self.crop_mapping = {
            'paddy': 'rice',
            'rice': 'rice',
            'wheat': 'wheat',
            'maize': 'maize',
            'corn': 'maize',
            'cotton': 'cotton',
            'sugarcane': 'sugarcane',
            'groundnut': 'groundnut',
            'peanut': 'groundnut',
            'soybean': 'soybean',
            'turmeric': 'turmeric',
            'onion': 'onion',
            'tomato': 'tomato',
            'chilli': 'chilli',
            'potato': 'potato',
            'coconut': 'coconut',
            'areca': 'areca',
            'cashew': 'cashew',
            'mango': 'mango',
            'banana': 'banana'
        }
        
        # Environmental data cache to minimize API calls
        self.env_data_cache = {}
    
    def load_kaggle_dataset(self, csv_path: str) -> bool:
        """Load and process Kaggle agricultural dataset with optimized API usage"""
        try:
            logger.info(f"Loading dataset from {csv_path}")
            
            # Read CSV file
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"Loaded {len(df)} records from CSV")
            
            # Clean column names
            df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
            logger.info(f"Columns in dataset: {list(df.columns)}")
            
            # Map column names
            column_mapping = {
                'crop': 'crop',
                'crop_year': 'crop_year', 
                'season': 'season',
                'state': 'state',
                'area': 'area',
                'production': 'production',
                'annual_rainfall': 'annual_rainfall',
                'fertilizer': 'fertilizer',
                'pesticide': 'pesticide',
                'yield': 'yield'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Filter for target states
            target_states = ['odisha', 'andhra pradesh', 'telangana']
            if 'state' in df.columns:
                df_filtered = df[df['state'].str.lower().isin(target_states)]
                if len(df_filtered) > 0:
                    df = df_filtered
                    logger.info(f"Filtered to {len(df)} records for target states")
                else:
                    logger.info(f"No records found for target states, using all {len(df)} records")
            
            # Remove rows with critical missing values
            critical_columns = ['crop', 'crop_year', 'state', 'area', 'production']
            df = df.dropna(subset=critical_columns)
            logger.info(f"After removing critical missing values: {len(df)} records")
            
            if len(df) == 0:
                logger.warning("No valid records found after filtering")
                return False
            
            # Optimize environmental data fetching
            logger.info("Optimizing environmental data fetching...")
            self._prefetch_environmental_data(df)
            
            # Process records with optimized approach
            successful_records = 0
            failed_records = 0
            records_batch = []
            batch_size = 100
            
            for index, row in df.iterrows():
                try:
                    record = self._process_record_optimized(row)
                    if record:
                        records_batch.append(record)
                        successful_records += 1
                        
                        # Insert in batches
                        if len(records_batch) >= batch_size:
                            self._insert_batch(records_batch)
                            records_batch = []
                    else:
                        failed_records += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to process record at index {index}: {e}")
                    failed_records += 1
                
                # Progress logging
                if (index + 1) % 250 == 0:
                    logger.info(f"Processed {index + 1} records. Success: {successful_records}, Failed: {failed_records}")
            
            # Insert remaining records
            if records_batch:
                self._insert_batch(records_batch)
            
            logger.info(f"Dataset loading completed. Successfully processed: {successful_records}, Failed: {failed_records}")
            return successful_records > 0
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False
    
    def _prefetch_environmental_data(self, df: pd.DataFrame):
        """Prefetch environmental data for unique state-year combinations"""
        logger.info("Prefetching environmental data to minimize API calls...")
        
        # Get unique state-year combinations
        unique_combinations = df[['state', 'crop_year']].drop_duplicates()
        logger.info(f"Found {len(unique_combinations)} unique state-year combinations")
        
        for index, row in unique_combinations.iterrows():
            try:
                state = str(row['state']).lower().strip()
                year = int(row['crop_year'])
                
                if state not in self.state_coordinates:
                    continue
                
                lat, lon = self.state_coordinates[state]
                cache_key = f"{state}_{year}"
                
                if cache_key not in self.env_data_cache:
                    logger.info(f"Fetching environmental data for {state}, {year}")
                    
                    # Fetch all environmental data for this state-year combination
                    try:
                        soil_data = get_soil_data(lat, lon)
                        elevation = get_elevation(lat, lon)
                        weather_data = fetch_historical_weather(lat, lon, year)
                        
                        self.env_data_cache[cache_key] = {
                            'soil_data': soil_data,
                            'elevation': elevation,
                            'weather_data': weather_data,
                            'coordinates': (lat, lon)
                        }
                        
                        logger.debug(f"Cached environmental data for {cache_key}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to fetch environmental data for {state}, {year}: {e}")
                        # Cache default values
                        self.env_data_cache[cache_key] = self._get_default_env_data(state, lat, lon)
                
                # Progress update for prefetching
                if (index + 1) % 5 == 0:
                    logger.info(f"Prefetched environmental data for {index + 1}/{len(unique_combinations)} combinations")
                    
            except Exception as e:
                logger.warning(f"Error prefetching data for row {index}: {e}")
    
    def _get_default_env_data(self, state: str, lat: float, lon: float) -> Dict:
        """Get default environmental data for a state"""
        return {
            'soil_data': {
                'ph': 6.5, 'nitrogen': 5.0, 'organic_carbon': 1.5,
                'sand_content': 40.0, 'clay_content': 25.0, 'silt_content': 35.0,
                'cec': 15.0, 'bulk_density': 1.4
            },
            'elevation': 100.0,
            'weather_data': {
                'total_rainfall': 1000.0, 'avg_temp_max': 32.0, 'avg_temp_min': 20.0,
                'avg_humidity': 70.0, 'avg_wind_speed': 10.0
            },
            'coordinates': (lat, lon)
        }
    
    def _process_record_optimized(self, row: pd.Series) -> Dict:
        """Process a single record using cached environmental data"""
        try:
            # Extract basic information
            state = str(row['state']).lower().strip()
            crop = str(row['crop']).lower().strip()
            year = int(row['crop_year'])
            
            # Validate required fields
            required_fields = ['crop_year', 'area', 'production']
            for field in required_fields:
                if pd.isna(row.get(field)):
                    logger.debug(f"Missing required field: {field}")
                    return None
            
            # Check if state is supported
            if state not in self.state_coordinates:
                logger.debug(f"State '{state}' not found in coordinates mapping")
                return None
            
            # Standardize crop name
            crop = self.crop_mapping.get(crop, crop)
            
            # Extract and validate numerical data
            area = float(row['area']) if pd.notna(row['area']) and row['area'] > 0 else None
            production = float(row['production']) if pd.notna(row['production']) and row['production'] > 0 else None
            
            # Handle yield calculation
            if pd.notna(row['yield']) and row['yield'] > 0:
                yield_per_hectare = float(row['yield'])
            elif area and production:
                yield_per_hectare = production / area
            else:
                logger.debug("Cannot determine yield for record")
                return None
            
            # Validate data ranges
            if not area or not production or yield_per_hectare <= 0 or yield_per_hectare > 1000:
                return None
            
            # Extract other fields
            fertilizer = float(row['fertilizer']) if pd.notna(row['fertilizer']) and row['fertilizer'] >= 0 else 0
            pesticide = float(row['pesticide']) if pd.notna(row['pesticide']) and row['pesticide'] >= 0 else 0
            annual_rainfall = float(row['annual_rainfall']) if pd.notna(row['annual_rainfall']) and row['annual_rainfall'] >= 0 else 0
            season = str(row.get('season', 'unknown')).lower() if pd.notna(row.get('season')) else 'unknown'
            
            # Get environmental data from cache
            cache_key = f"{state}_{year}"
            if cache_key in self.env_data_cache:
                env_data = self.env_data_cache[cache_key]
                soil_data = env_data['soil_data']
                elevation = env_data['elevation']
                weather_data = env_data['weather_data']
                lat, lon = env_data['coordinates']
            else:
                # Fallback to direct fetch (shouldn't happen with prefetching)
                logger.warning(f"Environmental data not found in cache for {cache_key}")
                lat, lon = self.state_coordinates[state]
                soil_data = get_soil_data(lat, lon)
                elevation = get_elevation(lat, lon)
                weather_data = fetch_historical_weather(lat, lon, year)
            
            # Use provided rainfall if available and reasonable
            if annual_rainfall > 0:
                weather_data = weather_data.copy()
                weather_data['total_rainfall'] = annual_rainfall
            
            # Build the complete record
            record = {
                'source': 'kaggle_dataset',
                'state': state,
                'district': state,  # Using state as district
                'crop_type': crop,
                'year': year,
                'season': season,
                'area_hectares': area,
                'production_quintals': production,
                'yield_per_hectare': yield_per_hectare,
                'fertilizer': fertilizer,
                'pesticide': pesticide,
                'annual_rainfall': annual_rainfall,
                'latitude': lat,
                'longitude': lon,
                'elevation': elevation,
                'created_at': datetime.utcnow(),
                **soil_data,
                **weather_data
            }
            
            # Validate the record
            if self._validate_record(record):
                return record
            else:
                logger.debug("Record failed validation")
                return None
                
        except Exception as e:
            logger.warning(f"Error processing record: {e}")
            return None
    
    def _process_record(self, row: pd.Series) -> Dict:
        """Original process record method for backward compatibility"""
        return self._process_record_optimized(row)
    
    def _validate_record(self, record: Dict) -> bool:
        """Validate a processed record"""
        try:
            # Check for required fields
            required_fields = [
                'state', 'crop_type', 'year', 'area_hectares',
                'production_quintals', 'yield_per_hectare', 'latitude', 'longitude'
            ]
            
            for field in required_fields:
                if field not in record or record[field] is None:
                    logger.debug(f"Missing required field: {field}")
                    return False
            
            # Check for reasonable ranges
            validations = [
                (0 < record['area_hectares'] <= 100000, f"Invalid area: {record['area_hectares']}"),
                (record['production_quintals'] > 0, f"Invalid production: {record['production_quintals']}"),
                (0 < record['yield_per_hectare'] <= 1000, f"Invalid yield: {record['yield_per_hectare']}"),
                (-90 <= record['latitude'] <= 90, f"Invalid latitude: {record['latitude']}"),
                (-180 <= record['longitude'] <= 180, f"Invalid longitude: {record['longitude']}"),
                (1900 <= record['year'] <= 2030, f"Invalid year: {record['year']}")
            ]
            
            for condition, error_msg in validations:
                if not condition:
                    logger.debug(error_msg)
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating record: {e}")
            return False
    
    def _insert_batch(self, records: List[Dict]):
        """Insert a batch of records into the database with upsert"""
        try:
            if records:
                bulk_operations = []
                for record in records:
                    # Create unique identifier for upsert
                    filter_doc = {
                        'state': record['state'],
                        'crop_type': record['crop_type'],
                        'year': record['year'],
                        'area_hectares': record['area_hectares']
                    }
                    
                    bulk_operations.append({
                        'updateOne': {
                            'filter': filter_doc,
                            'update': {'$set': record},
                            'upsert': True
                        }
                    })
                
                # Execute bulk operations
                if bulk_operations:
                    result = self.historical_data_collection.bulk_write(bulk_operations)
                    logger.debug(f"Bulk operation completed: {result.upserted_count} inserted, {result.modified_count} updated")
                
        except Exception as e:
            logger.error(f"Error in bulk insert: {e}")
            # Fallback to individual inserts
            for record in records:
                try:
                    self.historical_data_collection.update_one(
                        {
                            'state': record['state'],
                            'crop_type': record['crop_type'],
                            'year': record['year'],
                            'area_hectares': record['area_hectares']
                        },
                        {'$set': record},
                        upsert=True
                    )
                except Exception as individual_error:
                    logger.warning(f"Failed to insert individual record: {individual_error}")
    
    def load_custom_data(self, data: List[Dict]) -> bool:
        """Load custom data provided as a list of dictionaries"""
        try:
            processed_records = []
            
            for item in data:
                if self._validate_custom_record(item):
                    processed_records.append({
                        **item,
                        'source': 'custom_data',
                        'created_at': datetime.utcnow()
                    })
            
            if processed_records:
                self._insert_batch(processed_records)
                logger.info(f"Loaded {len(processed_records)} custom records")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading custom data: {e}")
            return False
    
    def _validate_custom_record(self, record: Dict) -> bool:
        """Validate custom data record"""
        required_fields = ['state', 'crop_type', 'area_hectares', 'yield_per_hectare']
        return all(field in record and record[field] is not None for field in required_fields)
    
    def get_data_stats(self) -> Dict:
        """Get statistics about loaded data"""
        try:
            total_records = self.historical_data_collection.count_documents({})
            
            if total_records == 0:
                return {
                    'total_records': 0,
                    'by_state': [],
                    'by_crop': [],
                    'year_range': None
                }
            
            # Aggregate statistics
            pipelines = {
                'by_state': [
                    {"$group": {"_id": "$state", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}}
                ],
                'by_crop': [
                    {"$group": {"_id": "$crop_type", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": 10}
                ],
                'year_range': [
                    {"$group": {
                        "_id": None,
                        "min_year": {"$min": "$year"},
                        "max_year": {"$max": "$year"}
                    }}
                ]
            }
            
            results = {}
            for key, pipeline in pipelines.items():
                try:
                    results[key] = list(self.historical_data_collection.aggregate(pipeline))
                except Exception as e:
                    logger.warning(f"Error in {key} aggregation: {e}")
                    results[key] = []
            
            return {
                'total_records': total_records,
                'by_state': results.get('by_state', []),
                'by_crop': results.get('by_crop', []),
                'year_range': results.get('year_range', [None])[0] if results.get('year_range') else None
            }
            
        except Exception as e:
            logger.error(f"Error getting data stats: {e}")
            return {'total_records': 0, 'by_state': [], 'by_crop': [], 'year_range': None}
