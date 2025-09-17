from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from config import MONGO_URI
import logging

logger = logging.getLogger(__name__)

class MongoConnector:
    _instance = None
    _client = None
    _db = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoConnector, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            try:
                self._client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
                # Test connection
                self._client.admin.command('ismaster')
                self._db = self._client['crop_yield_db']
                logger.info("MongoDB connected successfully")
                
                # Create indexes for better performance
                self._setup_indexes()
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.error(f"MongoDB connection failed: {e}")
                raise
    
    def _setup_indexes(self):
        try:
            # Index for soil cache
            self._db.soil_cache.create_index("_id")
            
            # Index for elevation cache
            self._db.elevation_cache.create_index("_id")
            
            # Index for historical data
            self._db.historical_data.create_index([
                ("state", 1), ("district", 1), ("crop_type", 1)
            ])
            
            # Index for predictions
            self._db.predictions.create_index([("timestamp", -1)])
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    @property
    def db(self):
        return self._db
    
    def get_collection(self, name):
        return self._db[name]
    
    def close(self):
        if self._client:
            self._client.close()

# Global instance
mongo_connector = MongoConnector()

def get_db():
    return mongo_connector.db