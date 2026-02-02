# Technical Design Document

## System Architecture Overview

### High-Level Architecture

The Krishi platform follows a modular, service-oriented architecture designed for scalability and maintainability. The system is built using Python/Flask as the web framework with MongoDB for data persistence.

#### Core Components

1. **Web Application Layer** (Flask)
   - User interface and API endpoints
   - Request routing and response handling
   - Authentication and session management

2. **Business Logic Layer**
   - Crop yield prediction engine
   - Explainable AI service
   - Recommendation generation
   - Market intelligence processing

3. **Data Layer**
   - MongoDB database for persistent storage
   - External API integrations
   - Data validation and transformation

4. **External Services Integration**
   - Weather data APIs
   - Soil database services
   - Market data providers
   - LLM services (OpenAI, Groq)

### Technology Stack

- **Backend**: Python 3.8+, Flask web framework
- **Database**: MongoDB for document storage
- **Machine Learning**: scikit-learn, joblib for model persistence
- **External APIs**: Weather services, soil databases, market data
- **LLM Integration**: OpenAI API, Groq API
- **Frontend**: HTML5, CSS3, JavaScript (responsive design)

## Data Flow

### Primary Data Flow

1. **User Input Collection**
   - Farm profile creation and updates
   - Crop selection and planting schedules
   - Manual data entry validation

2. **External Data Aggregation**
   - Weather data retrieval based on farm coordinates
   - Soil composition data lookup
   - Market price and trend data collection

3. **Data Processing Pipeline**
   - Data validation and cleaning
   - Feature engineering for ML models
   - Data transformation for prediction inputs

4. **Prediction Generation**
   - ML model inference for yield prediction
   - Confidence interval calculation
   - Result validation and storage

5. **Insight and Recommendation Generation**
   - Explainable AI analysis of prediction factors
   - LLM-powered recommendation generation
   - Localization and cultural adaptation

6. **User Interface Presentation**
   - Dashboard data aggregation
   - Visualization preparation
   - Response formatting and delivery

### Data Storage Strategy

- **Farm Profiles**: MongoDB collections with indexed location data
- **Predictions**: Time-series storage with model versioning
- **External Data Cache**: TTL-based caching for API responses
- **User Sessions**: In-memory session storage with database backup

## ML Pipeline

### Model Architecture

The crop yield prediction system uses an ensemble approach combining multiple machine learning algorithms:

#### Primary Models
1. **Random Forest Regressor**
   - Handles non-linear relationships
   - Provides feature importance rankings
   - Robust to outliers and missing data

2. **Gradient Boosting Regressor**
   - Captures complex patterns in agricultural data
   - Sequential learning for improved accuracy
   - Handles heterogeneous data types

3. **Linear Regression (Baseline)**
   - Interpretable baseline model
   - Fast inference for real-time predictions
   - Fallback when complex models fail

#### Feature Engineering

**Input Features:**
- **Soil Features**: pH, nitrogen, phosphorus, potassium, organic matter
- **Weather Features**: Temperature (min/max/avg), precipitation, humidity, solar radiation
- **Farm Features**: Location coordinates, elevation, farm size, irrigation type
- **Crop Features**: Crop type, variety, planting date, growth stage
- **Historical Features**: Previous yields, seasonal patterns, multi-year trends

**Derived Features:**
- Growing degree days (GDD)
- Precipitation accumulation periods
- Soil moisture indices
- Seasonal weather anomalies
- Crop-specific stress indicators

#### Model Training Pipeline

1. **Data Preprocessing**
   ```python
   # Data cleaning and validation
   - Remove outliers using IQR method
   - Handle missing values with domain-specific imputation
   - Normalize numerical features
   - Encode categorical variables
   ```

2. **Feature Selection**
   ```python
   # Feature importance analysis
   - Recursive feature elimination
   - Correlation analysis
   - Domain expert validation
   ```

3. **Model Training**
   ```python
   # Cross-validation and hyperparameter tuning
   - 5-fold cross-validation
   - Grid search for optimal parameters
   - Ensemble weight optimization
   ```

4. **Model Validation**
   ```python
   # Performance metrics
   - Mean Absolute Error (MAE)
   - Root Mean Square Error (RMSE)
   - R-squared score
   - Prediction interval coverage
   ```

#### Model Deployment and Versioning

- **Model Serialization**: joblib for Python model persistence
- **Version Control**: Model versioning with metadata tracking
- **A/B Testing**: Gradual rollout of new model versions
- **Performance Monitoring**: Continuous accuracy tracking

### Prediction Confidence and Uncertainty

The system provides prediction confidence through:
- **Prediction Intervals**: 95% confidence intervals for yield estimates
- **Model Ensemble Variance**: Disagreement between models as uncertainty measure
- **Data Quality Scores**: Confidence based on input data completeness
- **Historical Accuracy**: Model performance on similar farms and conditions

## Explainable AI Layer

### SHAP (SHapley Additive exPlanations) Integration

The explainable AI component uses SHAP values to provide interpretable insights:

#### Feature Importance Analysis
```python
# SHAP value calculation for prediction explanation
import shap

# Generate SHAP values for prediction
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_features)

# Identify top contributing factors
top_features = get_top_shap_features(shap_values, feature_names)
```

#### Explanation Generation

1. **Factor Identification**
   - Rank features by absolute SHAP values
   - Group related features (e.g., all weather features)
   - Identify positive and negative contributors

2. **Natural Language Explanation**
   - Convert SHAP values to farmer-friendly explanations
   - Provide actionable insights where possible
   - Include confidence levels for each explanation

3. **Visual Explanations**
   - SHAP waterfall plots for individual predictions
   - Feature importance bar charts
   - Partial dependence plots for key features

#### Explanation Categories

**Environmental Factors:**
- Weather impact (temperature, rainfall, humidity)
- Soil condition effects (pH, nutrients, moisture)
- Seasonal and climatic influences

**Management Factors:**
- Irrigation timing and quantity
- Fertilizer application rates
- Planting date optimization
- Crop variety selection

**External Factors:**
- Regional growing conditions
- Historical performance patterns
- Market-driven recommendations

## LLM Integration

### Multi-Provider Architecture

The system integrates with multiple LLM providers for robust recommendation generation:

#### OpenAI Integration
```python
# OpenAI API configuration
openai_config = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 30
}
```

#### Groq Integration
```python
# Groq API configuration
groq_config = {
    "model": "llama2-70b-4096",
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 30
}
```

### Recommendation Generation Pipeline

1. **Context Preparation**
   ```python
   # Prepare agricultural context for LLM
   context = {
       "farm_profile": farm_data,
       "prediction_results": yield_prediction,
       "explanation_factors": shap_insights,
       "market_conditions": market_data,
       "seasonal_context": seasonal_info
   }
   ```

2. **Prompt Engineering**
   ```python
   # Structured prompts for agricultural recommendations
   prompt_template = """
   Based on the following farm data and yield prediction:
   
   Farm Location: {location}
   Crop Type: {crop}
   Predicted Yield: {yield} ± {confidence}
   Key Factors: {factors}
   Market Conditions: {market}
   
   Provide specific, actionable farming recommendations for:
   1. Irrigation management
   2. Fertilizer application
   3. Pest and disease prevention
   4. Harvest timing
   5. Market strategy
   """
   ```

3. **Response Processing**
   ```python
   # Validate and structure LLM responses
   - Parse structured recommendations
   - Validate agricultural accuracy
   - Add confidence scores
   - Format for user presentation
   ```

### Failover and Quality Control

- **Service Failover**: Automatic switching between OpenAI and Groq
- **Response Validation**: Agricultural domain validation of recommendations
- **Quality Scoring**: Relevance and accuracy scoring of generated content
- **Human Review**: Periodic review of recommendation quality

## External APIs

### Weather Service Integration

#### Primary Weather APIs
1. **OpenWeatherMap API**
   - Current weather conditions
   - 5-day weather forecast
   - Historical weather data
   - Agricultural weather indices

2. **WeatherAPI**
   - Backup weather service
   - Extended forecast data
   - Agricultural-specific metrics

#### Weather Data Processing
```python
# Weather data structure
weather_data = {
    "current": {
        "temperature": float,
        "humidity": float,
        "precipitation": float,
        "wind_speed": float,
        "solar_radiation": float
    },
    "forecast": [
        {
            "date": datetime,
            "temp_min": float,
            "temp_max": float,
            "precipitation_prob": float,
            "conditions": str
        }
    ],
    "historical": {
        "growing_degree_days": float,
        "precipitation_total": float,
        "stress_days": int
    }
}
```

### Soil Service Integration

#### Soil Database APIs
1. **SoilGrids API**
   - Global soil property maps
   - Soil type classification
   - Nutrient content estimates

2. **USDA Soil Survey**
   - Detailed US soil data
   - Soil capability classifications
   - Agricultural suitability ratings

#### Soil Data Structure
```python
# Soil data model
soil_data = {
    "physical_properties": {
        "texture": str,
        "bulk_density": float,
        "water_holding_capacity": float,
        "drainage_class": str
    },
    "chemical_properties": {
        "ph": float,
        "organic_carbon": float,
        "nitrogen": float,
        "phosphorus": float,
        "potassium": float
    },
    "classification": {
        "soil_type": str,
        "capability_class": str,
        "suitability_rating": str
    }
}
```

### Market Data Integration

#### Market Data Sources
1. **Agricultural Market APIs**
   - Commodity price feeds
   - Market trend analysis
   - Supply and demand indicators

2. **Regional Price APIs**
   - Local market prices
   - Transportation costs
   - Regional price variations

#### Market Data Processing
```python
# Market data structure
market_data = {
    "current_prices": {
        "crop_type": str,
        "price_per_unit": float,
        "currency": str,
        "market_location": str,
        "last_updated": datetime
    },
    "price_trends": {
        "trend_direction": str,
        "price_change_percent": float,
        "volatility_index": float,
        "seasonal_pattern": dict
    },
    "forecasts": {
        "30_day_forecast": float,
        "90_day_forecast": float,
        "harvest_season_forecast": float,
        "confidence_level": float
    }
}
```

### API Management and Reliability

#### Rate Limiting and Caching
```python
# API rate limiting strategy
rate_limits = {
    "weather_api": "1000 requests/day",
    "soil_api": "500 requests/day",
    "market_api": "2000 requests/day",
    "openai_api": "100 requests/minute",
    "groq_api": "50 requests/minute"
}

# Caching strategy
cache_ttl = {
    "weather_current": 3600,  # 1 hour
    "weather_forecast": 21600,  # 6 hours
    "soil_data": 2592000,  # 30 days
    "market_prices": 1800,  # 30 minutes
    "llm_responses": 86400  # 24 hours
}
```

#### Error Handling and Fallbacks
```python
# API error handling
def api_call_with_fallback(primary_api, fallback_api, params):
    try:
        return primary_api.call(params)
    except APIException as e:
        log_api_error(primary_api, e)
        try:
            return fallback_api.call(params)
        except APIException as e2:
            log_api_error(fallback_api, e2)
            return get_cached_data(params) or default_response
```

## Deployment Considerations

### Infrastructure Architecture

#### Production Environment
```yaml
# Docker deployment configuration
services:
  web:
    image: krishi-app:latest
    ports:
      - "80:5000"
    environment:
      - FLASK_ENV=production
      - MONGODB_URI=mongodb://mongo:27017/krishi
    depends_on:
      - mongo
      - redis

  mongo:
    image: mongo:5.0
    volumes:
      - mongo_data:/data/db
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=${MONGO_PASSWORD}

  redis:
    image: redis:6.2
    volumes:
      - redis_data:/data
```

#### Scalability Considerations

1. **Horizontal Scaling**
   - Load balancer for multiple Flask instances
   - MongoDB replica set for database scaling
   - Redis cluster for session management

2. **Vertical Scaling**
   - CPU optimization for ML model inference
   - Memory optimization for large datasets
   - Storage optimization for historical data

3. **Auto-scaling Policies**
   - CPU-based scaling triggers
   - Memory usage thresholds
   - Request queue length monitoring

### Security Implementation

#### Authentication and Authorization
```python
# JWT-based authentication
from flask_jwt_extended import JWTManager, create_access_token

jwt_config = {
    "JWT_SECRET_KEY": os.environ.get("JWT_SECRET_KEY"),
    "JWT_ACCESS_TOKEN_EXPIRES": timedelta(hours=24),
    "JWT_REFRESH_TOKEN_EXPIRES": timedelta(days=30)
}
```

#### Data Encryption
- **At Rest**: MongoDB encryption with customer-managed keys
- **In Transit**: TLS 1.3 for all API communications
- **API Keys**: Encrypted storage of external service credentials

#### Security Headers
```python
# Security headers configuration
security_headers = {
    "Content-Security-Policy": "default-src 'self'",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
}
```

### Monitoring and Observability

#### Application Monitoring
```python
# Monitoring configuration
monitoring_config = {
    "metrics": {
        "prediction_accuracy": "gauge",
        "api_response_time": "histogram",
        "user_engagement": "counter",
        "error_rate": "gauge"
    },
    "alerts": {
        "high_error_rate": "error_rate > 5%",
        "slow_predictions": "prediction_time > 10s",
        "api_failures": "external_api_failures > 10/hour"
    }
}
```

#### Logging Strategy
```python
# Structured logging
import structlog

logger = structlog.get_logger()

# Log prediction requests
logger.info(
    "prediction_generated",
    farm_id=farm_id,
    crop_type=crop_type,
    predicted_yield=yield_value,
    confidence=confidence_score,
    model_version=model_version
)
```

### Performance Optimization

#### Database Optimization
```javascript
// MongoDB indexes for performance
db.farms.createIndex({ "location.coordinates": "2dsphere" })
db.predictions.createIndex({ "farm_id": 1, "created_at": -1 })
db.market_data.createIndex({ "crop_type": 1, "date": -1 })
```

#### Caching Strategy
```python
# Redis caching implementation
import redis

cache = redis.Redis(host='redis', port=6379, db=0)

def cached_prediction(farm_id, cache_ttl=3600):
    cache_key = f"prediction:{farm_id}"
    cached_result = cache.get(cache_key)
    
    if cached_result:
        return json.loads(cached_result)
    
    # Generate new prediction
    result = generate_prediction(farm_id)
    cache.setex(cache_key, cache_ttl, json.dumps(result))
    return result
```

#### CDN and Static Asset Optimization
- **Static Assets**: CSS, JavaScript, and image optimization
- **CDN Integration**: Global content delivery for improved performance
- **Compression**: Gzip compression for text-based responses

### Backup and Disaster Recovery

#### Backup Strategy
```bash
# Automated backup script
#!/bin/bash
# Daily MongoDB backup
mongodump --uri="mongodb://localhost:27017/krishi" --out="/backups/$(date +%Y%m%d)"

# Backup retention (30 days)
find /backups -type d -mtime +30 -exec rm -rf {} \;
```

#### Disaster Recovery Plan
1. **Recovery Time Objective (RTO)**: 4 hours
2. **Recovery Point Objective (RPO)**: 24 hours
3. **Backup Verification**: Weekly restore testing
4. **Failover Procedures**: Documented manual and automated procedures

### Environment Configuration

#### Development Environment
```python
# Development configuration
class DevelopmentConfig:
    DEBUG = True
    TESTING = False
    MONGODB_URI = "mongodb://localhost:27017/krishi_dev"
    CACHE_TYPE = "simple"
    LOG_LEVEL = "DEBUG"
```

#### Production Environment
```python
# Production configuration
class ProductionConfig:
    DEBUG = False
    TESTING = False
    MONGODB_URI = os.environ.get("MONGODB_URI")
    CACHE_TYPE = "redis"
    LOG_LEVEL = "INFO"
    SSL_REQUIRED = True
```

#### Configuration Management
```python
# Environment-specific configuration loading
import os

config_map = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig
}

config = config_map.get(os.environ.get("FLASK_ENV", "development"))
```

This technical design provides a comprehensive blueprint for implementing the Krishi agricultural intelligence platform, covering all major architectural components, data flows, and deployment considerations while maintaining alignment with the functional requirements.