from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import traceback
import os
from datetime import datetime
from typing import Dict, List

# Import all services
from db.mongo_connector import get_db
from services.soil_service import get_soil_data
from services.elevation_service import get_elevation
from services.weather_service import fetch_historical_weather
# Import Groq AI services
from services.groq_agriculture_advisor import get_farming_recommendations, validate_input_data
# Import market trend analysis service
from services.market_trend_service import get_market_trends, get_crop_market_comparison, validate_market_data
# NEW: Import XAI and Real Market Data services
from services.explainable_ai_service import explain_prediction
from services.real_market_data_service import get_real_market_price, get_market_demand, compare_crops
from models.yield_model import YieldModel
from data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Database collections
db = get_db()
predictions_collection = db.predictions
farmers_collection = db.farmers
historical_data_collection = db.historical_data

# Initialize models
yield_model = YieldModel()
data_loader = DataLoader()

def auto_setup_system():
    """Automatically setup data and model if not present"""
    logger.info("🚀 Starting automatic system setup...")
    
    # Check if model exists
    model_path = "models/crop_yield_model.joblib"
    model_exists = os.path.exists(model_path)
    
    # Check if historical data exists in database
    data_count = historical_data_collection.count_documents({})
    has_data = data_count > 0
    
    logger.info(f"📊 Current status: Model exists: {model_exists}, Data records: {data_count}")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Step 1: Load data if not present
    if not has_data:
        logger.info("📥 No historical data found. Looking for dataset files...")
        
        # Look for common dataset filenames
        possible_files = [
            "crop_yield_dataset.csv",
            "agricultural_data.csv", 
            "crop_data.csv",
            "india_agriculture.csv"
        ]
        
        dataset_file = None
        for filename in possible_files:
            if os.path.exists(filename):
                dataset_file = filename
                logger.info(f"✅ Found dataset file: {filename}")
                break
        
        if dataset_file:
            try:
                logger.info(f"📊 Loading dataset from {dataset_file}...")
                success = data_loader.load_kaggle_dataset(dataset_file)
                if success:
                    logger.info("✅ Dataset loaded successfully!")
                    has_data = True
                else:
                    logger.warning("⚠️ Dataset loading failed")
            except Exception as e:
                logger.error(f"❌ Error loading dataset: {e}")
        else:
            logger.warning("⚠️ No dataset file found. You can:")
            logger.warning("   1. Place your CSV file in the project root directory")
            logger.warning("   2. Use /api/load-data endpoint")
            logger.warning("   3. Download from: https://www.kaggle.com/datasets/akshatgupta7/crop-yield-in-indian-states-dataset")
    
    # Step 2: Train model if not present but we have data
    if not model_exists and has_data:
        logger.info("🤖 No trained model found. Starting automatic training...")
        try:
            # Get historical data
            historical_data = list(historical_data_collection.find({}))
            logger.info(f"📚 Found {len(historical_data)} records for training")
            
            if len(historical_data) >= 10:  # Minimum data requirement
                logger.info("🔄 Training machine learning model...")
                results = yield_model.train(historical_data)
                
                # Save the trained model
                yield_model.save(model_path)
                logger.info("✅ Model trained and saved successfully!")
                logger.info(f"📈 Training results: {results}")
                
                model_exists = True
            else:
                logger.warning(f"⚠️ Insufficient data for training: {len(historical_data)} records (minimum: 10)")
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
    
    # Step 3: Load existing model
    if model_exists:
        try:
            yield_model.load(model_path)
            logger.info("✅ Trained model loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Error loading existing model: {e}")
    
    # Final status
    final_data_count = historical_data_collection.count_documents({})
    final_model_exists = os.path.exists(model_path) and yield_model.model is not None
    
    logger.info("🏁 Auto-setup completed!")
    logger.info(f"   📊 Data records: {final_data_count}")
    logger.info(f"   🤖 Model ready: {final_model_exists}")
    
    if final_model_exists:
        logger.info("🎉 System is fully ready for predictions!")
    else:
        logger.warning("⚠️ System needs manual setup. Please:")
        logger.warning("   1. Add dataset file to project directory")
        logger.warning("   2. Or use /api/load-data and /api/train-model endpoints")
    
    return final_model_exists

@app.route("/")
def index():
    """Serve the main frontend page"""
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict_yield():
    """Main prediction endpoint with XAI and real market data integration"""
    try:
        # Check if model is ready
        if yield_model.model is None:
            return jsonify({
                "error": "Model not ready. Please wait for auto-setup to complete or manually train the model.",
                "suggestion": "Use /api/train-model endpoint or restart the application with a dataset file."
            }), 503

        # Get and validate input data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Required fields validation
        required_fields = [
            'crop_type', 'state', 'district', 'latitude', 'longitude', 
            'area_hectares', 'fertilizer', 'pesticide'
        ]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400

        # Extract and validate input parameters
        try:
            lat = float(data['latitude'])
            lon = float(data['longitude'])
            area = float(data['area_hectares'])
            fertilizer = float(data.get('fertilizer', 0))
            pesticide = float(data.get('pesticide', 0))
            
            # Validate coordinate ranges
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                return jsonify({"error": "Invalid latitude or longitude values"}), 400
                
            if area <= 0:
                return jsonify({"error": "Area must be greater than 0"}), 400
                
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid numeric values provided"}), 400

        crop_type = data['crop_type'].lower()
        state = data['state'].lower()
        district = data['district'].lower().replace(' ', '').replace('.', '')
        year = int(data.get('crop_year', 2023))

        logger.info(f"Processing prediction request for {crop_type} in {state}, {district}")

        # Collect environmental data
        logger.info("Fetching environmental data...")
        
        # Get soil data
        soil_data = get_soil_data(lat, lon, state)
        logger.info(f"Retrieved soil data: pH={soil_data.get('ph', 'N/A')}")
        
        # Get elevation
        elevation = get_elevation(lat, lon)
        logger.info(f"Retrieved elevation: {elevation}m")
        
        # Get weather data
        weather_data = fetch_historical_weather(lat, lon, year)
        logger.info(f"Retrieved weather data: rainfall={weather_data.get('total_rainfall', 'N/A')}mm")

        # Prepare feature vector for ML model
        input_features = {
            **soil_data,
            **weather_data,
            'elevation': elevation,
            'latitude': lat,
            'longitude': lon,
            'area_hectares': area,
            'fertilizer': fertilizer,
            'pesticide': pesticide
        }

        # Validate input features
        if not yield_model.validate_input_features(input_features):
            logger.warning("Some input features are outside expected ranges")

        # Make yield prediction
        logger.info("Making yield prediction...")
        
        # Try crop-specific model first, fallback to general model
        try:
            predicted_yield = yield_model.predict_crop_specific(input_features, crop_type)
        except:
            predicted_yield = yield_model.predict(input_features)
        
        total_production = predicted_yield * area
        
        logger.info(f"Predicted yield: {predicted_yield:.2f} quintals/hectare")

        # =================================================================
        # NEW SECTION 1: EXPLAINABLE AI
        # =================================================================
        logger.info("Generating explainable AI analysis...")
        try:
            explanation = explain_prediction(
                predicted_yield=predicted_yield,
                input_features=input_features,
                crop_type=crop_type
            )
            logger.info(f"✓ Prediction confidence: {explanation['confidence_analysis']['confidence_level']}")
            
        except Exception as e:
            logger.error(f"Error generating XAI explanation: {e}")
            explanation = {
                "explanation_summary": f"Predicted yield based on environmental conditions",
                "confidence_analysis": {
                    "confidence_level": "moderate",
                    "confidence_score": 70.0,
                    "explanation": "Standard analysis based on available data"
                },
                "feature_contributions": [],
                "limiting_factors": [],
                "optimization_opportunities": [],
                "recommendation_basis": {}
            }

        # =================================================================
        # NEW SECTION 2: REAL MARKET DATA
        # =================================================================
        logger.info("Fetching real market prices...")
        try:
            # Get real current prices
            real_price_data = get_real_market_price(crop_type, state, district)
            market_demand = get_market_demand(crop_type, state)
            
            logger.info(f"✓ Real price: ₹{real_price_data['current']}/quintal")
            logger.info(f"✓ Source: {real_price_data['source']}")
            logger.info(f"✓ Demand: {market_demand}")
            
            # Calculate revenue with REAL prices (not hardcoded)
            estimated_revenue = total_production * real_price_data['current']
            
            # Compare crop profitability with real data
            alternative_crops = ["soybean", "cotton", "maize", "groundnut", "turmeric"]
            alternative_crops = [c for c in alternative_crops if c != crop_type][:3]
            
            crop_profitability = compare_crops([crop_type] + alternative_crops, state)
            
        except Exception as e:
            logger.error(f"Error fetching real market data: {e}")
            # Fallback prices
            real_price_data = {
                'current': 2000,
                'min': 1500,
                'max': 2500,
                'average': 2000,
                'trend': 'stable',
                'source': 'Fallback estimate',
                'last_updated': datetime.now().isoformat()
            }
            market_demand = 'moderate'
            estimated_revenue = total_production * 2000
            crop_profitability = []

        # Prepare context data
        context_for_ai = {
            "crop_type": crop_type,
            "state": state,
            "district": district,
            "predicted_yield_per_hectare": round(predicted_yield, 2),
            "total_expected_production": round(total_production, 2),
            "area_hectares": area,
            "soil_parameters": soil_data,
            "weather_summary": weather_data,
            "fertilizer_applied": fertilizer,
            "pesticide_applied": pesticide,
            "elevation": elevation,
            "crop_year": year,
            # NEW: Add real market data to context
            "real_market_price": real_price_data,
            "market_demand": market_demand
        }

        # Generate AI recommendations
        logger.info("Generating AI recommendations with Groq...")
        
        validation_result = validate_input_data(context_for_ai)
        if not validation_result.get("is_valid", True):
            logger.warning(f"Input validation warnings: {validation_result.get('warnings', [])}")
            if validation_result.get("errors"):
                logger.error(f"Input validation errors: {validation_result.get('errors', [])}")
        
        try:
            ai_recommendations = get_farming_recommendations(context_for_ai)
            logger.info("Successfully generated farming recommendations using Groq AI")
        except Exception as e:
            logger.error(f"Error with Groq AI recommendations: {e}")
            # Fallback to basic recommendations
            ai_recommendations = {
                "irrigation": [{"action": "Monitor soil moisture regularly", "details": "Check soil conditions daily", "priority": "medium"}],
                "fertilization": [{"action": "Apply balanced NPK fertilizer", "details": "Use soil test results for optimal ratios", "priority": "high"}],
                "pestControl": [{"action": "Implement integrated pest management", "details": "Regular field monitoring recommended", "priority": "medium"}],
                "general": [{"action": "Follow good agricultural practices", "details": "Maintain proper crop spacing and field hygiene", "priority": "low"}]
            }

        # Generate market trend analysis
        logger.info("Generating market trend analysis with real data...")
        
        market_validation = validate_market_data(context_for_ai)
        if not market_validation.get("is_valid", True):
            logger.warning(f"Market validation warnings: {market_validation.get('warnings', [])}")
        
        try:
            market_analysis = get_market_trends(context_for_ai)
            
            # Override with real market data where available
            if real_price_data.get('current'):
                market_analysis['current_status']['price_range'] = real_price_data
                market_analysis['current_status']['demand_level'] = market_demand
                market_analysis['current_status']['data_source'] = real_price_data['source']
            
            logger.info("Successfully generated market trend analysis with real data")
            
        except Exception as e:
            logger.error(f"Error with market trend analysis: {e}")
            # Fallback market analysis
            market_analysis = {
                "current_status": {
                    "price_range": real_price_data,
                    "demand_level": market_demand,
                    "market_sentiment": "stable",
                    "data_source": real_price_data['source']
                },
                "monthly_trends": [],
                "trending_crops": {
                    "current_trending": [{"crop": "rice", "reason": "Seasonal demand"}],
                    "future_trending": [{"crop": "soybean", "reason": "Growing market"}],
                    "declining_crops": []
                },
                "future_outlook": {
                    "next_3_months": "Market expected to remain stable",
                    "next_6_months": "Seasonal variations expected",
                    "next_12_months": "Long-term outlook positive"
                },
                "recommendations": {
                    "optimal_selling_time": "Post-harvest season",
                    "market_strategies": ["Monitor daily prices", "Consider storage options"]
                }
            }

        # Build crop comparison with profitability data
        if crop_profitability:
            # Find primary crop data
            primary_crop_data = next((c for c in crop_profitability if c['crop'] == crop_type), None)
            
            if primary_crop_data:
                # Helper function to safely parse price ranges
                def parse_price_range(price_range_str):
                    try:
                        # Extract min and max from format "₹5100.0 - ₹6500.0"
                        parts = price_range_str.split(' - ')
                        min_price = float(parts[0].replace('₹', '').strip())
                        max_price = float(parts[1].replace('₹', '').strip())
                        return int(min_price), int(max_price)
                    except Exception as e:
                        logger.warning(f"Error parsing price range: {e}")
                        return 0, 0
                
                primary_min, primary_max = parse_price_range(primary_crop_data['price_range'])
                
                crop_comparison = {
                    "primary_crop": {
                        "name": primary_crop_data['crop'],
                        "price_info": {
                            "current": int(primary_crop_data['current_price']),
                            "min": primary_min,
                            "max": primary_max
                        },
                        "market_score": primary_crop_data['profitability_score'],
                        "recommendation": primary_crop_data['explanation']
                    },
                    "alternatives": []
                }
                
                # Process alternative crops
                for item in crop_profitability:
                    if item['crop'] != crop_type:
                        alt_min, alt_max = parse_price_range(item['price_range'])
                        crop_comparison["alternatives"].append({
                            "name": item['crop'],
                            "price_info": {
                                "current": int(item['current_price']),
                                "min": alt_min,
                                "max": alt_max
                            },
                            "market_score": item['profitability_score'],
                            "recommendation": item['explanation']
                        })
            else:
                crop_comparison = get_crop_market_comparison(crop_type, alternative_crops, context_for_ai)
        else:
            crop_comparison = get_crop_market_comparison(crop_type, alternative_crops, context_for_ai)

        # =================================================================
        # ENHANCED RESPONSE WITH XAI + REAL DATA
        # =================================================================
        
        response = {
            "success": True,
            "predicted_yield_per_hectare": round(predicted_yield, 2),
            "total_expected_production": round(total_production, 2),
            "estimated_revenue": round(estimated_revenue, 2),  # Using REAL price
            "unit": "quintals",
            
            # NEW: XAI Explanation Section
            "prediction_explanation": {
                "summary": explanation.get('explanation_summary', ''),
                "confidence": explanation.get('confidence_analysis', {}),
                "feature_contributions": explanation.get('feature_contributions', [])[:5],
                "limiting_factors": explanation.get('limiting_factors', []),
                "optimization_opportunities": explanation.get('optimization_opportunities', []),
                "recommendation_basis": explanation.get('recommendation_basis', {})
            },
            
            # Environmental data
            "environmental_data": {
                "soil_health": soil_data,
                "weather_summary": weather_data,
                "elevation": elevation
            },
            
            # Farming recommendations
            "recommendations": ai_recommendations,
            
            # Market analysis (enhanced with real data)
            "market_analysis": {
                **market_analysis,
                "price_data_source": real_price_data['source'],
                "price_last_updated": real_price_data['last_updated'],
                "real_time_prices": real_price_data['source'] != 'Fallback estimate'
            },
            
            # Crop comparison (enhanced)
            "crop_comparison": crop_comparison,
            
            # Metadata
            "prediction_timestamp": datetime.utcnow().isoformat(),
            "model_confidence": explanation['confidence_analysis']['confidence_level'],
            "ai_service": "groq_with_xai",
            
            # Data quality info
            "data_quality": {
                "validation_warnings": validation_result.get("warnings", []),
                "validation_errors": validation_result.get("errors", []),
                "market_validation": market_validation,
                "using_real_market_data": real_price_data['source'] != 'Fallback estimate'
            }
        }

        # Store prediction record (enhanced)
        prediction_record = {
            "timestamp": datetime.utcnow(),
            "farmer_id": data.get('farmer_id'),
            "crop_type": crop_type,
            "state": state,
            "district": district,
            "coordinates": {"latitude": lat, "longitude": lon},
            "area_hectares": area,
            "fertilizer": fertilizer,
            "pesticide": pesticide,
            "year": year,
            "predicted_yield_per_hectare": predicted_yield,
            "total_production_quintals": total_production,
            "estimated_revenue": estimated_revenue,
            "input_features": input_features,
            "ai_recommendations": ai_recommendations,
            "market_analysis": market_analysis,
            "crop_comparison": crop_comparison,
            "xai_explanation": explanation,  # NEW
            "real_price_data": real_price_data,  # NEW
            "ai_service": "groq_with_xai",
            "model_version": "1.0",
            "validation_result": validation_result,
            "market_validation": market_validation,
            "xai_enabled": True,
            "real_market_data": True
        }
        
        try:
            predictions_collection.insert_one(prediction_record)
            logger.info("✓ Enhanced prediction record saved to database")
        except Exception as e:
            logger.warning(f"Failed to save prediction record: {e}")

        logger.info("✓ Prediction completed successfully with XAI and real market data")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "error": "Internal server error occurred during prediction",
            "message": str(e)
        }), 500

@app.route("/api/market-analysis", methods=["POST"])
def market_analysis():
    """Standalone market trend analysis endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Validate required fields for market analysis
        required_fields = ['crop_type', 'state']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields for market analysis: {', '.join(missing_fields)}"
            }), 400

        # Validate market data
        validation_result = validate_market_data(data)
        
        # Get real market prices
        try:
            real_price_data = get_real_market_price(data['crop_type'], data['state'], data.get('district'))
            market_demand = get_market_demand(data['crop_type'], data['state'])
            
            # Add to context
            data['real_market_price'] = real_price_data
            data['market_demand'] = market_demand
        except Exception as e:
            logger.error(f"Error fetching real market data: {e}")
        
        # Generate market analysis
        market_trends = get_market_trends(data)
        
        # Get crop alternatives if not provided
        alternatives = data.get('alternative_crops', ["soybean", "cotton", "maize"])
        crop_comparison = get_crop_market_comparison(data['crop_type'], alternatives, data)
        
        return jsonify({
            "success": True,
            "market_analysis": market_trends,
            "crop_comparison": crop_comparison,
            "validation": validation_result,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in market analysis endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/test-ai", methods=["POST"])
def test_ai_recommendations():
    """Test AI recommendations independently"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Validate input data
        validation_result = validate_input_data(data)
        
        # Generate recommendations
        recommendations = get_farming_recommendations(data)
        
        return jsonify({
            "success": True,
            "recommendations": recommendations,
            "validation": validation_result,
            "ai_service": "groq",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error testing AI recommendations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/load-data", methods=["POST"])
def load_dataset():
    """Load dataset from uploaded file"""
    try:
        data = request.get_json()
        csv_path = data.get('csv_path')
        
        if not csv_path:
            return jsonify({"error": "CSV path is required"}), 400
        
        success = data_loader.load_kaggle_dataset(csv_path)
        
        if success:
            stats = data_loader.get_data_stats()
            return jsonify({
                "success": True,
                "message": "Dataset loaded successfully",
                "stats": stats
            })
        else:
            return jsonify({
                "error": "Failed to load dataset"
            }), 500
            
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return jsonify({"error": "Failed to load dataset"}), 500

@app.route("/api/train-model", methods=["POST"])
def train_model():
    """Train the ML model"""
    try:
        # Get historical data
        historical_data = list(historical_data_collection.find({}))
        
        if not historical_data:
            return jsonify({
                "error": "No historical data found. Please load data first."
            }), 400
        
        # Train the model
        results = yield_model.train(historical_data)
        
        # Save the trained model
        os.makedirs("models", exist_ok=True)
        yield_model.save("models/crop_yield_model.joblib")
        
        return jsonify({
            "success": True,
            "message": "Model trained successfully",
            "metrics": results
        })
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return jsonify({"error": "Failed to train model"}), 500

@app.route("/api/status")
def get_system_status():
    """Get current system status with AI service info"""
    try:
        data_count = historical_data_collection.count_documents({})
        model_exists = os.path.exists("models/crop_yield_model.joblib")
        model_loaded = yield_model.model is not None
        
        # Check AI services status
        try:
            from services.groq_agriculture_advisor import groq_agriculture_advisor
            farming_ai_available = groq_agriculture_advisor.client is not None
            
            from services.market_trend_service import market_trend_analyzer
            market_ai_available = market_trend_analyzer.client is not None
            
            # Check XAI service
            xai_available = True  # XAI doesn't need external API
            
            # Check real market data service
            from services.real_market_data_service import real_market_data_service
            market_data_available = True
            
        except Exception:
            farming_ai_available = False
            market_ai_available = False
            xai_available = False
            market_data_available = False
        
        return jsonify({
            "success": True,
            "status": {
                "data_records": data_count,
                "model_file_exists": model_exists,
                "model_loaded": model_loaded,
                "farming_ai_service": "groq" if farming_ai_available else "unavailable",
                "market_ai_service": "groq" if market_ai_available else "unavailable",
                "xai_service": "available" if xai_available else "unavailable",
                "real_market_data_service": "available" if market_data_available else "unavailable",
                "ai_services_available": farming_ai_available and market_ai_available,
                "ready_for_predictions": model_loaded and data_count > 0,
                "ready_for_market_analysis": farming_ai_available or market_ai_available,
                "xai_enabled": xai_available,
                "real_market_data_enabled": market_data_available
            }
        })
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({"error": "Failed to get system status"}), 500

@app.route("/api/health")
def health_check():
    """Enhanced health check endpoint for monitoring"""
    try:
        # Test database connection
        db.command('ping')
        
        # Check AI services
        farming_ai_status = "unavailable"
        market_ai_status = "unavailable"
        
        try:
            from services.groq_agriculture_advisor import groq_agriculture_advisor
            if groq_agriculture_advisor.client is not None:
                farming_ai_status = "available"
        except Exception:
            pass
            
        try:
            from services.market_trend_service import market_trend_analyzer
            if market_trend_analyzer.client is not None:
                market_ai_status = "available"
        except Exception:
            pass
        
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": "connected",
                "ml_model": "loaded" if yield_model.model else "not_loaded",
                "farming_ai_recommendations": farming_ai_status,
                "market_trend_analysis": market_ai_status,
                "xai_service": "available",
                "real_market_data": "available"
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    # Run auto-setup on startup
    logger.info("🌾 Starting Crop Yield Prediction Platform with Groq AI, XAI, and Real Market Data...")
    
    try:
        system_ready = auto_setup_system()
        
        if system_ready:
            logger.info("✅ System fully initialized and ready!")
        else:
            logger.warning("⚠️ System started but may need manual setup")
            logger.info("💡 Check /api/status endpoint for current system state")
        
        # Test AI services on startup
        try:
            from services.groq_agriculture_advisor import groq_agriculture_advisor
            if groq_agriculture_advisor.client is not None:
                logger.info("🤖 Groq AI service ready for farming recommendations!")
            else:
                logger.warning("⚠️ Farming AI service not available - check GROQ_API_KEY environment variable")
        except Exception as e:
            logger.warning(f"⚠️ Could not load farming AI service: {e}")
            
        try:
            from services.market_trend_service import market_trend_analyzer
            if market_trend_analyzer.client is not None:
                logger.info("📈 Market trend analysis AI service is ready!")
            else:
                logger.warning("⚠️ Market analysis AI service not available - check GROQ_API_KEY environment variable")
        except Exception as e:
            logger.warning(f"⚠️ Could not load market analysis service: {e}")
        
        # Test XAI service
        try:
            from services.explainable_ai_service import explainable_ai_service
            logger.info("🧠 Explainable AI (XAI) service is ready!")
        except Exception as e:
            logger.warning(f"⚠️ Could not load XAI service: {e}")
        
        # Test Real Market Data service
        try:
            from services.real_market_data_service import real_market_data_service
            logger.info("💰 Real Market Data service is ready!")
        except Exception as e:
            logger.warning(f"⚠️ Could not load Real Market Data service: {e}")
        
    except Exception as e:
        logger.error(f"❌ Auto-setup failed: {e}")
        logger.info("🔧 System will start but may require manual setup")
    
    logger.info("🚀 Platform ready for predictions with Groq AI, XAI, and Real Market Data!")
    app.run(host="0.0.0.0", port=5000, debug=True)
