import os
import logging
import json
from typing import Dict, List, Optional, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class GroqAgricultureAdvisor:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq Agriculture Advisor
        
        Args:
            api_key: Groq API key. If None, will try to get from environment
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            logger.warning("Groq API key not found. AI recommendations will be unavailable.")
            self.client = None
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info("Groq client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.client = None

    def get_recommendations(self, context_data: Dict) -> Dict[str, List[Dict]]:
        """Generate structured farming recommendations via Groq LLM"""
        if not self.client:
            logger.warning("Groq client not available, using default recommendations")
            return self._default_recommendations()

        try:
            prompt = self._build_comprehensive_prompt(context_data)
            
            # Use the current production model
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Updated to current production model
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower for more consistent results
                max_tokens=1500,  # Increased for more detailed responses
                top_p=0.9,
                stream=False,
            )

            advice_text = completion.choices[0].message.content.strip()
            logger.info("Successfully received recommendations from Groq")
            
            return self._parse_structured_recommendations(advice_text, context_data)

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return self._default_recommendations()

    def _get_system_prompt(self) -> str:
        """Enhanced system prompt for better AI performance"""
        return """You are Dr. AgriBot, an expert agricultural advisor specializing in precision farming for Indian agricultural conditions. 

Your expertise includes:
- Soil science and nutrient management
- Crop physiology and growth optimization
- Integrated pest management (IPM)
- Water resource management and irrigation
- Climate-smart agriculture practices
- Sustainable farming techniques
- Regional crop varieties and local conditions

IMPORTANT: Structure your response using these exact section headers:
### IRRIGATION RECOMMENDATIONS
### FERTILIZATION RECOMMENDATIONS  
### PEST CONTROL RECOMMENDATIONS
### GENERAL FARMING ADVICE

For each recommendation:
- Provide specific, actionable advice
- Include quantities, timing, and methods where applicable
- Consider local Indian farming conditions and constraints
- Mention priority level (HIGH/MEDIUM/LOW) in each point
- Be practical and cost-effective

Focus on evidence-based recommendations that consider the farmer's current inputs, soil conditions, weather patterns, and crop requirements."""

    def _build_comprehensive_prompt(self, context_data: Dict) -> str:
        """Build detailed prompt with all available data"""
        crop = context_data.get("crop_type", "unknown")
        yield_pred = context_data.get("predicted_yield_per_hectare", 0)
        area = context_data.get('area_hectares', 'unknown')
        soil = context_data.get("soil_parameters", {})
        weather = context_data.get("weather_summary", {})
        current_fertilizer = context_data.get('fertilizer_applied', 0)
        current_pesticide = context_data.get('pesticide_applied', 0)

        # Enhanced prompt with more context
        prompt = f"""Please analyze the following agricultural data and provide comprehensive farming recommendations:

**FARM PROFILE:**
- Crop Type: {crop.title() if crop != 'unknown' else 'Not specified'}
- Predicted Yield: {yield_pred} quintals/hectare
- Farm Area: {area} hectares
- Current Season: Based on weather data provided

**SOIL ANALYSIS:**
- pH Level: {soil.get('ph', 'Not tested')}
- Nitrogen Content: {soil.get('nitrogen', 'Unknown')} g/kg
- Organic Carbon: {soil.get('organic_carbon', 'Unknown')}%
- Sand Content: {soil.get('sand_content', 'Unknown')}%
- Clay Content: {soil.get('clay_content', 'Unknown')}%
- Soil Type Assessment: {self._assess_soil_type(soil)}

**WEATHER CONDITIONS:**
- Total Rainfall: {weather.get('total_rainfall', 'Unknown')} mm
- Average Max Temperature: {weather.get('avg_temp_max', 'Unknown')}°C
- Average Min Temperature: {weather.get('avg_temp_min', 'Unknown')}°C
- Average Humidity: {weather.get('avg_humidity', 'Unknown')}%
- Weather Assessment: {self._assess_weather_conditions(weather)}

**CURRENT INPUT USAGE:**
- Fertilizer Applied: {current_fertilizer} kg/hectare
- Pesticide Applied: {current_pesticide} kg/hectare
- Input Efficiency: {self._assess_input_efficiency(current_fertilizer, current_pesticide, yield_pred)}

**SPECIFIC REQUIREMENTS:**
Please provide detailed recommendations addressing:
1. Optimal irrigation scheduling and water management
2. Nutrient management and fertilization strategy
3. Integrated pest and disease management
4. General crop management and optimization tips

Consider the current input levels and suggest adjustments for optimal yield and cost-effectiveness."""

        return prompt

    def _assess_soil_type(self, soil: Dict) -> str:
        """Assess soil type based on composition"""
        try:
            sand = float(soil.get('sand_content', 0))
            clay = float(soil.get('clay_content', 0))
            
            if sand > 70:
                return "Sandy soil - good drainage, may need frequent irrigation"
            elif clay > 40:
                return "Clay soil - good water retention, may need drainage management"
            else:
                return "Loamy soil - well-balanced for most crops"
        except (ValueError, TypeError):
            return "Soil composition analysis needed"

    def _assess_weather_conditions(self, weather: Dict) -> str:
        """Assess weather suitability"""
        try:
            rainfall = float(weather.get('total_rainfall', 0))
            temp_max = float(weather.get('avg_temp_max', 0))
            
            conditions = []
            if rainfall < 400:
                conditions.append("Low rainfall - irrigation critical")
            elif rainfall > 1200:
                conditions.append("High rainfall - drainage important")
            
            if temp_max > 35:
                conditions.append("High temperature stress risk")
            elif temp_max < 20:
                conditions.append("Cool conditions - may affect growth rate")
                
            return "; ".join(conditions) if conditions else "Moderate conditions"
        except (ValueError, TypeError):
            return "Weather analysis needed"

    def _assess_input_efficiency(self, fertilizer: float, pesticide: float, yield_pred: float) -> str:
        """Assess current input efficiency"""
        try:
            if yield_pred > 0:
                fert_efficiency = yield_pred / max(fertilizer, 1)
                if fert_efficiency > 0.5:
                    return "Good input efficiency"
                elif fert_efficiency > 0.2:
                    return "Moderate efficiency - optimization possible"
                else:
                    return "Low efficiency - review input strategy"
            return "Efficiency assessment pending"
        except (ValueError, TypeError):
            return "Unable to assess efficiency"

    def _parse_structured_recommendations(self, advice_text: str, context_data: Dict) -> Dict[str, List[Dict]]:
        """Enhanced parser with better structure recognition"""
        categories = {
            "irrigation": [],
            "fertilization": [], 
            "pestControl": [],
            "general": []
        }
        
        lines = advice_text.split("\n")
        current_category = None
        
        # Section mapping
        section_map = {
            "irrigation": ["irrigation", "water", "watering"],
            "fertilization": ["fertilization", "fertilizer", "nutrient", "npk", "organic"],
            "pestControl": ["pest", "disease", "insect", "fungicide", "herbicide"],
            "general": ["general", "harvest", "planting", "management", "advice"]
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            line_lower = line.lower()
            for category, keywords in section_map.items():
                if any(keyword in line_lower for keyword in keywords) and "###" in line:
                    current_category = category
                    break
            
            # Parse recommendation points
            if current_category and (line.startswith(("-", "•", "*")) or 
                                   (line[0].isdigit() and "." in line[:3])):
                
                cleaned_text = self._clean_text(line)
                if cleaned_text:  # Only add non-empty recommendations
                    priority = self._extract_priority(cleaned_text)
                    categories[current_category].append({
                        "action": cleaned_text,
                        "details": self._add_contextual_details(cleaned_text, context_data),
                        "priority": priority,
                        "category": current_category
                    })
        
        # Ensure minimum recommendations per category
        self._ensure_minimum_recommendations(categories, context_data)
        
        return categories

    def _clean_text(self, text: str) -> str:
        """Clean and format recommendation text"""
        # Remove markers and clean up
        text = text.lstrip("-•*0123456789. ").strip()
        
        # Remove priority indicators for cleaner action text
        priority_words = ["HIGH:", "MEDIUM:", "LOW:", "(HIGH)", "(MEDIUM)", "(LOW)"]
        for word in priority_words:
            text = text.replace(word, "").strip()
            
        return text

    def _extract_priority(self, text: str) -> str:
        """Extract priority level from recommendation text"""
        text_upper = text.upper()
        
        if any(indicator in text_upper for indicator in ["HIGH", "URGENT", "CRITICAL", "IMMEDIATELY", "ESSENTIAL"]):
            return "high"
        elif any(indicator in text_upper for indicator in ["MEDIUM", "SHOULD", "RECOMMENDED", "IMPORTANT"]):
            return "medium"
        else:
            return "low"

    def _add_contextual_details(self, recommendation: str, context_data: Dict) -> str:
        """Add contextual details based on farm data"""
        rec_lower = recommendation.lower()
        details = []
        
        # Irrigation context
        if any(word in rec_lower for word in ["water", "irrigat"]):
            rainfall = context_data.get("weather_summary", {}).get("total_rainfall", 0)
            if isinstance(rainfall, (int, float)) and rainfall < 500:
                details.append("Critical due to low rainfall conditions")
            elif isinstance(rainfall, (int, float)) and rainfall > 1200:
                details.append("Consider drainage management due to high rainfall")
        
        # Fertilization context  
        if any(word in rec_lower for word in ["fertiliz", "nutrient", "npk"]):
            soil = context_data.get("soil_parameters", {})
            ph = soil.get("ph")
            if isinstance(ph, (int, float)):
                if ph < 6.0:
                    details.append("Soil pH is acidic - consider lime application")
                elif ph > 8.0:
                    details.append("Soil pH is alkaline - use acidifying fertilizers")
        
        # Pest control context
        if any(word in rec_lower for word in ["pest", "spray", "insect"]):
            humidity = context_data.get("weather_summary", {}).get("avg_humidity", 0)
            if isinstance(humidity, (int, float)) and humidity > 80:
                details.append("High humidity increases disease pressure")
        
        return "; ".join(details) if details else "Follow local agricultural extension guidelines"

    def _ensure_minimum_recommendations(self, categories: Dict, context_data: Dict):
        """Ensure each category has at least one recommendation"""
        defaults = self._get_category_defaults(context_data)
        
        for category, recs in categories.items():
            if not recs:  # If category is empty
                categories[category] = defaults.get(category, [])

    def _get_category_defaults(self, context_data: Dict) -> Dict[str, List[Dict]]:
        """Get contextual default recommendations"""
        crop = context_data.get("crop_type", "general crop")
        
        return {
            "irrigation": [
                {
                    "action": f"Monitor soil moisture levels for {crop}",
                    "details": "Check moisture at root zone depth before each irrigation",
                    "priority": "high",
                    "category": "irrigation"
                }
            ],
            "fertilization": [
                {
                    "action": f"Apply balanced fertilization for {crop}",
                    "details": "Base application on soil test results and crop requirements",
                    "priority": "high",
                    "category": "fertilization"
                }
            ],
            "pestControl": [
                {
                    "action": f"Implement IPM for {crop}",
                    "details": "Regular monitoring and integrated approach to pest management",
                    "priority": "medium",
                    "category": "pestControl"
                }
            ],
            "general": [
                {
                    "action": f"Follow good agricultural practices for {crop}",
                    "details": "Maintain field hygiene and proper crop management",
                    "priority": "medium",
                    "category": "general"
                }
            ]
        }

    def _default_recommendations(self) -> Dict[str, List[Dict]]:
        """Comprehensive fallback recommendations when Groq is unavailable"""
        return {
            "irrigation": [
                {
                    "action": "Implement drip irrigation system",
                    "details": "More efficient water usage and reduced disease pressure",
                    "priority": "high",
                    "category": "irrigation"
                },
                {
                    "action": "Monitor soil moisture regularly",
                    "details": "Use moisture meter or finger test at 6-inch depth",
                    "priority": "medium",
                    "category": "irrigation"
                }
            ],
            "fertilization": [
                {
                    "action": "Apply balanced NPK fertilizer",
                    "details": "Use 19:19:19 or similar balanced formulation",
                    "priority": "high",
                    "category": "fertilization"
                },
                {
                    "action": "Add organic matter",
                    "details": "Apply compost or farmyard manure at 2-3 tons/hectare",
                    "priority": "medium",
                    "category": "fertilization"
                }
            ],
            "pestControl": [
                {
                    "action": "Regular field scouting",
                    "details": "Inspect crops twice weekly for pest and disease symptoms",
                    "priority": "high",
                    "category": "pestControl"
                },
                {
                    "action": "Use pheromone traps",
                    "details": "Install appropriate traps for early pest detection",
                    "priority": "medium",
                    "category": "pestControl"
                }
            ],
            "general": [
                {
                    "action": "Maintain optimal plant population",
                    "details": "Follow recommended spacing for better yield and disease management",
                    "priority": "medium",
                    "category": "general"
                },
                {
                    "action": "Plan harvest timing",
                    "details": "Monitor maturity indicators for optimal quality and yield",
                    "priority": "low",
                    "category": "general"
                }
            ]
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available Groq models"""
        if not self.client:
            return {"error": "Groq client not available"}
        
        try:
            return {
                "current_model": "llama-3.3-70b-versatile",  # Updated to current model
                "status": "active",
                "features": ["chat", "completion", "agriculture_advisory"]
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}

    def validate_context_data(self, context_data: Dict) -> Dict[str, Any]:
        """Validate input context data"""
        validation_result = {"is_valid": True, "warnings": [], "errors": []}
        
        # Check required fields
        required_fields = ["crop_type"]
        for field in required_fields:
            if field not in context_data or not context_data[field]:
                validation_result["errors"].append(f"Missing required field: {field}")
        
        # Validate numeric fields
        numeric_fields = {
            "predicted_yield_per_hectare": (0, 200),
            "area_hectares": (0, 10000),
            "fertilizer_applied": (0, 1000),
            "pesticide_applied": (0, 100)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in context_data:
                try:
                    val = float(context_data[field])
                    if not min_val <= val <= max_val:
                        validation_result["warnings"].append(
                            f"{field}: {val} is outside expected range ({min_val}-{max_val})"
                        )
                except (ValueError, TypeError):
                    validation_result["warnings"].append(f"{field} should be numeric")
        
        validation_result["is_valid"] = len(validation_result["errors"]) == 0
        return validation_result

    def get_available_models(self) -> List[str]:
        """Get list of currently available Groq models"""
        return [
            "llama-3.3-70b-versatile",      # Primary recommendation (Production)
            "llama-3.1-8b-instant",        # Fast alternative (Production)
            "qwen/qwen3-32b",               # Advanced features (Preview)
            "gemma2-9b-it",                 # Google model (Production)
            "deepseek-r1-distill-llama-70b" # DeepSeek model (Preview)
        ]

    def switch_model(self, model_name: str) -> bool:
        """Switch to a different Groq model"""
        available_models = self.get_available_models()
        
        if model_name not in available_models:
            logger.error(f"Model {model_name} not available. Available models: {available_models}")
            return False
        
        self.current_model = model_name
        logger.info(f"Switched to model: {model_name}")
        return True

    def get_recommendations_with_model(self, context_data: Dict, model_name: str = None) -> Dict[str, List[Dict]]:
        """Generate recommendations with a specific model"""
        if not self.client:
            logger.warning("Groq client not available, using default recommendations")
            return self._default_recommendations()

        # Use specified model or default
        model_to_use = model_name if model_name in self.get_available_models() else "llama-3.3-70b-versatile"
        
        try:
            prompt = self._build_comprehensive_prompt(context_data)
            
            completion = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
                top_p=0.9,
                stream=False,
            )

            advice_text = completion.choices[0].message.content.strip()
            logger.info(f"Successfully received recommendations from {model_to_use}")
            
            return self._parse_structured_recommendations(advice_text, context_data)

        except Exception as e:
            logger.error(f"Groq API error with model {model_to_use}: {e}")
            return self._default_recommendations()


# Global service instance
groq_agriculture_advisor = GroqAgricultureAdvisor()

def get_farming_recommendations(context_data: Dict) -> Dict[str, List[Dict]]:
    """
    Public function to generate farming recommendations using Groq
    
    Args:
        context_data: Dictionary containing farm and crop data
        
    Returns:
        Dictionary with categorized recommendations
    """
    return groq_agriculture_advisor.get_recommendations(context_data)

def get_farming_recommendations_with_model(context_data: Dict, model_name: str = None) -> Dict[str, List[Dict]]:
    """
    Public function to generate farming recommendations using a specific Groq model
    
    Args:
        context_data: Dictionary containing farm and crop data
        model_name: Specific model to use (optional)
        
    Returns:
        Dictionary with categorized recommendations
    """
    return groq_agriculture_advisor.get_recommendations_with_model(context_data, model_name)

def validate_input_data(context_data: Dict) -> Dict[str, Any]:
    """
    Public function to validate input data
    
    Args:
        context_data: Dictionary containing farm and crop data
        
    Returns:
        Validation results with errors and warnings
    """
    return groq_agriculture_advisor.validate_context_data(context_data)

def get_available_models() -> List[str]:
    """
    Public function to get list of available Groq models
    
    Returns:
        List of available model names
    """
    return groq_agriculture_advisor.get_available_models()

# # Example usage and testing
# if __name__ == "__main__":
#     # Example context data
#     sample_context = {
#         "crop_type": "rice",
#         "predicted_yield_per_hectare": 45,
#         "area_hectares": 2.5,
#         "soil_parameters": {
#             "ph": 6.8,
#             "nitrogen": 0.45,
#             "organic_carbon": 1.2,
#             "sand_content": 35,
#             "clay_content": 25
#         },
#         "weather_summary": {
#             "total_rainfall": 850,
#             "avg_temp_max": 32,
#             "avg_temp_min": 22,
#             "avg_humidity": 75
#         },
#         "fertilizer_applied": 120,
#         "pesticide_applied": 5
#     }
    
#     # Test the updated system
#     print("Available models:", get_available_models())
#     print("\nValidation results:", validate_input_data(sample_context))
#     print("\nRecommendations:", get_farming_recommendations(sample_context))