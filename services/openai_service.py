import os
import logging
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not found. AI recommendations will be unavailable.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
    
    def generate_farming_advice(self, context_data: Dict) -> Dict[str, List[Dict]]:
        """Generate structured farming recommendations"""
        if not self.client:
            return self._get_default_recommendations()
        
        try:
            prompt = self._build_prompt(context_data)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert agricultural advisor specializing in precision farming for Indian conditions. 
                        Provide specific, actionable recommendations based on the given data. Format your response as a structured recommendation system."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            advice_text = response.choices[0].message.content.strip()
            
            # Parse the advice into structured format
            return self._parse_recommendations(advice_text, context_data)
            
        except Exception as e:
            logger.error(f"OpenAI API call error: {e}")
            return self._get_default_recommendations()
    
    def _build_prompt(self, context_data: Dict) -> str:
        """Build a comprehensive prompt for farming advice"""
        crop_type = context_data.get('crop_type', 'unknown')
        yield_pred = context_data.get('predicted_yield_per_hectare', 0)
        soil = context_data.get('soil_parameters', {})
        weather = context_data.get('weather_summary', {})
        
        prompt = f"""
Based on the following agricultural data, provide specific recommendations for {crop_type} cultivation:

CROP INFORMATION:
- Crop: {crop_type}
- Predicted yield: {yield_pred} quintals/hectare
- Area: {context_data.get('area_hectares', 'unknown')} hectares

SOIL CONDITIONS:
- pH: {soil.get('ph', 'unknown')}
- Nitrogen: {soil.get('nitrogen', 'unknown')} g/kg
- Organic Carbon: {soil.get('organic_carbon', 'unknown')}%
- Sand: {soil.get('sand_content', 'unknown')}%
- Clay: {soil.get('clay_content', 'unknown')}%

WEATHER PATTERN:
- Rainfall: {weather.get('total_rainfall', 'unknown')} mm
- Max Temp: {weather.get('avg_temp_max', 'unknown')}°C
- Min Temp: {weather.get('avg_temp_min', 'unknown')}°C
- Humidity: {weather.get('avg_humidity', 'unknown')}%

CURRENT INPUTS:
- Fertilizer: {context_data.get('fertilizer_applied', 0)} kg/hectare
- Pesticide: {context_data.get('pesticide_applied', 0)} kg/hectare

Provide specific recommendations for:
1. IRRIGATION: Water management, timing, and quantity
2. FERTILIZATION: NPK recommendations, organic matter, micronutrients
3. PEST CONTROL: Preventive measures, monitoring, treatment options
4. GENERAL: Planting, harvesting, soil improvement, market timing

Make recommendations practical for Indian farming conditions and specify priority levels (high/medium/low).
"""
        return prompt
    
    def _parse_recommendations(self, advice_text: str, context_data: Dict) -> Dict[str, List[Dict]]:
        """Parse AI advice into structured recommendations"""
        try:
            # Split into categories
            sections = {
                'irrigation': [],
                'fertilization': [],
                'pestControl': [],
                'general': []
            }
            
            # Simple parsing logic - this could be enhanced with NLP
            lines = advice_text.split('\n')
            current_category = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Identify category headers
                if any(keyword in line.lower() for keyword in ['irrigation', 'water']):
                    current_category = 'irrigation'
                elif any(keyword in line.lower() for keyword in ['fertil', 'npk', 'nutrient']):
                    current_category = 'fertilization'
                elif any(keyword in line.lower() for keyword in ['pest', 'disease', 'insect']):
                    current_category = 'pestControl'
                elif any(keyword in line.lower() for keyword in ['general', 'harvest', 'plant']):
                    current_category = 'general'
                elif line.startswith(('-', '•', '*')) or line[0].isdigit():
                    # This is a recommendation point
                    if current_category:
                        priority = self._determine_priority(line, context_data)
                        sections[current_category].append({
                            'action': self._clean_recommendation_text(line),
                            'details': self._add_context_details(line, context_data),
                            'priority': priority
                        })
            
            # Ensure each category has at least one recommendation
            for category in sections:
                if not sections[category]:
                    sections[category] = self._get_default_category_recommendations(category, context_data)
            
            return sections
            
        except Exception as e:
            logger.error(f"Error parsing recommendations: {e}")
            return self._get_default_recommendations()
    
    def _clean_recommendation_text(self, text: str) -> str:
        """Clean and format recommendation text"""
        # Remove bullet points and numbers
        text = text.lstrip('-•*0123456789. ')
        return text.strip()
    
    def _add_context_details(self, recommendation: str, context_data: Dict) -> str:
        """Add contextual details to recommendations"""
        # This is a simplified version - could be enhanced with more sophisticated matching
        if 'water' in recommendation.lower():
            rainfall = context_data.get('weather_summary', {}).get('total_rainfall', 0)
            if rainfall < 500:
                return "Given low rainfall conditions, frequent monitoring is essential."
            elif rainfall > 1500:
                return "High rainfall area - focus on drainage management."
        
        if 'fertil' in recommendation.lower():
            ph = context_data.get('soil_parameters', {}).get('ph', 7.0)
            if ph < 6.0:
                return "Soil pH is acidic - consider lime application before fertilization."
            elif ph > 8.0:
                return "Soil pH is alkaline - use acidifying fertilizers."
        
        return "Follow local agricultural guidelines for implementation."
    
    def _determine_priority(self, recommendation: str, context_data: Dict) -> str:
        """Determine priority level based on recommendation content and context"""
        high_priority_keywords = ['urgent', 'critical', 'immediately', 'essential', 'must']
        medium_priority_keywords = ['should', 'recommended', 'important', 'consider']
        
        rec_lower = recommendation.lower()
        
        if any(keyword in rec_lower for keyword in high_priority_keywords):
            return 'high'
        elif any(keyword in rec_lower for keyword in medium_priority_keywords):
            return 'medium'
        else:
            return 'low'
    
    def _get_default_category_recommendations(self, category: str, context_data: Dict) -> List[Dict]:
        """Get default recommendations for a category"""
        defaults = {
            'irrigation': [
                {
                    'action': 'Monitor soil moisture regularly',
                    'details': 'Check soil moisture at root depth before each irrigation',
                    'priority': 'medium'
                }
            ],
            'fertilization': [
                {
                    'action': 'Apply balanced NPK fertilizer',
                    'details': 'Use soil test results to determine optimal ratios',
                    'priority': 'medium'
                }
            ],
            'pestControl': [
                {
                    'action': 'Implement integrated pest management',
                    'details': 'Regular field monitoring for early pest detection',
                    'priority': 'medium'
                }
            ],
            'general': [
                {
                    'action': 'Follow good agricultural practices',
                    'details': 'Maintain proper crop spacing and field hygiene',
                    'priority': 'low'
                }
            ]
        }
        return defaults.get(category, [])
    
    def _get_default_recommendations(self) -> Dict[str, List[Dict]]:
        """Return default recommendations when AI service is unavailable"""
        logger.info("Using default farming recommendations")
        
        return {
            'irrigation': [
                {
                    'action': 'Water management based on crop stage',
                    'details': 'Adjust irrigation frequency according to plant growth phase',
                    'priority': 'high'
                },
                {
                    'action': 'Monitor soil moisture levels',
                    'details': 'Check moisture at 6-inch depth before watering',
                    'priority': 'medium'
                }
            ],
            'fertilization': [
                {
                    'action': 'Apply balanced NPK fertilizer',
                    'details': 'Use 4:2:1 NPK ratio as baseline, adjust based on soil test',
                    'priority': 'high'
                },
                {
                    'action': 'Add organic compost',
                    'details': 'Apply 2-3 tons per hectare to improve soil health',
                    'priority': 'medium'
                }
            ],
            'pestControl': [
                {
                    'action': 'Regular field inspection',
                    'details': 'Scout fields weekly for pest and disease symptoms',
                    'priority': 'high'
                },
                {
                    'action': 'Use integrated pest management',
                    'details': 'Combine biological, cultural, and chemical controls',
                    'priority': 'medium'
                }
            ],
            'general': [
                {
                    'action': 'Maintain proper plant spacing',
                    'details': 'Follow recommended spacing for optimal growth and air circulation',
                    'priority': 'medium'
                },
                {
                    'action': 'Plan harvest timing',
                    'details': 'Monitor crop maturity indicators for optimal harvest window',
                    'priority': 'low'
                }
            ]
        }

# Global service instance
openai_service = OpenAIService()

def generate_farming_advice(context_data: Dict) -> Dict[str, List[Dict]]:
    """Public function to generate farming recommendations"""
    return openai_service.generate_farming_advice(context_data)