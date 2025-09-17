import os
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class MarketTrendAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Market Trend Analyzer with Groq AI
        
        Args:
            api_key: Groq API key. If None, will try to get from environment
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            logger.warning("Groq API key not found. Market trend analysis will be unavailable.")
            self.client = None
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info("Market Trend Analyzer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.client = None

        # Crop price mapping (example prices in INR per quintal)
        self.crop_prices = {
            "rice": {"min": 1800, "max": 2500, "current": 2200},
            "wheat": {"min": 1900, "max": 2400, "current": 2150},
            "cotton": {"min": 5500, "max": 7500, "current": 6200},
            "maize": {"min": 1600, "max": 2200, "current": 1900},
            "sugarcane": {"min": 300, "max": 400, "current": 350},
            "groundnut": {"min": 4800, "max": 6500, "current": 5400},
            "soybean": {"min": 3800, "max": 5200, "current": 4500},
            "turmeric": {"min": 7500, "max": 12000, "current": 9500}
        }

    def analyze_market_trends(self, context_data: Dict) -> Dict[str, Any]:
        """Generate comprehensive market trend analysis"""
        if not self.client:
            logger.warning("Groq client not available, using default market analysis")
            return self._default_market_analysis(context_data)

        try:
            prompt = self._build_market_analysis_prompt(context_data)
            
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": self._get_market_system_prompt(),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
                top_p=0.9,
                stream=False,
            )

            analysis_text = completion.choices[0].message.content.strip()
            logger.info("Successfully received market analysis from Groq")
            
            return self._parse_market_analysis(analysis_text, context_data)

        except Exception as e:
            logger.error(f"Groq API error in market analysis: {e}")
            return self._default_market_analysis(context_data)

    def _get_market_system_prompt(self) -> str:
        """System prompt for market trend analysis"""
        return """You are Dr. MarketBot, an expert agricultural market analyst specializing in Indian crop markets and pricing trends.

Your expertise includes:
- Crop price analysis and forecasting
- Seasonal market trends and demand patterns
- Weather impact on crop markets
- Regional market variations across Indian states
- Supply chain and logistics considerations
- Government policy impacts on agriculture markets
- Export-import trends affecting domestic prices

IMPORTANT: Structure your response using these exact section headers:
### CURRENT MARKET STATUS
### MONTHLY TREND ANALYSIS
### TRENDING CROPS ANALYSIS
### FUTURE MARKET OUTLOOK
### PRICE RECOMMENDATIONS

For each section:
- Provide specific, data-driven insights
- Include price ranges and trend directions
- Consider seasonal factors and weather impacts
- Mention market timing recommendations
- Focus on actionable market intelligence
- Consider regional variations within India

Base your analysis on current Indian agricultural market conditions, seasonal patterns, and weather-related supply-demand dynamics."""

    def _build_market_analysis_prompt(self, context_data: Dict) -> str:
        """Build comprehensive market analysis prompt"""
        crop = context_data.get("crop_type", "unknown")
        state = context_data.get("state", "unknown")
        district = context_data.get("district", "unknown")
        predicted_yield = context_data.get("predicted_yield_per_hectare", 0)
        total_production = context_data.get("total_expected_production", 0)
        weather = context_data.get("weather_summary", {})
        current_month = datetime.now().strftime("%B")
        current_year = datetime.now().year

        prompt = f"""Please provide a comprehensive market trend analysis for the following agricultural scenario:

**CROP & LOCATION:**
- Primary Crop: {crop.title()}
- State: {state.title()}
- District: {district.title()}
- Predicted Production: {total_production} quintals
- Expected Yield: {predicted_yield} quintals/hectare

**TEMPORAL CONTEXT:**
- Current Month: {current_month} {current_year}
- Analysis Period: Next 12 months
- Harvest Season Consideration: Based on crop type and regional patterns

**WEATHER CONDITIONS:**
- Total Rainfall: {weather.get('total_rainfall', 'Unknown')} mm
- Temperature Range: {weather.get('avg_temp_min', 'Unknown')}°C - {weather.get('avg_temp_max', 'Unknown')}°C
- Humidity: {weather.get('avg_humidity', 'Unknown')}%
- Weather Impact: {self._assess_weather_market_impact(weather)}

**ANALYSIS REQUIREMENTS:**

1. **Current Market Status**: Analyze current prices and demand for {crop} in {state}
2. **Monthly Trends**: Provide month-wise price trends for the next 12 months
3. **Regional Competitive Crops**: Identify trending crops in {state} and {district}
4. **Future Outlook**: Forecast market conditions considering weather patterns
5. **Optimal Timing**: Recommend best selling periods and market strategies

**SPECIFIC FOCUS AREAS:**
- Price volatility and seasonal patterns
- Weather-induced supply disruptions
- Regional crop alternatives with better market potential
- Government policy impacts (MSP, subsidies)
- Export opportunities and domestic demand

Please provide actionable insights that help farmers make informed decisions about crop selection, timing, and marketing strategies."""

        return prompt

    def _assess_weather_market_impact(self, weather: Dict) -> str:
        """Assess weather impact on market conditions"""
        try:
            rainfall = float(weather.get('total_rainfall', 0))
            temp_max = float(weather.get('avg_temp_max', 0))
            
            impacts = []
            if rainfall < 400:
                impacts.append("Drought stress may reduce supply and increase prices")
            elif rainfall > 1200:
                impacts.append("Excess rainfall may affect quality and timing")
            
            if temp_max > 35:
                impacts.append("High temperatures may stress crops and affect yields")
            elif temp_max < 20:
                impacts.append("Cool weather may delay harvest and extend growing season")
                
            return "; ".join(impacts) if impacts else "Normal weather impact on markets"
        except (ValueError, TypeError):
            return "Weather impact assessment needed"

    def _parse_market_analysis(self, analysis_text: str, context_data: Dict) -> Dict[str, Any]:
        """Parse structured market analysis from AI response"""
        crop = context_data.get("crop_type", "unknown")
        
        # Initialize analysis structure
        market_analysis = {
            "current_status": {
                "price_range": self._get_price_info(crop),
                "demand_level": "moderate",
                "market_sentiment": "stable"
            },
            "monthly_trends": [],
            "trending_crops": {
                "current_trending": [],
                "future_trending": [],
                "declining_crops": []
            },
            "future_outlook": {
                "next_3_months": "",
                "next_6_months": "",
                "next_12_months": ""
            },
            "recommendations": {
                "optimal_selling_time": "",
                "alternative_crops": [],
                "market_strategies": []
            },
            "price_forecast": {
                "expected_trend": "stable",
                "risk_factors": [],
                "opportunities": []
            }
        }

        lines = analysis_text.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            if "###" in line:
                line_lower = line.lower()
                if "current market" in line_lower:
                    current_section = "current_status"
                elif "monthly trend" in line_lower:
                    current_section = "monthly_trends"
                elif "trending crops" in line_lower:
                    current_section = "trending_crops"
                elif "future" in line_lower or "outlook" in line_lower:
                    current_section = "future_outlook"
                elif "price" in line_lower or "recommendation" in line_lower:
                    current_section = "recommendations"
                continue
            
            # Parse content based on section
            if current_section and (line.startswith(("-", "•", "*")) or 
                                   (line[0].isdigit() and "." in line[:3])):
                
                cleaned_text = self._clean_text(line)
                if cleaned_text:
                    self._add_to_section(market_analysis, current_section, cleaned_text)

        # Generate monthly trends if not populated
        if not market_analysis["monthly_trends"]:
            market_analysis["monthly_trends"] = self._generate_monthly_trends(crop)

        # Add trending crops if not populated
        if not any(market_analysis["trending_crops"].values()):
            market_analysis["trending_crops"] = self._get_regional_trending_crops(
                context_data.get("state", ""), context_data.get("weather_summary", {})
            )

        return market_analysis

    def _add_to_section(self, analysis: Dict, section: str, content: str):
        """Add parsed content to appropriate section"""
        if section == "current_status":
            # Parse current market info
            if "price" in content.lower():
                analysis["current_status"]["current_info"] = content
            elif "demand" in content.lower():
                analysis["current_status"]["demand_info"] = content
                
        elif section == "monthly_trends":
            # Extract month and trend info
            month_trend = self._parse_monthly_trend(content)
            if month_trend:
                analysis["monthly_trends"].append(month_trend)
                
        elif section == "trending_crops":
            # Categorize trending crop information
            if "current" in content.lower() or "now" in content.lower():
                crop_name = self._extract_crop_name(content)
                if crop_name:
                    analysis["trending_crops"]["current_trending"].append({
                        "crop": crop_name,
                        "reason": content
                    })
            elif "future" in content.lower() or "upcoming" in content.lower():
                crop_name = self._extract_crop_name(content)
                if crop_name:
                    analysis["trending_crops"]["future_trending"].append({
                        "crop": crop_name,
                        "reason": content
                    })
                    
        elif section == "future_outlook":
            analysis["future_outlook"]["summary"] = content
            
        elif section == "recommendations":
            analysis["recommendations"]["market_strategies"].append(content)

    def _parse_monthly_trend(self, content: str) -> Optional[Dict]:
        """Parse monthly trend information"""
        months = ["january", "february", "march", "april", "may", "june",
                 "july", "august", "september", "october", "november", "december"]
        
        content_lower = content.lower()
        for month in months:
            if month in content_lower:
                trend = "stable"
                if any(word in content_lower for word in ["increase", "rise", "up", "higher"]):
                    trend = "increasing"
                elif any(word in content_lower for word in ["decrease", "fall", "down", "lower"]):
                    trend = "decreasing"
                
                return {
                    "month": month.title(),
                    "trend": trend,
                    "description": content
                }
        return None

    def _extract_crop_name(self, content: str) -> Optional[str]:
        """Extract crop name from content"""
        crops = ["rice", "wheat", "cotton", "maize", "sugarcane", "groundnut", "soybean", "turmeric"]
        content_lower = content.lower()
        
        for crop in crops:
            if crop in content_lower:
                return crop
        return None

    def _generate_monthly_trends(self, crop: str) -> List[Dict]:
        """Generate default monthly trends based on crop seasonality"""
        current_month = datetime.now().month
        trends = []
        
        # Basic seasonal patterns for major crops
        seasonal_patterns = {
            "rice": {
                "peak_months": [10, 11, 12],  # Post-harvest
                "low_months": [6, 7, 8]      # Pre-harvest
            },
            "wheat": {
                "peak_months": [4, 5, 6],
                "low_months": [11, 12, 1]
            },
            "cotton": {
                "peak_months": [1, 2, 3],
                "low_months": [8, 9, 10]
            }
        }
        
        pattern = seasonal_patterns.get(crop, {"peak_months": [4, 5], "low_months": [10, 11]})
        
        for i in range(12):
            month_num = ((current_month + i - 1) % 12) + 1
            month_name = datetime(2024, month_num, 1).strftime("%B")
            
            if month_num in pattern["peak_months"]:
                trend = "increasing"
                desc = f"Peak season for {crop} - higher demand and prices"
            elif month_num in pattern["low_months"]:
                trend = "decreasing"
                desc = f"Off-season for {crop} - lower prices expected"
            else:
                trend = "stable"
                desc = f"Moderate market conditions for {crop}"
            
            trends.append({
                "month": month_name,
                "trend": trend,
                "description": desc
            })
        
        return trends

    def _get_regional_trending_crops(self, state: str, weather: Dict) -> Dict[str, List]:
        """Get trending crops based on region and weather"""
        state_lower = state.lower()
        
        regional_trends = {
            "odisha": {
                "current": ["rice", "turmeric", "groundnut"],
                "future": ["cotton", "maize", "soybean"],
                "declining": ["traditional rice varieties"]
            },
            "andhra pradesh": {
                "current": ["cotton", "rice", "groundnut"],
                "future": ["soybean", "maize", "turmeric"],
                "declining": ["traditional cotton varieties"]
            },
            "telangana": {
                "current": ["cotton", "maize", "rice"],
                "future": ["soybean", "groundnut", "turmeric"],
                "declining": ["traditional crops"]
            }
        }
        
        base_trends = regional_trends.get(state_lower, {
            "current": ["rice", "wheat"],
            "future": ["cotton", "soybean"],
            "declining": ["traditional varieties"]
        })
        
        # Adjust based on weather conditions
        rainfall = float(weather.get('total_rainfall', 500))
        if rainfall > 1000:  # High rainfall
            base_trends["future"].append("rice")
            base_trends["declining"].append("drought-resistant crops")
        elif rainfall < 400:  # Low rainfall
            base_trends["future"].extend(["cotton", "groundnut"])
            base_trends["declining"].append("water-intensive crops")
        
        return {
            "current_trending": [{"crop": crop, "reason": f"High demand in {state}"} 
                               for crop in base_trends["current"]],
            "future_trending": [{"crop": crop, "reason": f"Growing market potential in {state}"} 
                              for crop in base_trends["future"]],
            "declining_crops": [{"crop": crop, "reason": "Decreasing market demand"} 
                              for crop in base_trends["declining"]]
        }

    def _get_price_info(self, crop: str) -> Dict[str, float]:
        """Get current price information for crop"""
        return self.crop_prices.get(crop, {"min": 1500, "max": 3000, "current": 2000})

    def _clean_text(self, text: str) -> str:
        """Clean and format text"""
        return text.lstrip("-•*0123456789. ").strip()

    def _default_market_analysis(self, context_data: Dict) -> Dict[str, Any]:
        """Default market analysis when Groq is unavailable"""
        crop = context_data.get("crop_type", "rice")
        state = context_data.get("state", "unknown")
        
        return {
            "current_status": {
                "price_range": self._get_price_info(crop),
                "demand_level": "moderate",
                "market_sentiment": "stable",
                "current_info": f"Current {crop} prices are within normal seasonal range"
            },
            "monthly_trends": self._generate_monthly_trends(crop),
            "trending_crops": self._get_regional_trending_crops(state, context_data.get("weather_summary", {})),
            "future_outlook": {
                "next_3_months": f"{crop.title()} market expected to remain stable with seasonal variations",
                "next_6_months": "Market conditions depend on monsoon and government policies",
                "next_12_months": "Long-term outlook positive with growing domestic demand"
            },
            "recommendations": {
                "optimal_selling_time": "Post-harvest season for best prices",
                "alternative_crops": ["soybean", "cotton", "maize"],
                "market_strategies": [
                    "Monitor daily market prices",
                    "Consider contract farming opportunities",
                    "Store produce for better prices if possible"
                ]
            },
            "price_forecast": {
                "expected_trend": "stable",
                "risk_factors": ["Weather variations", "Government policy changes"],
                "opportunities": ["Export demand", "Processing industry growth"]
            }
        }

    def get_crop_comparison(self, primary_crop: str, alternative_crops: List[str], context_data: Dict) -> Dict[str, Any]:
        """Compare market potential of different crops"""
        comparison = {
            "primary_crop": {
                "name": primary_crop,
                "price_info": self._get_price_info(primary_crop),
                "market_score": self._calculate_market_score(primary_crop, context_data)
            },
            "alternatives": []
        }
        
        for crop in alternative_crops:
            comparison["alternatives"].append({
                "name": crop,
                "price_info": self._get_price_info(crop),
                "market_score": self._calculate_market_score(crop, context_data),
                "recommendation": self._get_crop_recommendation(crop, context_data)
            })
        
        return comparison

    def _calculate_market_score(self, crop: str, context_data: Dict) -> float:
        """Calculate market attractiveness score (0-10)"""
        base_score = 5.0
        
        # Adjust based on current prices
        price_info = self._get_price_info(crop)
        if price_info["current"] > (price_info["min"] + price_info["max"]) / 2:
            base_score += 1.0
        
        # Adjust based on weather suitability
        weather = context_data.get("weather_summary", {})
        rainfall = float(weather.get('total_rainfall', 500))
        
        # Crop-specific weather preferences
        if crop in ["rice"] and rainfall > 800:
            base_score += 1.0
        elif crop in ["cotton", "groundnut"] and 400 < rainfall < 800:
            base_score += 1.0
        elif rainfall < 300:
            base_score -= 1.0
        
        return min(10.0, max(1.0, base_score))

    def _get_crop_recommendation(self, crop: str, context_data: Dict) -> str:
        """Get specific recommendation for crop"""
        weather = context_data.get("weather_summary", {})
        state = context_data.get("state", "").lower()
        
        if crop == "cotton" and state in ["andhra pradesh", "telangana"]:
            return "Highly suitable - strong regional market and infrastructure"
        elif crop == "rice" and float(weather.get('total_rainfall', 0)) > 800:
            return "Good choice - adequate water availability"
        elif crop == "soybean":
            return "Emerging market - good for diversification"
        else:
            return "Consider based on local conditions and market access"


# Global service instance
market_trend_analyzer = MarketTrendAnalyzer()

def get_market_trends(context_data: Dict) -> Dict[str, Any]:
    """
    Public function to get market trend analysis
    
    Args:
        context_data: Dictionary containing crop and location data
        
    Returns:
        Dictionary with comprehensive market analysis
    """
    return market_trend_analyzer.analyze_market_trends(context_data)

def get_crop_market_comparison(primary_crop: str, alternatives: List[str], context_data: Dict) -> Dict[str, Any]:
    """
    Public function to compare market potential of different crops
    
    Args:
        primary_crop: Main crop being grown
        alternatives: List of alternative crops to consider
        context_data: Dictionary containing farm and location data
        
    Returns:
        Dictionary with crop comparison analysis
    """
    return market_trend_analyzer.get_crop_comparison(primary_crop, alternatives, context_data)

def validate_market_data(context_data: Dict) -> Dict[str, Any]:
    """
    Public function to validate market analysis input data
    
    Args:
        context_data: Dictionary containing input data
        
    Returns:
        Validation results
    """
    validation_result = {"is_valid": True, "warnings": [], "errors": []}
    
    # Check required fields for market analysis
    required_fields = ["crop_type", "state"]
    for field in required_fields:
        if field not in context_data or not context_data[field]:
            validation_result["errors"].append(f"Missing required field for market analysis: {field}")
    
    validation_result["is_valid"] = len(validation_result["errors"]) == 0
    return validation_result

# # Example usage
# if __name__ == "__main__":
#     sample_context = {
#         "crop_type": "cotton",
#         "state": "andhra pradesh",
#         "district": "guntur",
#         "predicted_yield_per_hectare": 15,
#         "total_expected_production": 37.5,
#         "weather_summary": {
#             "total_rainfall": 650,
#             "avg_temp_max": 34,
#             "avg_temp_min": 22,
#             "avg_humidity": 70
#         }
#     }
    
#     print("Market Analysis:", get_market_trends(sample_context))
#     print("\nCrop Comparison:", get_crop_market_comparison("cotton", ["soybean", "maize", "groundnut"], sample_context))