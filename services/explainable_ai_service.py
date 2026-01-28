import logging
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ExplainableAIService:
    """
    Service to provide explanations for AI predictions and recommendations.
    Implements XAI principles to help farmers understand WHY predictions are made.
    """
    
    def __init__(self):
        # Feature importance weights based on agricultural research
        self.feature_weights = {
            'soil_ph': 0.12,
            'nitrogen': 0.15,
            'organic_carbon': 0.10,
            'total_rainfall': 0.18,
            'avg_temp_max': 0.12,
            'avg_temp_min': 0.08,
            'fertilizer': 0.10,
            'pesticide': 0.05,
            'elevation': 0.05,
            'avg_humidity': 0.05
        }
        
        # Optimal ranges for different crops (research-based)
        self.optimal_ranges = {
            'rice': {
                'ph': (5.5, 7.0),
                'nitrogen': (4.0, 8.0),
                'total_rainfall': (1000, 2000),
                'avg_temp_max': (28, 35),
                'fertilizer': (80, 150)
            },
            'wheat': {
                'ph': (6.0, 7.5),
                'nitrogen': (5.0, 9.0),
                'total_rainfall': (450, 650),
                'avg_temp_max': (20, 28),
                'fertilizer': (100, 180)
            },
            'cotton': {
                'ph': (6.0, 7.5),
                'nitrogen': (4.5, 8.0),
                'total_rainfall': (500, 1000),
                'avg_temp_max': (25, 35),
                'fertilizer': (60, 120)
            },
            'maize': {
                'ph': (5.8, 7.0),
                'nitrogen': (5.0, 9.0),
                'total_rainfall': (500, 800),
                'avg_temp_max': (25, 32),
                'fertilizer': (100, 160)
            },
            'soybean': {
                'ph': (6.0, 7.0),
                'nitrogen': (3.0, 6.0),
                'total_rainfall': (450, 700),
                'avg_temp_max': (25, 30),
                'fertilizer': (20, 40)
            },
            'groundnut': {
                'ph': (6.0, 7.0),
                'nitrogen': (3.5, 6.5),
                'total_rainfall': (500, 1000),
                'avg_temp_max': (28, 33),
                'fertilizer': (20, 40)
            }
        }
    
    def explain_yield_prediction(
        self, 
        predicted_yield: float,
        input_features: Dict[str, float],
        crop_type: str
    ) -> Dict[str, Any]:
        """
        Provide detailed explanation for yield prediction.
        
        Returns:
            Dictionary with explanation components including:
            - Feature contributions
            - Limiting factors
            - Optimization opportunities
            - Confidence factors
        """
        try:
            # Calculate feature contributions
            contributions = self._calculate_feature_contributions(input_features, crop_type)
            
            # Identify limiting factors
            limiting_factors = self._identify_limiting_factors(input_features, crop_type)
            
            # Find optimization opportunities
            optimization_opps = self._find_optimization_opportunities(input_features, crop_type)
            
            # Calculate confidence factors
            confidence_analysis = self._analyze_confidence(input_features, crop_type)
            
            # Generate human-readable explanation
            explanation_text = self._generate_explanation_text(
                predicted_yield, contributions, limiting_factors, crop_type
            )
            
            return {
                "predicted_yield": predicted_yield,
                "explanation_summary": explanation_text,
                "feature_contributions": contributions,
                "limiting_factors": limiting_factors,
                "optimization_opportunities": optimization_opps,
                "confidence_analysis": confidence_analysis,
                "recommendation_basis": self._generate_recommendation_basis(
                    input_features, limiting_factors, optimization_opps
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating yield explanation: {e}")
            return {
                "error": "Unable to generate detailed explanation",
                "basic_explanation": f"Predicted yield of {predicted_yield} quintals/hectare based on environmental analysis"
            }
    
    def _calculate_feature_contributions(
        self, 
        features: Dict[str, float], 
        crop_type: str
    ) -> List[Dict[str, Any]]:
        """Calculate how each feature contributes to the prediction"""
        contributions = []
        optimal = self.optimal_ranges.get(crop_type, {})
        
        for feature, value in features.items():
            if feature not in self.feature_weights:
                continue
            
            weight = self.feature_weights[feature]
            optimal_range = optimal.get(feature)
            
            if optimal_range:
                min_val, max_val = optimal_range
                mid_point = (min_val + max_val) / 2
                
                # Calculate how close the value is to optimal
                if min_val <= value <= max_val:
                    optimality_score = 1.0 - abs(value - mid_point) / (max_val - min_val)
                    status = "optimal"
                elif value < min_val:
                    optimality_score = max(0, 1.0 - (min_val - value) / min_val)
                    status = "below_optimal"
                else:
                    optimality_score = max(0, 1.0 - (value - max_val) / max_val)
                    status = "above_optimal"
                
                contribution = weight * optimality_score
                
                contributions.append({
                    "feature": feature,
                    "feature_name": self._get_feature_display_name(feature),
                    "value": round(value, 2),
                    "optimal_range": f"{min_val} - {max_val}",
                    "status": status,
                    "contribution_score": round(contribution * 100, 1),
                    "impact": "positive" if contribution > weight * 0.7 else "neutral" if contribution > weight * 0.4 else "negative",
                    "explanation": self._explain_feature_impact(feature, value, optimal_range, status)
                })
        
        # Sort by contribution score
        contributions.sort(key=lambda x: x['contribution_score'], reverse=True)
        return contributions
    
    def _identify_limiting_factors(
        self, 
        features: Dict[str, float], 
        crop_type: str
    ) -> List[Dict[str, Any]]:
        """Identify factors that are limiting yield potential"""
        limiting_factors = []
        optimal = self.optimal_ranges.get(crop_type, {})
        
        for feature, value in features.items():
            if feature not in optimal:
                continue
            
            min_val, max_val = optimal[feature]
            
            if value < min_val:
                severity = min(1.0, (min_val - value) / min_val)
                limiting_factors.append({
                    "factor": self._get_feature_display_name(feature),
                    "current_value": round(value, 2),
                    "optimal_range": f"{min_val} - {max_val}",
                    "severity": "high" if severity > 0.5 else "medium" if severity > 0.2 else "low",
                    "impact_on_yield": f"Reducing potential yield by approximately {int(severity * 20)}%",
                    "recommendation": self._get_correction_recommendation(feature, value, min_val, max_val, "low")
                })
            elif value > max_val:
                severity = min(1.0, (value - max_val) / max_val)
                limiting_factors.append({
                    "factor": self._get_feature_display_name(feature),
                    "current_value": round(value, 2),
                    "optimal_range": f"{min_val} - {max_val}",
                    "severity": "high" if severity > 0.5 else "medium" if severity > 0.2 else "low",
                    "impact_on_yield": f"May reduce yield by approximately {int(severity * 15)}%",
                    "recommendation": self._get_correction_recommendation(feature, value, min_val, max_val, "high")
                })
        
        # Sort by severity
        severity_order = {"high": 3, "medium": 2, "low": 1}
        limiting_factors.sort(key=lambda x: severity_order[x['severity']], reverse=True)
        
        return limiting_factors
    
    def _find_optimization_opportunities(
        self, 
        features: Dict[str, float], 
        crop_type: str
    ) -> List[Dict[str, Any]]:
        """Find opportunities to optimize yield"""
        opportunities = []
        optimal = self.optimal_ranges.get(crop_type, {})
        
        for feature, value in features.items():
            if feature not in optimal:
                continue
            
            min_val, max_val = optimal[feature]
            mid_point = (min_val + max_val) / 2
            
            # Check if value is within optimal range but could be improved
            if min_val <= value <= max_val:
                deviation = abs(value - mid_point) / (max_val - min_val)
                
                if deviation > 0.3:  # More than 30% away from midpoint
                    potential_gain = int(deviation * 10)  # Rough estimate
                    
                    opportunities.append({
                        "area": self._get_feature_display_name(feature),
                        "current_value": round(value, 2),
                        "optimal_target": round(mid_point, 2),
                        "potential_yield_gain": f"{potential_gain}%",
                        "priority": "high" if potential_gain > 5 else "medium",
                        "action": self._get_optimization_action(feature, value, mid_point)
                    })
        
        # Add input-based opportunities
        fertilizer = features.get('fertilizer', 0)
        optimal_fert = optimal.get('fertilizer', (0, 200))
        
        if fertilizer < optimal_fert[0] * 0.8:
            opportunities.append({
                "area": "Fertilizer Application",
                "current_value": round(fertilizer, 2),
                "optimal_target": f"{optimal_fert[0]} - {optimal_fert[1]} kg/ha",
                "potential_yield_gain": "15-25%",
                "priority": "high",
                "action": f"Increase fertilizer application by {int(optimal_fert[0] * 0.8 - fertilizer)} kg/ha"
            })
        
        return opportunities
    
    def _analyze_confidence(
        self, 
        features: Dict[str, float], 
        crop_type: str
    ) -> Dict[str, Any]:
        """Analyze confidence in the prediction"""
        optimal = self.optimal_ranges.get(crop_type, {})
        
        # Calculate how many features are in optimal range
        in_optimal = 0
        total_checked = 0
        
        for feature, value in features.items():
            if feature in optimal:
                min_val, max_val = optimal[feature]
                if min_val <= value <= max_val:
                    in_optimal += 1
                total_checked += 1
        
        confidence_score = (in_optimal / total_checked * 100) if total_checked > 0 else 50
        
        # Determine confidence level
        if confidence_score >= 80:
            level = "high"
            explanation = "Most environmental factors are optimal for this crop"
        elif confidence_score >= 60:
            level = "moderate"
            explanation = "Many factors are suitable, but some optimization possible"
        else:
            level = "low"
            explanation = "Several factors are outside optimal range, prediction has higher uncertainty"
        
        return {
            "confidence_score": round(confidence_score, 1),
            "confidence_level": level,
            "explanation": explanation,
            "factors_in_optimal_range": in_optimal,
            "total_factors_analyzed": total_checked
        }
    
    def _generate_explanation_text(
        self, 
        yield_val: float, 
        contributions: List[Dict], 
        limiting_factors: List[Dict],
        crop_type: str
    ) -> str:
        """Generate human-readable explanation"""
        explanation = f"The predicted yield of {yield_val:.1f} quintals/hectare for {crop_type} is based on:\n\n"
        
        # Top positive contributors
        top_positive = [c for c in contributions if c['impact'] == 'positive'][:3]
        if top_positive:
            explanation += "✓ POSITIVE FACTORS:\n"
            for c in top_positive:
                explanation += f"  • {c['feature_name']}: {c['explanation']}\n"
        
        # Limiting factors
        if limiting_factors:
            explanation += "\n⚠ LIMITING FACTORS:\n"
            for lf in limiting_factors[:3]:
                explanation += f"  • {lf['factor']}: {lf['impact_on_yield']}\n"
        
        return explanation
    
    def _generate_recommendation_basis(
        self,
        features: Dict[str, float],
        limiting_factors: List[Dict],
        optimization_opps: List[Dict]
    ) -> Dict[str, str]:
        """Generate basis for recommendations"""
        basis = {}
        
        # Irrigation basis
        rainfall = features.get('total_rainfall', 1000)
        if rainfall < 500:
            basis['irrigation'] = f"Low rainfall ({rainfall}mm) requires careful irrigation management to prevent water stress"
        elif rainfall > 1500:
            basis['irrigation'] = f"High rainfall ({rainfall}mm) may cause waterlogging; focus on drainage"
        else:
            basis['irrigation'] = f"Moderate rainfall ({rainfall}mm) allows for supplemental irrigation as needed"
        
        # Fertilization basis
        nitrogen = features.get('nitrogen', 5.0)
        fertilizer = features.get('fertilizer', 0)
        if nitrogen < 4.0 or fertilizer < 50:
            basis['fertilization'] = f"Soil nitrogen ({nitrogen}g/kg) and fertilizer application ({fertilizer}kg/ha) are below optimal; balanced NPK needed"
        else:
            basis['fertilization'] = f"Current fertilizer levels ({fertilizer}kg/ha) are adequate; focus on timing and split application"
        
        # Pest control basis
        humidity = features.get('avg_humidity', 70)
        if humidity > 80:
            basis['pest_control'] = f"High humidity ({humidity}%) increases disease pressure; preventive measures essential"
        else:
            basis['pest_control'] = "Regular monitoring recommended based on current conditions"
        
        return basis
    
    def _get_feature_display_name(self, feature: str) -> str:
        """Convert feature name to display name"""
        names = {
            'ph': 'Soil pH',
            'nitrogen': 'Soil Nitrogen',
            'organic_carbon': 'Organic Carbon',
            'total_rainfall': 'Rainfall',
            'avg_temp_max': 'Maximum Temperature',
            'avg_temp_min': 'Minimum Temperature',
            'fertilizer': 'Fertilizer Application',
            'pesticide': 'Pesticide Application',
            'elevation': 'Elevation',
            'avg_humidity': 'Humidity'
        }
        return names.get(feature, feature.replace('_', ' ').title())
    
    def _explain_feature_impact(
        self, 
        feature: str, 
        value: float, 
        optimal_range: Tuple[float, float],
        status: str
    ) -> str:
        """Explain how a feature impacts yield"""
        min_val, max_val = optimal_range
        
        if status == "optimal":
            return f"Value ({value:.1f}) is within optimal range, supporting good yield"
        elif status == "below_optimal":
            return f"Value ({value:.1f}) is below optimal minimum ({min_val}), limiting yield potential"
        else:
            return f"Value ({value:.1f}) exceeds optimal maximum ({max_val}), may stress the crop"
    
    def _get_correction_recommendation(
        self,
        feature: str,
        value: float,
        min_val: float,
        max_val: float,
        condition: str
    ) -> str:
        """Get recommendation to correct limiting factor"""
        recommendations = {
            'ph': {
                'low': f"Apply lime to raise pH from {value:.1f} to {min_val}-{max_val}",
                'high': f"Apply sulfur or organic matter to lower pH from {value:.1f} to {min_val}-{max_val}"
            },
            'nitrogen': {
                'low': f"Apply nitrogen-rich fertilizer to increase from {value:.1f} to {min_val}+ g/kg",
                'high': f"Reduce nitrogen application; current level {value:.1f} may cause lodging"
            },
            'total_rainfall': {
                'low': f"Implement supplemental irrigation; current rainfall {value:.0f}mm below {min_val}mm",
                'high': f"Ensure proper drainage; rainfall {value:.0f}mm exceeds optimal {max_val}mm"
            },
            'fertilizer': {
                'low': f"Increase fertilizer by {int(min_val - value)} kg/ha for optimal nutrition",
                'high': f"Reduce fertilizer by {int(value - max_val)} kg/ha to prevent nutrient burn"
            }
        }
        
        return recommendations.get(feature, {}).get(condition, f"Adjust to {min_val}-{max_val} range")
    
    def _get_optimization_action(
        self,
        feature: str,
        current: float,
        target: float
    ) -> str:
        """Get specific action to optimize feature"""
        diff = target - current
        
        actions = {
            'ph': f"Adjust pH by {abs(diff):.1f} units using {'lime' if diff > 0 else 'sulfur'}",
            'nitrogen': f"{'Increase' if diff > 0 else 'Decrease'} nitrogen by {abs(diff):.1f} g/kg through fertilization",
            'fertilizer': f"{'Increase' if diff > 0 else 'Decrease'} fertilizer by {abs(diff):.0f} kg/ha",
            'total_rainfall': f"Provide {abs(diff):.0f}mm {'supplemental irrigation' if diff > 0 else 'improved drainage'}"
        }
        
        return actions.get(feature, f"Adjust to target value of {target:.1f}")
    
    def explain_market_recommendation(
        self,
        crop: str,
        market_score: float,
        price_trend: str,
        reasons: List[str]
    ) -> Dict[str, Any]:
        """Explain why a crop is recommended based on market analysis"""
        
        explanation = {
            "crop": crop,
            "recommendation_strength": "strong" if market_score >= 7 else "moderate" if market_score >= 5 else "weak",
            "market_score": market_score,
            "price_trend": price_trend,
            "explanation_summary": "",
            "supporting_factors": [],
            "risk_factors": [],
            "timing_recommendation": ""
        }
        
        # Generate explanation summary
        if market_score >= 7:
            explanation["explanation_summary"] = f"{crop.title()} is strongly recommended due to favorable market conditions and price trends"
        elif market_score >= 5:
            explanation["explanation_summary"] = f"{crop.title()} has moderate market potential with some favorable factors"
        else:
            explanation["explanation_summary"] = f"{crop.title()} has limited market appeal currently; consider alternatives"
        
        # Categorize reasons
        for reason in reasons:
            reason_lower = reason.lower()
            if any(word in reason_lower for word in ['high', 'strong', 'growing', 'increasing', 'favorable']):
                explanation["supporting_factors"].append(reason)
            elif any(word in reason_lower for word in ['low', 'weak', 'declining', 'risk', 'volatile']):
                explanation["risk_factors"].append(reason)
        
        # Timing recommendation
        if price_trend.lower() in ['increasing', 'rising']:
            explanation["timing_recommendation"] = "Current market timing is favorable for cultivation"
        elif price_trend.lower() in ['decreasing', 'falling']:
            explanation["timing_recommendation"] = "Consider delaying or storing produce for better prices"
        else:
            explanation["timing_recommendation"] = "Market is stable; standard cultivation timing recommended"
        
        return explanation


# Global service instance
explainable_ai_service = ExplainableAIService()

def explain_prediction(
    predicted_yield: float,
    input_features: Dict[str, float],
    crop_type: str
) -> Dict[str, Any]:
    """
    Public function to get explanation for yield prediction
    
    Args:
        predicted_yield: The predicted yield value
        input_features: Dictionary of input features used for prediction
        crop_type: Type of crop
        
    Returns:
        Comprehensive explanation of the prediction
    """
    return explainable_ai_service.explain_yield_prediction(
        predicted_yield, input_features, crop_type
    )

def explain_market_recommendation(
    crop: str,
    market_score: float,
    price_trend: str,
    reasons: List[str]
) -> Dict[str, Any]:
    """
    Public function to explain market-based crop recommendations
    
    Args:
        crop: Crop name
        market_score: Market attractiveness score (0-10)
        price_trend: Current price trend
        reasons: List of reasons for recommendation
        
    Returns:
        Explanation of why crop is recommended
    """
    return explainable_ai_service.explain_market_recommendation(
        crop, market_score, price_trend, reasons
    )
