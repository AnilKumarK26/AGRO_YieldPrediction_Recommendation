# YieldWise AI: Crop Yield Prediction & Market Advisor

YieldWise AI is a full-stack platform that predicts crop yield from real-world environmental signals and provides AI-powered farming recommendations plus market trend analysis tailored for Indian agricultural conditions. It combines a machine learning model with data from SoilGrids, Open‑Elevation, and Open‑Meteo, and augments results with Groq LLM insights for agronomy and pricing strategies. A simple web UI is included.

## Key Features
- **Yield prediction**: ML model (RandomForest/GBR) predicts `quintals/hectare` using soil, weather, elevation, and farm inputs.
- **Environmental data ingestion**: Fetches soil (SoilGrids), elevation (Open‑Elevation), and historical weather (Open‑Meteo) with MongoDB caching.
- **AI farming advice**: Contextual recommendations for irrigation, fertilization, pest control, and general practices via Groq.
- **Market trend analysis**: Current status, monthly price trends, future outlook, and strategies with alternative crop comparison.
- **Auto setup**: On startup, auto-loads dataset and trains/loads model when possible.
- **Web UI**: Tailwind-based UI for form input and results at `/`.

## Architecture
- **Backend**: Flask app in [app.py](app.py) exposing REST endpoints.
- **Data services**:
  - Soil: [services/soil_service.py](services/soil_service.py)
  - Elevation: [services/elevation_service.py](services/elevation_service.py)
  - Weather: [services/weather_service.py](services/weather_service.py)
  - Groq agronomy: [services/groq_agriculture_advisor.py](services/groq_agriculture_advisor.py)
  - Market analysis: [services/market_trend_service.py](services/market_trend_service.py)
- **Model**: [models/yield_model.py](models/yield_model.py) (stored to [models/crop_yield_model.joblib](models/crop_yield_model.joblib) after training)
- **Data loader**: [data_loader.py](data_loader.py) for ingesting Kaggle CSV into MongoDB with environmental enrichment
- **DB**: MongoDB connection & indexes in [db/mongo_connector.py](db/mongo_connector.py)
- **Config**: API URLs, defaults, and env in [config.py](config.py)
- **Frontend**: Single page in [templates/index.html](templates/index.html)

Collections used in MongoDB:
- `historical_data`, `predictions`, `farmers`
- caches: `soil_cache`, `elevation_cache`, `weather_cache`

## Prerequisites
- Python 3.10+ recommended
- MongoDB (local or hosted; default URI `mongodb://localhost:27017/`)
- Internet access for external APIs (SoilGrids, Open‑Elevation, Open‑Meteo)
- Groq API key for AI features

## Quick Start
1. Clone and enter the project folder:
   ```bash
   git clone https://github.com/AnilKumarK26/AGRO_YieldPrediction_Recommendation.git
   cd AGRO_YieldPrediction_Recommendation/AGRO
   ```
2. Create a virtual environment (Windows PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set environment variables (create `.env` in the project root):
   ```env
   # MongoDB
   MONGO_URI=mongodb://localhost:27017/

   # Groq for AI recommendations and market analysis
   GROQ_API_KEY=your_groq_api_key_here

   # Optional: legacy OpenAI support (not used by main app)
   OPENAI_API_KEY=your_openai_api_key_here
   ```
5. Ensure MongoDB is running locally or point `MONGO_URI` to your cluster.
6. Start the server:
   ```bash
   python app.py
   ```
7. Open http://localhost:5000 to use the UI.

## Configuration
Managed via environment variables loaded in [config.py](config.py):
- `MONGO_URI`: Mongo connection string
- `GROQ_API_KEY`: Groq API key (required for AI features)
- `OPENAI_API_KEY`: optional legacy support in [services/openai_service.py](services/openai_service.py)

API endpoints used by services:
- SoilGrids: `https://rest.isric.org/soilgrids/v2.0/properties/query`
- Open‑Elevation: `https://api.open-elevation.com/api/v1/lookup`
- Open‑Meteo archive: `https://archive-api.open-meteo.com/v1/era5`
- Open‑Meteo forecast: `https://api.open-meteo.com/v1/forecast`

Defaults and fallbacks: see `DEFAULT_VALUES` in [config.py](config.py).

## Data & Model
- Dataset: place CSV (e.g., [crop_yield_dataset.csv](crop_yield_dataset.csv)) in the root. On startup, auto-setup attempts to load and train.
- Loader: [data_loader.py](data_loader.py) enriches Kaggle records with soil, elevation, and weather, and writes to `historical_data`.
- Features used by the model: soil (`ph`, `nitrogen`, `organic_carbon`, `sand_content`, `clay_content`, `silt_content`, `cec`, `bulk_density`), weather (`avg_temp_max`, `avg_temp_min`, `total_rainfall`, `avg_humidity`, `avg_wind_speed`), location (`elevation`, `latitude`, `longitude`), management (`area_hectares`, `fertilizer`, `pesticide`).
- Model training: `RandomForestRegressor` and `GradientBoostingRegressor` are evaluated; best MAE model is selected. Metrics are stored; model saved to [models/crop_yield_model.joblib](models/crop_yield_model.joblib).

## REST API
Base URL: `http://localhost:5000`

- **GET /**
  - Serves the web UI in [templates/index.html](templates/index.html).

- **POST /api/load-data**
  - Body: `{ "csv_path": "crop_yield_dataset.csv" }`
  - Loads CSV, enriches, and stores to MongoDB. Returns stats.

- **POST /api/train-model**
  - Trains the ML model using `historical_data` and saves it.

- **POST /api/predict**
  - Request JSON (required keys):
    ```json
    {
      "crop_type": "rice",
      "state": "odisha",
      "district": "bhubaneswar",
      "latitude": 20.2961,
      "longitude": 85.8245,
      "area_hectares": 2.5,
      "fertilizer": 50,
      "pesticide": 5,
      "crop_year": 2025
    }
    ```
  - Response JSON (abridged):
    ```json
    {
      "success": true,
      "predicted_yield_per_hectare": 42.5,
      "total_expected_production": 106.25,
      "unit": "quintals",
      "environmental_data": { "soil_health": {"ph": 6.8, ...}, "weather_summary": {"total_rainfall": 850, ...}, "elevation": 64 },
      "recommendations": { "irrigation": [ ... ], "fertilization": [ ... ], "pestControl": [ ... ], "general": [ ... ] },
      "market_analysis": { "current_status": { "price_range": {"current": 2200}, ... }, "monthly_trends": [ ... ] },
      "crop_comparison": { "primary_crop": { ... }, "alternatives": [ ... ] },
      "prediction_timestamp": "...",
      "ai_service": "groq"
    }
    ```

- **POST /api/market-analysis**
  - Analyze market for a crop/state, optionally compare alternatives.
  - Body: `{ "crop_type": "cotton", "state": "andhra pradesh", "alternative_crops": ["soybean","maize","groundnut"] }`

- **POST /api/test-ai**
  - Validate inputs and get farming recommendations from Groq independently.

- **GET /api/status**
  - System status: data counts, model loaded, AI services available.

- **GET /api/health**
  - Health check: DB, model, AI services.

## Running the Web UI
- Navigate to `/` to open [templates/index.html](templates/index.html)
- Fill in the form, submit, and see: yield, total production, estimated value, AI advice, market trends, and alternative crop comparison.

## Development Tips
- SoilGrids rate limits: service applies internal delays; repeated calls are cached in MongoDB.
- Weather and elevation calls are cached similarly.
- Auto-setup: On app start, attempts to find a CSV in root and train/load the model. See `auto_setup_system()` in [app.py](app.py).

## Deployment Notes
- Set `debug=False` when deploying Flask.
- Use a production WSGI server (e.g., Gunicorn or Waitress) behind Nginx/Apache.
- Secure environment variables and external API keys.
- Ensure MongoDB indexes are created (see [db/mongo_connector.py](db/mongo_connector.py)).

## Troubleshooting
- **MongoDB connection failed**: Verify `MONGO_URI` and that the server is reachable.
- **Groq AI unavailable**: Ensure `GROQ_API_KEY` is set; the app falls back to default recommendations.
- **Soil/Weather API timeouts**: Services will return defaults; try again later. Check network and rate limits.
- **Model not ready (503 on /api/predict)**: Load data with `/api/load-data` and train with `/api/train-model`, or place CSV in root and restart.

## Contributing
- Fork the repo, create a feature branch, open a PR.
- Keep changes focused; update docs when modifying APIs or services.

## License
Add a license file (e.g., MIT) to clarify usage.
