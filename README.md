# AI Multi-Criteria Geo Selector

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-geo-selector.streamlit.app/)

**Natural Language to Spatial Suitability Mapping**

An AI-powered geospatial tool that converts plain-English site criteria into multi-criteria suitability heatmaps using Google Earth Engine, Gemini AI, and OpenStreetMap.

## Features

- **Natural Language Input**: Describe your ideal site in plain English/Greek (e.g. "flat land, close to a road, away from water")
- **AI-Powered Parsing**: Gemini AI extracts spatial parameters (slope, elevation, land cover, POIs, etc.) from your description
- **Multi-Criteria Analysis (MCA)**: 7 spatial layers are computed and weighted on Google Earth Engine
- **POI Proximity**: Query OpenStreetMap for specific places (streets, cafes, villages) with AND/OR combinatorial logic
- **Interactive Map**: Draw your Area of Interest (AOI) directly on the map
- **Export**: Download results as interactive HTML maps and multi-band GeoTIFF rasters

## Spatial Layers

| Layer | Source | Resolution |
|---|---|---|
| Slope | SRTM 30m DEM | 30m |
| Road Proximity | GRIP4 Global Roads | Vector |
| Flood/Water Buffer | JRC Global Surface Water | 30m |
| Land Cover | ESA WorldCover | 10m |
| Elevation | CGIAR SRTM | 90m |
| Population Density | WorldPop | 100m |
| POI Proximity | OpenStreetMap (Overpass API) | Vector |

## Tech Stack

- **Frontend**: Streamlit + Folium
- **AI**: Google Gemini API (natural language parsing)
- **Geospatial Engine**: Google Earth Engine (server-side raster analysis)
- **POI Data**: OpenStreetMap Overpass API
- **Data Models**: Pydantic v2

## Setup (Local Development)

```bash
# Clone the repo
git clone https://github.com/xristos-tert/ai-multi-criteria-geo-selector.git
cd ai-multi-criteria-geo-selector

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Authenticate with Google Earth Engine (one-time)
earthengine authenticate

# Create .env file with your keys
echo GEMINI_API_KEY=your-key-here > .env
echo GEE_PROJECT_ID=your-project-id >> .env
echo DAILY_TOKEN_LIMIT=1000000 >> .env

# Run
streamlit run app.py
```

## Deployment (Streamlit Cloud)

1. Fork/clone this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect the repo
3. Add secrets in the Streamlit Cloud dashboard:
   - `GEMINI_API_KEY`
   - `GEE_PROJECT_ID`
   - `GEE_SERVICE_ACCOUNT_JSON` (full JSON key from Google Cloud)
   - `DAILY_TOKEN_LIMIT`

