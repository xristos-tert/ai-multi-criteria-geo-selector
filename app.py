import streamlit as st
import os
import json
from datetime import datetime
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import shape
from dotenv import load_dotenv

# Import our core analysis modules
from core.gemini_parser import GeminiParser, QuotaExceededError, CriteriaParams
from core.gee_analysis import init_gee, get_gee_layers, calculate_aoi_area_km2
from core.suitability import compute_suitability_score, export_suitability_geotiff
from core.map_builder import create_suitability_map, save_html_map

# Set premium page layout configuration
st.set_page_config(
    page_title="AI Multi-Criteria Geo Selector",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN ENTERPRISE UI OVERHAUL ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        
        /* Apply clean Inter font globally */
        html, body, .stApp, p, h1, h2, h3, h4, h5, h6, label {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Ensure Material Symbols (icons) are NOT overridden by Inter */
        span[class*="icon"], span.material-symbols-rounded, .stIcon {
            font-family: "Material Symbols Rounded", "Material Icons" !important;
        }
        
        /* Enterprise Dashboard Theme */
        .reportview-container {
            background: #0f1115;
        }
        .main-header {
            font-size: 1.7rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 0.1rem;
            letter-spacing: -0.02em;
        }
        .sub-header {
            font-size: 0.95rem;
            color: #94a3b8;
            margin-bottom: 1.5rem;
            font-weight: 400;
        }
        .step-box {
            background-color: #161b22;
            padding: 1.0rem;
            border: 1px solid #30363d;
            border-radius: 4px;
            border-left: 4px solid #2563eb;
            margin-bottom: 1.0rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        .step-box h4 {
            font-size: 1.05rem;
            color: #c9d1d9;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        .stat-val {
            font-size: 1.4rem;
            font-weight: 600;
            color: #38bdf8;
        }
        /* Make Streamlit elements more compact */
        .stTextInput>div>div>input {
            border-radius: 3px;
        }
        .stTextArea>div>div>textarea {
            border-radius: 3px;
            font-size: 0.95rem;
        }
        .stButton>button {
            border-radius: 3px;
            font-weight: 600;
            border: 1px solid #2ea043;
            background-color: #238636;
            color: white;
        }
        .stButton>button:hover {
            background-color: #2ea043;
            border-color: #3fb950;
            color: white;
        }
        hr {
            margin-top: 1rem;
            margin-bottom: 1rem;
            border-color: #30363d;
        }
        /* Make selectbox (preset list) behave like a solid button instead of a text input */
        div[data-baseweb="select"] > div {
            cursor: pointer !important;
        }
        div[data-baseweb="select"] input {
            cursor: pointer !important;
            caret-color: transparent !important;
        }
    </style>
""", unsafe_allow_html=True)
# ----------------------------------------------------


# Helper: Read secrets from Streamlit Cloud first, then fall back to .env
def get_secret(key: str, default: str = "") -> str:
    """Checks st.secrets (Streamlit Cloud) first, then os.getenv (local .env)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)

# Helper function to set/update .env file from the UI inputs (local only)
def save_env_variables(api_key: str, project_id: str):
    with open(".env", "w") as f:
        f.write(f"GEMINI_API_KEY={api_key}\n")
        f.write(f"DAILY_TOKEN_LIMIT=1000000\n")
        f.write(f"GEE_PROJECT_ID={project_id}\n")
    # Reload environment
    load_dotenv(override=True)

# Load existing environment variables (local .env file)
load_dotenv()
default_api_key = get_secret("GEMINI_API_KEY", "")
default_project_id = get_secret("GEE_PROJECT_ID", "")

# Initialize quota state file path
QUOTA_FILE = "quota_state.json"

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("<h2 style='color: #00FF41; margin-bottom: 1rem; text-shadow: 0 0 5px #00FF41;'>[SYS_CONFIG]</h2>", unsafe_allow_html=True)

# API Keys & Credentials
api_key_input = st.sidebar.text_input("Gemini API Key", value=default_api_key, type="password", help="Enter your Gemini API key from Google AI Studio")
project_id_input = st.sidebar.text_input("GEE Project ID", value=default_project_id, help="Google Earth Engine Project ID (required for v1.0.0+)")

# Connection trigger
if st.sidebar.button("Save & Test Connection", use_container_width=True):
    if not api_key_input:
        st.sidebar.error("Gemini API Key is required.")
    else:
        save_env_variables(api_key_input, project_id_input)
        try:
            # Test GEE Initialization
            init_gee()
            st.sidebar.success("GEE & Keys initialized successfully!")
            st.session_state["gee_connected"] = True
        except Exception as e:
            st.sidebar.error(f"GEE initialization failed: {e}")
            st.session_state["gee_connected"] = False

# Display Gemini API token usage indicator in Sidebar
parser_for_stats = GeminiParser(state_file_path=QUOTA_FILE)
saved_date, tokens_used = parser_for_stats.load_quota_state()
limit = parser_for_stats.daily_limit
usage_percent = min(100.0, (tokens_used / limit) * 100)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**API Quota ({saved_date}):** {int(usage_percent)}%")
st.sidebar.progress(usage_percent / 100.0)
st.sidebar.caption(f"{tokens_used:,} / {limit:,} tokens")
st.sidebar.markdown("---")

# Cache GEE initialization so it only runs ONCE per session, not on every Streamlit rerun
@st.cache_resource
def get_gee_connection(project_id: str):
    """Initializes GEE once and caches the connection for the entire session."""
    try:
        import ee
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        return True
    except Exception:
        return False

# Check GEE initial status using cached function
if "gee_connected" not in st.session_state:
    st.session_state["gee_connected"] = get_gee_connection(default_project_id)

# Connection status badge
if st.session_state["gee_connected"]:
    st.sidebar.markdown("🟢 **Google Earth Engine:** Connected")
else:
    st.sidebar.markdown("🔴 **Google Earth Engine:** Disconnected (Authenticate or check Project ID)")

# --- MAIN INTERFACE ---
st.markdown("<h1 class='main-header'>AI Multi-Criteria Geo Selector</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Natural Language to Spatial Suitability Mapping</div>", unsafe_allow_html=True)

# Preset examples for fast-testing
preset_criteria = {
    "Select a preset...": "",
    "Flat Solar Farm (Flat, away from water, low population density)": "flat land with low slope, away from water and flood zones, low population density",
    "Logistics Warehouse (Near roads, flat land, not urban)": "very flat land, maximum 500 meters from roads, not urban, not wetland",
    "Eco-tourism Lodge (Forest land, away from urban areas, medium elevation)": "forested land, at least 1 kilometer away from urban built-up areas, elevation between 200 and 800 meters",
    "Agricultural Crop Land (Cropland, flat, away from forests)": "cropland, very low slope, away from forest tree cover, elevation below 300 meters"
}

# Layout with two main columns
col_inputs, col_map = st.columns([1, 1.2])

with col_inputs:
    st.markdown("<div class='step-box'><h4>Step 1: Define Site Criteria</h4></div>", unsafe_allow_html=True)
    
    # Placeholder to render text area FIRST
    text_area_container = st.empty()
    
    # Placeholder to render demo mode checkbox SECOND
    demo_mode_container = st.empty()
    
    st.markdown("<br>", unsafe_allow_html=True)
    preset_selection = st.radio("...or select a Preset Criteria Example:", list(preset_criteria.keys()))
    initial_text = preset_criteria[preset_selection] if preset_selection != "Select a preset..." else ""
    
    # Now define the elements inside the containers
    demo_mode = demo_mode_container.checkbox("🟢 Use Offline Demo Mode (Bypasses Gemini API)", value=False, help="Use pre-calculated spatial parameters to test the app without consuming your API daily quota.")
    
    criteria_text = text_area_container.text_area(
        "Describe your ideal site criteria in plain English:",
        value=initial_text,
        placeholder="e.g. flat land, close to roads, at least 500m away from flood zones, not urban, and below 300m elevation...",
        height=120,
        disabled=demo_mode
    )
    
    st.markdown("<div class='step-box'><h4>Step 2: Area of Interest Selection</h4></div>", unsafe_allow_html=True)
    st.info("Designate study area geometry (max limit: 5,000 km²).")

with col_map:
    # Action Button
    analyze_btn = st.button("Execute Geospatial Analysis", use_container_width=True, type="primary")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Build drawing map
    # Center initially near Thessaly plain, Greece (historic Daniel flood region)
    m = folium.Map(location=[39.4, 22.3], zoom_start=9)
    draw = Draw(
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False
        },
        edit_options={'edit': False, 'remove': True}
    )
    draw.add_to(m)
    
    # Render drawing map
    map_data = st_folium(m, use_container_width=True, height=650, key="draw_map")
    
    # Process drawn polygon
    aoi_geojson = None
    if map_data and map_data.get("all_drawings"):
        drawings = map_data["all_drawings"]
        if len(drawings) > 0:
            # Take the latest drawing
            aoi_geojson = drawings[-1]["geometry"]
            
            try:
                area_km2 = calculate_aoi_area_km2(aoi_geojson)
                if area_km2 > 5000.0:
                    st.error(f"❌ Selected area is too large: **{area_km2:,.1f} km²**. Max limit is **5,000 km²**.")
                else:
                    st.success(f"✅ Selected Area of Interest: **{area_km2:,.1f} km²**")
            except Exception as e:
                st.error(f"Error calculating area: {e}")



if analyze_btn:
    # Validate settings & state
    if not api_key_input:
        st.error("Please enter a Gemini API Key in the sidebar settings first.")
    elif not st.session_state["gee_connected"]:
        st.error("Google Earth Engine is not connected. Please authenticate and set your Project ID in the sidebar settings.")
    elif not criteria_text.strip():
        st.error("Please describe your site criteria in Step 1.")
    elif not aoi_geojson:
        st.error("Please draw an Area of Interest on the map in Step 2.")
    else:
        # Check area limit locally first
        area_km2 = calculate_aoi_area_km2(aoi_geojson)
        if area_km2 > 5000.0:
            st.error(f"Area of Interest is too large ({area_km2:,.1f} km²). Please select a smaller region.")
        else:
            # Update env configuration before running
            save_env_variables(api_key_input, project_id_input)
            
            # Start full analysis pipeline
            try:
                status = st.status("Initializing Spatial Analysis Pipeline...", expanded=True)
                
                if demo_mode:
                    status.write("> Demo Mode Active: Using offline pre-calculated spatial parameters...")
                    # Hardcoded spatial parameters for Demo Mode
                    if "Solar Farm" in preset_selection:
                        parsed_params = CriteriaParams(
                            slope_max_degrees=5.0, road_distance_max_m=5000.0, flood_distance_min_m=500.0,
                            exclude_classes=["open_water", "herbaceous_wetland", "built_up", "tree_cover"],
                            elevation_min_m=0.0, elevation_max_m=1000.0, population_density_max=100.0,
                            weights={"slope": 0.2, "roads": 0.2, "flood": 0.2, "landcover": 0.1, "elevation": 0.1, "population": 0.2}
                        )
                    elif "Logistics Warehouse" in preset_selection:
                        parsed_params = CriteriaParams(
                            slope_max_degrees=2.0, road_distance_max_m=500.0, flood_distance_min_m=0.0,
                            exclude_classes=["open_water", "herbaceous_wetland", "built_up"],
                            elevation_min_m=-1000.0, elevation_max_m=9000.0, population_density_max=999999.0,
                            weights={"slope": 0.4, "roads": 0.4, "flood": 0.0, "landcover": 0.2, "elevation": 0.0, "population": 0.0}
                        )
                    elif "Eco-tourism" in preset_selection:
                        parsed_params = CriteriaParams(
                            slope_max_degrees=90.0, road_distance_max_m=100000.0, flood_distance_min_m=0.0,
                            exclude_classes=["built_up"],
                            elevation_min_m=200.0, elevation_max_m=800.0, population_density_max=50.0,
                            weights={"slope": 0.0, "roads": 0.0, "flood": 0.0, "landcover": 0.4, "elevation": 0.4, "population": 0.2}
                        )
                    else:
                        # Default fallback Demo params
                        parsed_params = CriteriaParams(
                            slope_max_degrees=10.0, road_distance_max_m=5000.0, flood_distance_min_m=200.0,
                            exclude_classes=["built_up", "open_water"],
                            elevation_min_m=-1000.0, elevation_max_m=9000.0, population_density_max=500.0,
                            weights={"slope": 0.2, "roads": 0.2, "flood": 0.2, "landcover": 0.2, "elevation": 0.0, "population": 0.2}
                        )
                else:
                    # 1. Parse natural language criteria to JSON parameters via Gemini
                    status.write("> Sending criteria to Gemini API for parsing...")
                    parser = GeminiParser(state_file_path=QUOTA_FILE)
                    parsed_params = parser.parse_criteria(criteria_text)
                
                # Show parsed variables to the user
                status.write("> Criteria successfully loaded!")
                # 2. Query POIs from OpenStreetMap if requested
                poi_points = []
                poi_results_messages = []
                if hasattr(parsed_params, 'poi_queries') and parsed_params.poi_queries:
                    from core.osm_utils import fetch_osm_pois
                    for q in parsed_params.poi_queries:
                        # q is a POIQuery dict (or object, pydantic handles both via dict() or dot notation)
                        poi_key = q.key if hasattr(q, 'key') else q['key']
                        poi_value = q.value if hasattr(q, 'value') else q['value']
                        
                        status.write(f"> Querying OpenStreetMap for: {poi_key}={poi_value}...")
                        points = fetch_osm_pois(aoi_geojson, poi_key, poi_value)
                        if points:
                            poi_points.append(points)
                            status.write(f"> Found {len(points)} '{poi_key}={poi_value}' locations.")
                            poi_results_messages.append(f"✅ Βρέθηκαν **{len(points)} '{poi_value}'** στην περιοχή σου!")
                        else:
                            status.write(f"> No '{poi_key}={poi_value}' found in this area.")
                            poi_results_messages.append(f"❌ Δεν βρέθηκε κανένα **'{poi_value}'** στην περιοχή σου.")
                            
                    if len(poi_points) == 0:
                        status.write("> No requested POIs found in the area. Disabling POI weight.")
                        parsed_params.weights["poi"] = 0.0
                
                # 3. Query individual layers on GEE
                status.write("> Requesting spatial data layers from Google Earth Engine...")
                layers = get_gee_layers(aoi_geojson, parsed_params.dict(), poi_points=poi_points)
                
                # 3. Calculate weighted suitability score image
                status.write("> Aggregating layers and computing suitability score...")
                suitability_image = compute_suitability_score(layers, parsed_params.weights, aoi_geojson)
                
                # 4. Generate local Folium map output
                status.write("> Generating interactive map layers...")
                final_map = create_suitability_map(suitability_image, aoi_geojson, layers)
                
                # Save maps/GeoTIFFs locally
                html_path = save_html_map(final_map)
                
                # Export GeoTIFF synchronously
                status.write("> Exporting suitability raster as GeoTIFF (this takes 10-15 seconds)...")
                geotiff_path = export_suitability_geotiff(suitability_image, aoi_geojson, layers)
                
                # Collapse the status terminal when completely done
                status.update(label="Analysis completed successfully!", state="complete", expanded=False)
                
                # Keep the JSON expander visible outside the status
                if hasattr(parsed_params, 'explanation') and parsed_params.explanation:
                    st.info(f"🤖 **Gemini Reasoning:** {parsed_params.explanation}")
                    
                # Show POI results if any were requested
                if poi_results_messages:
                    for msg in poi_results_messages:
                        if "✅" in msg:
                            st.success(msg)
                        else:
                            st.warning(msg)
                            
                with st.expander("View Parsed Spatial Parameters & Weights (JSON)", expanded=False):
                    st.json(parsed_params.dict())
                
                # Render the final interactive Folium map in Streamlit
                st.markdown("### 🗺️ Suitability Heatmap Overlay")
                
                col_map_out, col_stats_out = st.columns([2.5, 1])
                
                with col_map_out:
                    # Using iframe to render the saved local html map perfectly without rendering latency
                    with open(html_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=600)
                            
                with col_stats_out:
                    st.markdown("#### Export Outputs")
                    
                    # Direct download buttons for files
                    with open(html_path, "rb") as file:
                        st.download_button(
                            label="Download Interactive Map (.html)",
                            data=file,
                            file_name=os.path.basename(html_path),
                            mime="text/html",
                            use_container_width=True
                        )
                        
                    with open(geotiff_path, "rb") as file:
                        st.download_button(
                            label="Download Raster Dataset (.tif)",
                            data=file,
                            file_name=os.path.basename(geotiff_path),
                            mime="image/tiff",
                            use_container_width=True
                        )
                        
                    st.markdown("---")
                    st.markdown("#### Execution Metadata")
                    st.markdown(f"**Total Area Analyzed:** `{area_km2:.1f} km²`\n"
                                f"**Spatial Resolution:** `30 meters/pixel`\n"
                                f"**Base DEM:** `SRTM 30m`\n"
                                f"**Landcover classification:** `ESA WorldCover 10m`\n"
                                f"**Flood Occurrence Source:** `JRC Global Surface Water`\n"
                                f"**Road proximity network:** `GRIP Global Roads` \n"
                                f"**Map center:** `{shape(aoi_geojson).centroid.y:.5f}, {shape(aoi_geojson).centroid.x:.5f}`")
                    
            except QuotaExceededError as qe:
                st.error(f"❌ {qe}")
            except Exception as e:
                st.error(f"❌ Critical Error occurred during processing: {e}")
