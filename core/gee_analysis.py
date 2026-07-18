import os
import ee
import geopandas as gpd
from shapely.geometry import shape
from dotenv import load_dotenv

# Define ESA WorldCover class mappings to their respective numeric values in GEE
ESA_CLASS_MAPPING = {
    "tree_cover": 10,
    "shrubland": 20,
    "grassland": 30,
    "cropland": 40,
    "built_up": 50,
    "bare_vegetation": 60,
    "snow_ice": 70,
    "open_water": 80,
    "herbaceous_wetland": 90,
    "mangroves": 95,
    "moss_lichen": 100
}

def init_gee():
    """
    Initializes Google Earth Engine.
    On Streamlit Cloud: uses a Service Account JSON key stored in st.secrets.
    Locally: uses personal credentials from 'earthengine authenticate' + .env project ID.
    """
    load_dotenv()
    project_id = os.getenv("GEE_PROJECT_ID", "")

    # Try Streamlit Cloud secrets first
    service_account_json = None
    try:
        import streamlit as st
        project_id = st.secrets.get("GEE_PROJECT_ID", project_id)
        if "GEE_SERVICE_ACCOUNT_JSON" in st.secrets:
            service_account_json = st.secrets["GEE_SERVICE_ACCOUNT_JSON"]
    except Exception:
        pass  # Running locally, no st.secrets available

    # If we have a service account JSON, use it (cloud deployment)
    if service_account_json:
        try:
            import json
            service_info = json.loads(service_account_json)
            credentials = ee.ServiceAccountCredentials(
                email=service_info["client_email"],
                key_data=service_account_json
            )
            ee.Initialize(credentials=credentials, project=project_id)
            print("Google Earth Engine initialized with Service Account.")
            return
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Earth Engine with Service Account: {e}. "
                "Check that GEE_SERVICE_ACCOUNT_JSON is valid JSON in your Streamlit secrets, "
                "and that the service account has Earth Engine Resource Admin role in Google Cloud IAM."
            )

    # Fall back to personal auth (local development only)
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        print("Google Earth Engine successfully initialized.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Earth Engine: {e}. "
            "Please run 'earthengine authenticate' in your command prompt and ensure GEE_PROJECT_ID is set in your .env file."
        )

def calculate_aoi_area_km2(geojson_geometry: dict) -> float:
    """
    Calculates the area of a GeoJSON geometry in square kilometers.
    Uses GeoPandas to handle proper equal-area projection locally.
    """
    # Create a Shapely geometry from the GeoJSON dict
    geom = shape(geojson_geometry)
    
    # Create a GeoSeries and assign WGS84 CRS (EPSG:4326)
    gs = gpd.GeoSeries([geom], crs="EPSG:4326")
    
    # Reproject to a local UTM zone or World Equal Area (EPSG:6933) to calculate accurate area
    gs_equal = gs.to_crs("EPSG:6933")
    
    # Area in square meters -> convert to square kilometers
    area_km2 = gs_equal.area.iloc[0] / 1e6
    return area_km2

def get_gee_layers(aoi_geojson: dict, params: dict, poi_points: list = None) -> dict:
    """
    Runs the spatial layers analysis on Google Earth Engine.
    Clips all rasters to the AOI, normalizes scores between 0 and 1,
    and returns a dictionary of ee.Image objects.
    """
    # 1. Enforce AOI Area Limit
    area_km2 = calculate_aoi_area_km2(aoi_geojson)
    if area_km2 > 5000.0:
        raise ValueError(
            f"Selected Area of Interest is too large ({area_km2:.1f} km²). "
            "The maximum allowed area is 5,000 km² to maintain performance. Please draw a smaller area."
        )

    # 2. Convert GeoJSON to GEE Geometry
    # Extract coordinates and type from GeoJSON
    geom_type = aoi_geojson.get("type")
    coords = aoi_geojson.get("coordinates")
    if geom_type == "Polygon":
        aoi = ee.Geometry.Polygon(coords)
    elif geom_type == "MultiPolygon":
        aoi = ee.Geometry.MultiPolygon(coords)
    else:
        raise ValueError("Unsupported geometry type. Please provide a Polygon or MultiPolygon.")

    layers = {}

    # --- LAYER 1: SLOPE (SRTM 30m DEM) ---
    # Load the SRTM Digital Elevation Model
    dem = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    # Calculate slope in degrees
    slope = ee.Terrain.slope(dem)
    slope_max = params.get("slope_max_degrees", 90.0)
    # Score logic: 1.0 at 0 degrees, decreasing linearly to 0.0 at/above slope_max
    slope_score = ee.Image(1.0).subtract(slope.divide(slope_max)).clamp(0.0, 1.0)
    layers["slope"] = slope_score

    # --- LAYER 2: ROAD PROXIMITY (OpenStreetMap Roads via GEE community datasets) ---
    # GRIP4 roads dataset (Using Europe specifically as the global merge fails on GEE servers because 'Asia' is missing)
    # This guarantees it works perfectly for the current test in Greece.
    roads = ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/Europe").filterBounds(aoi)
    road_max_dist = params.get("road_distance_max_m", 100000.0)
    
    # Compute Euclidean distance from each pixel to the nearest road feature, in meters
    # (Using FeatureCollection.distance avoids the 512 pixel kernel limit of Image.distance)
    road_distance = roads.distance(road_max_dist).clip(aoi)
    # Score logic: 1.0 at 0m, linearly decreasing to 0.0 at road_max_dist
    # unmask(0) ensures pixels with no roads nearby get score 0 instead of being masked out
    road_score = ee.Image(1.0).subtract(road_distance.divide(road_max_dist)).clamp(0.0, 1.0).unmask(0.0).clip(aoi)
    layers["roads"] = road_score

    # --- LAYER 3: HISTORICAL FLOOD / WATER BUFFER (JRC Global Surface Water) ---
    # Load pre-computed historical surface water occurrence raster
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(aoi)
    # Extract historical water occurrence percentage (0-100%)
    occurrence = jrc.select("occurrence")
    # Identify pixels that were water > 10% of the time (historically flooded/wetland/riverbeds)
    water_pixels = occurrence.gt(10)
    flood_min_dist = params.get("flood_distance_min_m", 0.0)
    
    if flood_min_dist > 0:
        # Convert water binary mask to a raster with water=1, non-water masked
        # Then compute pixel distance to nearest water pixel directly on the raster (no vectors needed)
        water_raster = water_pixels.selfMask()  # keeps only water pixels, masks the rest
        water_dist = water_raster.distance(ee.Kernel.euclidean(flood_min_dist, "meters")).clip(aoi)
        # Score logic: pixels closer than flood_min_dist get 0.0, pixels beyond get 1.0
        # unmask(1.0) means areas with NO water nearby (unmasked by distance) are fully suitable
        flood_score = water_dist.gte(flood_min_dist).unmask(1.0)
    else:
        # No buffer required: score 1.0 everywhere except pixels directly on historical water
        flood_score = water_pixels.Not().unmask(1.0)
    layers["flood"] = flood_score

    # --- LAYER 4: LAND COVER EXCLUSION (ESA WorldCover 10m) ---
    # Load ESA WorldCover dataset (v200 is 2021 data, it's an ImageCollection)
    landcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").clip(aoi)
    exclude_classes = params.get("exclude_classes", [])
    
    # Initialize a mask with 1.0 (all suitable)
    landcover_score = ee.Image(1.0)
    if exclude_classes:
        # Create a mask where excluded classes are 0, allowed are 1
        mask = ee.Image(1.0)
        for cls_name in exclude_classes:
            val = ESA_CLASS_MAPPING.get(cls_name.lower())
            if val is not None:
                mask = mask.where(landcover.eq(val), 0.0)
        
        # Apply a focal mean (blur) to create a 'Proximity Penalty' buffer around excluded areas
        # This prevents areas right next to highways/cities from being 100% suitable.
        landcover_score = mask.focalMean(radius=300, units='meters')
        
    layers["landcover"] = landcover_score

    # --- LAYER 5: ELEVATION RANGE (SRTM 30m DEM) ---
    elevation = ee.Image("CGIAR/SRTM90_V4").select("elevation").clip(aoi)
    elevation_min = params.get("elevation_min_m", -1000.0)
    elevation_max = params.get("elevation_max_m", 9000.0)
    
    # Score logic: 1.0 if elevation is within [min, max] range.
    # Score decreases linearly as elevation falls below min OR rises above max.
    # The "buffer" defines how quickly the score drops to 0 outside the range.
    # We use a strict 200-meter buffer (if you are 200m outside the range, score = 0).
    buffer = 200.0
    
    # Distance below the minimum (positive value means "too low")
    below_min = ee.Image(elevation_min).subtract(elevation).max(0.0)
    # Distance above the maximum (positive value means "too high")
    above_max = elevation.subtract(ee.Image(elevation_max)).max(0.0)
    # Total distance from the acceptable range (0 if inside the range)
    distance_from_range = below_min.add(above_max)
    
    # Convert distance to a score: 1.0 at 0 distance, 0.0 at 'buffer' distance or more
    elevation_score = ee.Image(1.0).subtract(
        distance_from_range.divide(buffer)
    ).clamp(0.0, 1.0)
    
    layers["elevation"] = elevation_score

    # --- LAYER 6: POPULATION DENSITY (WorldPop 100m) ---
    # Using WorldPop for ultra-high resolution (100m) population mapping to prevent
    # bleeding into agricultural fields. The dataset provides population count per pixel.
    # WorldPop is tiled by country/region, so we must filter by bounds and mosaic.
    worldpop_raw = ee.ImageCollection("WorldPop/GP/100m/pop") \
                    .filterBounds(aoi) \
                    .filter(ee.Filter.date("2020-01-01", "2020-12-31")) \
                    .mosaic().select("population")
    
    # We must explicitly set projection because mosaic() discards it
    worldpop_proj = ee.ImageCollection("WorldPop/GP/100m/pop").first().projection()
    
    # WorldPop pixels are 100x100m (0.01 km2).
    # To convert count per pixel to density (people per km2), we multiply by 100.
    pop_density = worldpop_raw.setDefaultProjection(worldpop_proj).resample('bilinear').multiply(100.0).clip(aoi)
    
    pop_max = params.get("population_density_max", 999999.0)
    # Cap the threshold to a sensible maximum (500 people/km2) so the gradient is always visible.
    # Without this, a value like 999999 makes the score ~1.0 everywhere (all white).
    effective_pop_max = min(pop_max, 500.0) if pop_max > 500.0 else max(1.0, pop_max)
    
    # Score logic: Linear gradient from 1.0 (at 0 population) to 0.0 (at or above effective_pop_max)
    pop_score = ee.Image(1.0).subtract(
        pop_density.divide(effective_pop_max)
    ).clamp(0.0, 1.0).unmask(1.0).clip(aoi)
    
    layers["population"] = pop_score

    # --- LAYER 7: POI PROXIMITY (OpenStreetMap) ---
    if poi_points and len(poi_points) > 0:
        poi_max_dist = params.get("poi_distance_max_m", 5000.0)
        poi_logic = params.get("poi_logic", "AND").upper()
        
        if poi_logic == "OR" or len(poi_points) == 1:
            # Flatten the groups into a single list of points
            flat_points = [pt for group in poi_points for pt in group]
            features = [ee.Feature(ee.Geometry.Point(coords)) for coords in flat_points]
            poi_fc = ee.FeatureCollection(features)
            
            poi_distance = poi_fc.distance(poi_max_dist).clip(aoi)
            poi_score = ee.Image(1.0).subtract(poi_distance.divide(poi_max_dist)).clamp(0.0, 1.0).unmask(0.0).clip(aoi)
            layers["poi"] = poi_score
        else:
            # AND logic: Calculate suitability for each group independently, then average them
            suitability_images = []
            for group in poi_points:
                features = [ee.Feature(ee.Geometry.Point(coords)) for coords in group]
                poi_fc = ee.FeatureCollection(features)
                poi_distance = poi_fc.distance(poi_max_dist).clip(aoi)
                suit = ee.Image(1.0).subtract(poi_distance.divide(poi_max_dist)).clamp(0.0, 1.0).unmask(0.0).clip(aoi)
                suitability_images.append(suit)
                
            # Compute average of all suitability images
            image_collection = ee.ImageCollection.fromImages(suitability_images)
            layers["poi"] = image_collection.mean()

    return layers
