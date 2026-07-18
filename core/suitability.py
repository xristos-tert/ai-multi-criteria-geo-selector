import ee
import geemap
import os
import tempfile
from datetime import datetime
from shapely.geometry import shape

def compute_suitability_score(layers: dict, weights: dict, aoi_geojson: dict) -> ee.Image:
    """
    Combines the individual 0-1 spatial layers using the weight parameters.
    Landcover is treated as a hard Boolean mask (VETO), while the rest are weighted factors.
    Returns a combined suitability score ee.Image scaled from 0 to 100, strictly clipped to the AOI.
    """
    combined_image = ee.Image(0.0)
    total_active_weight = 0.0
    
    # 1. Add up all non-mask weighted layers
    for layer_name, layer_image in layers.items():
        if layer_name == "landcover":
            continue # Skip landcover here, it's a mask
            
        weight = weights.get(layer_name, 0.0)
        if weight > 0:
            weighted_layer = layer_image.multiply(weight)
            combined_image = combined_image.add(weighted_layer)
            total_active_weight += weight
            
    # 2. Re-normalize the score if the sum of active weights is less than 1.0
    # (Because we ignored the landcover weight that Gemini might have assigned)
    if total_active_weight > 0:
        combined_image = combined_image.divide(total_active_weight)
        
    # 3. Apply the Boolean Masks (VETO)
    if "landcover" in layers:
        # Multiply by landcover: anything excluded (0) becomes 0 overall (100% Red)
        combined_image = combined_image.multiply(layers["landcover"])
        
    # Scale from 0-1 to 0-100
    suitability_score = combined_image.multiply(100.0)
    
    # Strictly clip the final image to the AOI to prevent global rendering
    geom_type = aoi_geojson.get("type")
    coords = aoi_geojson.get("coordinates")
    if geom_type == "Polygon":
        region = ee.Geometry.Polygon(coords)
    elif geom_type == "MultiPolygon":
        region = ee.Geometry.MultiPolygon(coords)
    else:
        region = ee.Geometry.Polygon(coords) # Fallback
        
    suitability_score = suitability_score.clip(region)
    
    return suitability_score.rename("suitability")

def export_suitability_geotiff(suitability_image: ee.Image, aoi_geojson: dict, layers: dict = None, output_dir: str = "output") -> str:
    """
    Exports the GEE suitability image as a local GeoTIFF file using geemap.
    If 'layers' is provided, exports a multi-band GeoTIFF containing the final score AND all individual spatial layers.
    Returns the absolute path to the saved file.
    """
    # Ensure output directory exists (fall back to temp dir on cloud)
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    except OSError:
        output_dir = tempfile.mkdtemp()
        
    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"suitability_{timestamp}.tif"
    filepath = os.path.abspath(os.path.join(output_dir, filename))
    
    # Convert GeoJSON AOI to GEE geometry
    geom_type = aoi_geojson.get("type")
    coords = aoi_geojson.get("coordinates")
    if geom_type == "Polygon":
        region = ee.Geometry.Polygon(coords)
    elif geom_type == "MultiPolygon":
        region = ee.Geometry.MultiPolygon(coords)
    else:
        raise ValueError("Unsupported geometry type for export.")

    print(f"Exporting suitability raster as GeoTIFF to: {filepath}")
    
    # Combine layers into a multi-band image for export, ensuring uniform data types (Float)
    export_image = suitability_image.toFloat()
    if layers:
        for layer_name, layer_img in layers.items():
            if layer_img is not None:
                # Rename the band and cast to Float to prevent GeoTIFF export errors
                band = layer_img.rename(layer_name).toFloat()
                export_image = export_image.addBands(band)

    # Export using geemap (scale 100m to prevent 32MB GEE payload limit error for multi-band)
    # 30m resolution for 6 bands easily exceeds 150MB for medium-sized polygons.
    geemap.ee_export_image(
        export_image,
        filename=filepath,
        scale=100,
        region=region,
        file_per_band=False
    )
    
    # geemap intercepts EE errors and prints them without raising exceptions.
    # We must explicitly check if the file was created.
    if not os.path.exists(filepath):
        raise RuntimeError(f"Earth Engine Export failed silently. Area might be too large even at 100m scale.")
        
    return filepath
