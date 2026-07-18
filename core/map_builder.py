import ee
import folium
import os
from shapely.geometry import shape
from datetime import datetime

def create_suitability_map(suitability_image: ee.Image, aoi_geojson: dict, layers: dict = None) -> folium.Map:
    """
    Creates an interactive Folium map with the suitability score layer overlaid as tiles,
    along with the drawn AOI polygon boundary.
    """
    # 1. Calculate centroid of the AOI to center the map
    geom = shape(aoi_geojson)
    centroid = geom.centroid
    center_lat, center_lon = centroid.y, centroid.x
    
    # 2. Initialize Folium Map centered on the centroid
    # Using OpenStreetMap as the base layer, but also adding Satellite base layer option
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        control_scale=True
    )
    
    # Add Google Satellite Imagery hybrid basemap
    satellite_tiles = (
        "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"
    )
    folium.TileLayer(
        tiles=satellite_tiles,
        attr="Google Hybrid",
        name="Google Satellite",
        overlay=False,
        control=True
    ).add_to(m)
    
    # 3. Add Google Earth Engine Suitability Tile Layer
    # Vis params: red -> yellow -> green gradient
    vis_params = {
        "min": 0,
        "max": 100,
        "palette": ["#ff4b4b", "#f1c40f", "#2ecc71"]
    }
    
    # Fetch GEE map ID and Tile URL for final suitability
    map_id_dict = suitability_image.getMapId(vis_params)
    tile_url = map_id_dict["tile_fetcher"].url_format
    
    # Add suitability layer to Folium Map
    folium.TileLayer(
        tiles=tile_url,
        attr="Google Earth Engine",
        name="Suitability Score (0-100)",
        overlay=True,
        opacity=0.75
    ).add_to(m)
    
    # Add individual parameter layers if provided
    if layers:
        for layer_name, layer_img in layers.items():
            if layer_img is not None:
                # Individual layers are normalized from 0.0 to 1.0
                layer_vis = {"min": 0, "max": 1, "palette": ["#000000", "#ffffff"]}
                if layer_name == "landcover":
                    layer_vis = {"min": 0, "max": 1, "palette": ["#ff0000", "#00ff00"]} # Mask
                try:
                    l_dict = layer_img.getMapId(layer_vis)
                    folium.TileLayer(
                        tiles=l_dict["tile_fetcher"].url_format,
                        attr="GEE",
                        name=f"Layer: {layer_name.capitalize()}",
                        overlay=True,
                        show=False, # Hidden by default
                        opacity=0.9
                    ).add_to(m)
                except Exception:
                    pass
    
    # 4. Add AOI Boundary vector layer
    folium.GeoJson(
        aoi_geojson,
        name="Area of Interest (AOI)",
        style_function=lambda x: {
            "fillColor": "#3186cc",
            "color": "#123f66",
            "weight": 2.5,
            "fillOpacity": 0.05
        }
    ).add_to(m)
    
    # 5. Add a simple HTML color legend to the map
    legend_html = """
     <div style="position: fixed; 
                 bottom: 50px; left: 50px; width: 180px; height: 110px; 
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color:white; opacity: 0.9; padding: 10px;
                 border-radius: 5px; font-family: sans-serif;">
     <p style="margin: 0 0 5px 0; font-weight: bold;">Suitability Score</p>
     <div style="background: linear-gradient(to right, #ff4b4b, #f1c40f, #2ecc71); 
                 width: 100%; height: 15px; border-radius: 3px; margin-bottom: 5px;"></div>
     <div style="display: flex; justify-content: space-between;">
         <span>0 (Low)</span>
         <span>50</span>
         <span>100 (High)</span>
     </div>
     </div>
     """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 6. Add Layer Control so user can toggle Satellite/OSM base maps and suitability overlay
    folium.LayerControl().add_to(m)
    
    return m

def save_html_map(folium_map: folium.Map, output_dir: str = "output") -> str:
    """
    Saves the Folium map to a local HTML file.
    Returns the absolute path to the saved file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"suitability_map_{timestamp}.html"
    filepath = os.path.abspath(os.path.join(output_dir, filename))
    
    folium_map.save(filepath)
    return filepath
