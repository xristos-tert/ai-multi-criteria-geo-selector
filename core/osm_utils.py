import requests
from shapely.geometry import shape

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
]

def fetch_osm_pois(aoi_geojson: dict, poi_key: str, poi_value: str):
    """
    Fetches POIs of the specified key-value pair within the AOI bounding box
    using the OpenStreetMap Overpass API. Uses case-insensitive regex matching (~"value",i).
    Returns a list of GeoJSON Point coordinates: [[lon, lat], [lon, lat], ...]
    """
    # 1. Calculate bounding box of AOI
    geom = shape(aoi_geojson)
    bounds = geom.bounds # (minx, miny, maxx, maxy) -> (west, south, east, north)
    
    # Overpass expects (south, west, north, east)
    s, w, n, e = bounds[1], bounds[0], bounds[3], bounds[2]
    
    # Expand the bounding box slightly to catch POIs just outside the AOI (e.g. 0.05 degrees ~ 5km)
    buffer_deg = 0.05
    s, w, n, e = s - buffer_deg, w - buffer_deg, n + buffer_deg, e + buffer_deg
    
    # 2. Build Overpass query
    # "out center" calculates the centroid for ways/relations so we just get a point
    query = f"""
    [out:json][timeout:25];
    (
      node["{poi_key}"~"{poi_value}",i]({s},{w},{n},{e});
      way["{poi_key}"~"{poi_value}",i]({s},{w},{n},{e});
      relation["{poi_key}"~"{poi_value}",i]({s},{w},{n},{e});
    );
    out center;
    """
    
    # 3. Execute request with fallbacks
    for url in OVERPASS_URLS:
        try:
            response = requests.post(
                url, 
                data={'data': query},
                headers={'User-Agent': 'AIMapper/1.0 (SpatialAnalysisTool)'},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # 4. Parse results into a list of [lon, lat] coordinates
            points = []
            for element in data.get('elements', []):
                if element['type'] == 'node':
                    points.append([element['lon'], element['lat']])
                elif 'center' in element:
                    points.append([element['center']['lon'], element['center']['lat']])
                    
            return points
        except Exception:
            continue
            
    return []
