import os
import json
import datetime
from typing import List, Dict, Tuple
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Define custom exception for quota exceedance
class QuotaExceededError(Exception):
    """Exception raised when the daily Gemini token limit is exceeded."""
    pass

class POIQuery(BaseModel):
    key: str = Field(description="OSM tag key, e.g., 'amenity', 'place', 'highway', 'name'.")
    value: str = Field(description="OSM tag value, e.g., 'hospital', 'village', 'Ermou'.")

# Pydantic model for structured Gemini output parsing
class CriteriaParams(BaseModel):
    """
    Pydantic model representing the extracted spatial parameters.
    Default values are provided for variables that might not be mentioned by the user.
    """
    slope_max_degrees: float = Field(
        default=90.0, 
        description="Maximum slope in degrees. Defaults to 90.0 (no restriction) if not mentioned."
    )
    road_distance_max_m: float = Field(
        default=100000.0, 
        description="Maximum distance to nearest road in meters."
    )
    flood_distance_min_m: float = Field(
        default=0.0, 
        description="Minimum distance from flood zones in meters."
    )
    exclude_classes: List[str] = Field(
        default_factory=list, 
        description="List of ESA WorldCover classes to exclude. Valid values: 'tree_cover', 'shrubland', 'grassland', 'cropland', 'built_up', 'bare_vegetation', 'snow_ice', 'open_water', 'herbaceous_wetland', 'mangroves', 'moss_lichen'."
    )
    elevation_min_m: float = Field(
        default=-1000.0, 
        description="Minimum elevation in meters."
    )
    elevation_max_m: float = Field(
        default=9000.0, 
        description="Maximum elevation in meters."
    )
    population_density_max: float = Field(
        default=999999.0, 
        description="Maximum population density in people/km2. E.g. 'not urban' -> 150 people/km2."
    )
    poi_queries: List[POIQuery] = Field(
        default_factory=list,
        description="List of OpenStreetMap POI queries if the user requests proximity to specific things (e.g. key='place', value='village'). Leave empty if not requested."
    )
    poi_logic: str = Field(
        default="AND",
        description="Logical operator to combine multiple POIs. 'AND' means the user wants to be close to ALL of them. 'OR' means close to ANY of them."
    )
    poi_distance_max_m: float = Field(
        default=5000.0,
        description="Maximum distance to the requested POI in meters. E.g. 'close to a hospital' -> 1000.0."
    )
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "slope": 0.0,
            "roads": 0.0,
            "flood": 0.0,
            "landcover": 0.0,
            "elevation": 0.0,
            "population": 0.0,
            "poi": 0.0
        },
        description="Weights for each active layer, must sum to 1.0. Assign higher weights to criteria stressed by the user (e.g. 'strongly prefer flat' -> higher weight for slope)."
    )
    explanation: str = Field(
        default="",
        description="A detailed 2-4 sentence explanation of the reasoning behind the chosen weights and exclusions."
    )

class GeminiParser:
    def __init__(self, state_file_path: str = "quota_state.json"):
        # Load environment variables
        load_dotenv()
        # Check Streamlit Cloud secrets first, then fall back to .env
        try:
            import streamlit as st
            self.api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
            self.daily_limit = int(st.secrets.get("DAILY_TOKEN_LIMIT", os.getenv("DAILY_TOKEN_LIMIT", 1000000)))
        except Exception:
            self.api_key = os.getenv("GEMINI_API_KEY")
            self.daily_limit = int(os.getenv("DAILY_TOKEN_LIMIT", 1000000))
        self.state_file_path = state_file_path
        
        # Configure Gemini API client if key is present
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def _get_current_date_str(self) -> str:
        """Returns current date in YYYY-MM-DD format (UTC)."""
        return datetime.datetime.utcnow().strftime("%Y-%m-%d")

    def load_quota_state(self) -> Tuple[str, int]:
        """Loads the current quota usage from the local state JSON file."""
        if not os.path.exists(self.state_file_path):
            return self._get_current_date_str(), 0
        try:
            with open(self.state_file_path, "r") as f:
                data = json.load(f)
                return data.get("date", self._get_current_date_str()), data.get("tokens_used", 0)
        except Exception:
            # If corrupted, reset state
            return self._get_current_date_str(), 0

    def update_quota_state(self, date_str: str, tokens_used: int):
        """Saves the quota usage state to the local JSON file."""
        try:
            with open(self.state_file_path, "w") as f:
                json.dump({"date": date_str, "tokens_used": tokens_used}, f, indent=4)
        except Exception as e:
            print(f"Warning: Could not update quota state file: {e}")

    def check_quota(self) -> int:
        """
        Checks if the daily token limit has been exceeded.
        Resets the counter if the day has changed.
        Returns the current token count.
        """
        current_date = self._get_current_date_str()
        saved_date, tokens_used = self.load_quota_state()

        # If date has changed, reset the daily token counter
        if current_date != saved_date:
            tokens_used = 0
            self.update_quota_state(current_date, tokens_used)

        # Check if we are already over the safety limit
        if tokens_used >= self.daily_limit:
            raise QuotaExceededError(
                f"Daily Gemini API token quota exceeded ({tokens_used}/{self.daily_limit} tokens used today). "
                "No more calls allowed until tomorrow."
            )
        
        return tokens_used

    def parse_criteria(self, criteria_text: str) -> CriteriaParams:
        """
        Sends the user plain text criteria to Gemini 1.5 Flash and returns a structured CriteriaParams object.
        Tracks token usage to enforce the daily quota limit.
        """
        # Ensure API key is set
        if not self.api_key or not self.client:
            raise ValueError("GEMINI_API_KEY environment variable is not set. Please check your .env file.")

        # Check daily quota limit before running the API call
        tokens_used = self.check_quota()

        # Build the system instruction that tells Gemini to act as a GIS analyst
        system_instruction = (
            "You are a GIS spatial analyst expert. Extract site criteria parameters from "
            "the user's natural language input and return ONLY a valid JSON object. "
            "ESA WorldCover class names to use in exclude_classes: "
            "'built_up' (urban), 'tree_cover' (forest), 'cropland' (agriculture), "
            "'open_water' (lakes/rivers), 'herbaceous_wetland' (wetlands), "
            "'shrubland', 'grassland'. "
            "Weights MUST sum to exactly 1.0. Only weight layers that are mentioned. "
            "Set unmentioned layer weights to 0.0."
        )

        # Embed the exact JSON structure in the prompt so Gemini knows what to return.
        # This avoids using response_schema which is not supported on the free Developer API tier.
        prompt = f"""User criteria: '{criteria_text}'

Return ONLY a JSON object with exactly these fields (no extra fields, no markdown):
{{
  "slope_max_degrees": <float, e.g. 5.0 for flat land, 90.0 if not mentioned>,
  "road_distance_max_m": <float, e.g. 500.0 for near roads, 100000.0 if not mentioned>,
  "flood_distance_min_m": <float, e.g. 300.0 for away from floods, 0.0 if not mentioned>,
  "exclude_classes": <list of strings from: built_up, tree_cover, cropland, open_water, herbaceous_wetland, shrubland, grassland>,
  "elevation_min_m": <float, e.g. 0.0, use -1000.0 if not mentioned>,
  "elevation_max_m": <float, e.g. 300.0, use 9000.0 if not mentioned>,
  "population_density_max": <float, e.g. 100.0 for low density, 999999.0 if not mentioned>,
  "poi_queries": <list of objects, e.g. [{{"key": "amenity", "value": "hospital"}}, {{"key": "place", "value": "village"}}, {{"key": "name", "value": "Ermou (bare name without 'Odos' or 'Street')"}}], or [] if not mentioned>,
  "poi_logic": <"AND" or "OR", default to "AND" if they say "close to X and Y", use "OR" if they say "close to X or Y">,
  "poi_distance_max_m": <float, e.g. 1000.0 for close, 5000.0 if not mentioned>,
  "weights": {{
    "slope": <float>,
    "roads": <float>,
    "flood": <float>,
    "landcover": <float>,
    "elevation": <float>,
    "population": <float>,
    "poi": <float>
  }},
  "explanation": "<Detailed 2-4 sentence explanation of your spatial reasoning, why you assigned specific weights, and what constraints you applied based on the user's prompt. IMPORTANT: Write this explanation in the SAME LANGUAGE as the User criteria!>"
}}
All weight values must sum to exactly 1.0. Set weights to 0.0 for layers not mentioned."""

        # Ordered list of models to try — if one fails, fall back to the next
        # gemini-3.1-flash-lite: confirmed working on free tier (July 2026)
        # gemini-flash-lite-latest: alias that auto-resolves to latest stable lite model
        models_to_try = ["gemini-3.1-flash-lite", "gemini-flash-lite-latest"]
        
        import time
        last_error = None
        
        for model_name in models_to_try:
            # Each model gets up to 2 attempts (to handle transient 503/disconnects)
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            response_mime_type="application/json",
                            temperature=0.0  # Zero temperature for deterministic structured output
                        )
                    )

                    # Record token usage from the API response metadata
                    if response.usage_metadata:
                        total_tokens = response.usage_metadata.total_token_count
                        new_tokens_used = tokens_used + total_tokens
                        self.update_quota_state(self._get_current_date_str(), new_tokens_used)

                    # Parse the JSON and validate against Pydantic model
                    parsed_data = json.loads(response.text)
                    return CriteriaParams(**parsed_data)

                except QuotaExceededError:
                    raise  # Don't retry quota errors
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    # If it's a 404 (model not found) or quota error, skip retries and try next model
                    if "404" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        break
                    # For transient errors (503, disconnects), wait briefly and retry
                    if attempt < 1:
                        time.sleep(3)
                        continue
                    break  # Move to next model after 2 failed attempts
        
        # If all models and retries failed, raise the last error
        raise RuntimeError(f"Failed to parse criteria via Gemini API: {last_error}")
