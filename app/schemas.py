from pydantic import BaseModel, Field

class PredictionInput(BaseModel):
    """
    Pydantic schema to validate all 30 input features.
    Enforces positive numeric constraints appropriate for medical data.
    """
    mean_radius: float = Field(..., gt=0, description="Mean of distances from center to points on the perimeter")
    mean_texture: float = Field(..., gt=0, description="Standard deviation of gray-scale values")
    mean_perimeter: float = Field(..., gt=0, description="Mean size of the core tumor perimeter")
    mean_area: float = Field(..., gt=0, description="Mean size of the core tumor area")
    mean_smoothness: float = Field(..., gt=0, description="Mean of local variation in radius lengths")
    mean_compactness: float = Field(..., gt=0, description="Mean of perimeter^2 / area - 1.0")
    mean_concavity: float = Field(..., ge=0, description="Mean of severity of concave portions of the contour")
    mean_concave_points: float = Field(..., ge=0, description="Mean for number of concave portions of the contour")
    mean_symmetry: float = Field(..., gt=0, description="Mean tumor symmetry score")
    mean_fractal_dimension: float = Field(..., gt=0, description="Mean of 'coastline approximation' - 1")
    
    radius_error: float = Field(..., gt=0, description="Standard error of the radius")
    texture_error: float = Field(..., gt=0, description="Standard error of the texture")
    perimeter_error: float = Field(..., gt=0, description="Standard error of the perimeter")
    area_error: float = Field(..., gt=0, description="Standard error of the area")
    smoothness_error: float = Field(..., gt=0, description="Standard error of the smoothness")
    compactness_error: float = Field(..., gt=0, description="Standard error of the compactness")
    concavity_error: float = Field(..., ge=0, description="Standard error of the concavity")
    concave_points_error: float = Field(..., ge=0, description="Standard error of the concave points")
    symmetry_error: float = Field(..., gt=0, description="Standard error of the symmetry")
    fractal_dimension_error: float = Field(..., gt=0, description="Standard error of the fractal dimension")
    
    worst_radius: float = Field(..., gt=0, description="Worst or largest mean value of the radius")
    worst_texture: float = Field(..., gt=0, description="Worst or largest mean value of the texture")
    worst_perimeter: float = Field(..., gt=0, description="Worst or largest mean value of the perimeter")
    worst_area: float = Field(..., gt=0, description="Worst or largest mean value of the area")
    worst_smoothness: float = Field(..., gt=0, description="Worst or largest mean value of the smoothness")
    worst_compactness: float = Field(..., gt=0, description="Worst or largest mean value of the compactness")
    worst_concavity: float = Field(..., ge=0, description="Worst or largest mean value of the concavity")
    worst_concave_points: float = Field(..., ge=0, description="Worst or largest mean value of the concave points")
    worst_symmetry: float = Field(..., gt=0, description="Worst or largest mean value of the symmetry")
    worst_fractal_dimension: float = Field(..., gt=0, description="Worst or largest mean value of the fractal dimension")

    model_config = {
        "json_schema_extra": {
            "example": {
                "mean_radius": 17.99,
                "mean_texture": 10.38,
                "mean_perimeter": 122.8,
                "mean_area": 1001.0,
                "mean_smoothness": 0.1184,
                "mean_compactness": 0.2776,
                "mean_concavity": 0.3001,
                "mean_concave_points": 0.1471,
                "mean_symmetry": 0.2419,
                "mean_fractal_dimension": 0.07871,
                "radius_error": 1.095,
                "texture_error": 0.9053,
                "perimeter_error": 8.589,
                "area_error": 153.4,
                "smoothness_error": 0.006399,
                "compactness_error": 0.04904,
                "concavity_error": 0.05373,
                "concave_points_error": 0.01587,
                "symmetry_error": 0.03003,
                "fractal_dimension_error": 0.006193,
                "worst_radius": 25.38,
                "worst_texture": 17.33,
                "worst_perimeter": 184.6,
                "worst_area": 2019.0,
                "worst_smoothness": 0.1622,
                "worst_compactness": 0.6656,
                "worst_concavity": 0.7119,
                "worst_concave_points": 0.2654,
                "worst_symmetry": 0.4601,
                "worst_fractal_dimension": 0.1189
            }
        }
    }
