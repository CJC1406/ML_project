"""Compatibility import; implementation lives in pathsnap.geo.distance."""
from pathsnap.geo.distance import *

EARTH_RADIUS_M = 6_371_000.0


def validate_coordinate(longitude: float, latitude: float) -> None:
    if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
        raise ValueError("invalid longitude/latitude")


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    validate_coordinate(lon1, lat1); validate_coordinate(lon2, lat2)
    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * atan2(sqrt(a), sqrt(1 - a))


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    validate_coordinate(lon1, lat1); validate_coordinate(lon2, lat2)
    x = cos(radians(lat2)) * sin(radians(lon2 - lon1))
    y = cos(radians(lat1)) * sin(radians(lat2)) - sin(radians(lat1)) * cos(radians(lat2)) * cos(radians(lon2 - lon1))
    return (degrees(atan2(x, y)) + 360) % 360


def heading_difference_deg(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def point_to_segment_m(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    """Approximate locally in metres, avoiding raw lon/lat Euclidean distance."""
    lon, lat = point; slon, slat = start; elon, elat = end
    scale_x = EARTH_RADIUS_M * cos(radians(lat)) * radians(1)
    scale_y = EARTH_RADIUS_M * radians(1)
    px, py = (lon - slon) * scale_x, (lat - slat) * scale_y
    ex, ey = (elon - slon) * scale_x, (elat - slat) * scale_y
    denom = ex * ex + ey * ey
    t = max(0.0, min(1.0, (px * ex + py * ey) / denom)) if denom else 0.0
    return sqrt((px - t * ex) ** 2 + (py - t * ey) ** 2)
