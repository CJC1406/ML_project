from dataclasses import dataclass
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class TripProjection:
    source_crs: str
    target_crs: str
    epsg: int | None
    forward: Transformer
    inverse: Transformer


def choose_trip_projection(longitudes, latitudes) -> TripProjection:
    """Choose a local UTM CRS from the trip's full bounds and preserve WGS84 order."""
    lon = (min(longitudes) + max(longitudes)) / 2
    lat = (min(latitudes) + max(latitudes)) / 2
    if not (-80 <= lat <= 84):
        # Polar regions need a polar stereographic CRS; EPSG selection is explicit.
        epsg = 3413 if lat >= 0 else 3031
    else:
        zone = int((lon + 180) // 6) + 1
        epsg = (32600 if lat >= 0 else 32700) + zone
    source = CRS.from_epsg(4326)
    target = CRS.from_epsg(epsg)
    return TripProjection("EPSG:4326", target.to_string(), epsg, Transformer.from_crs(source, target, always_xy=True), Transformer.from_crs(target, source, always_xy=True))


def project_xy(projection: TripProjection, longitude: float, latitude: float) -> tuple[float, float]:
    return projection.forward.transform(longitude, latitude)
