from pathlib import Path
import folium


def gps_map(track, output: Path) -> None:
    center = [float(track.latitude.mean()), float(track.longitude.mean())]
    fmap = folium.Map(center, zoom_start=13)
    points = [[float(r.latitude), float(r.longitude)] for _, r in track.iterrows()]
    folium.PolyLine(points, color="blue", tooltip="GPS trajectory").add_to(fmap)
    fmap.save(output)
