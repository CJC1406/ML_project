"""Interactive Folium maps: GPS observations, nearby road network, predicted route(s), reference route."""
from __future__ import annotations

import folium
import numpy as np

COLORS = dict(network="#9a9890", gps="#e34948", reference="#1baf7a", classifier="#eb6834", hmm="#2a78d6")


def _line(nodes, arcs, route):
    """Polyline pieces (lat, lon) for an arc sequence, split wherever consecutive arcs do not touch."""
    pieces = []
    for a in route:
        u, v = arcs[int(a)]
        p_u, p_v = [float(nodes[u, 1]), float(nodes[u, 0])], [float(nodes[v, 1]), float(nodes[v, 0])]
        if pieces and pieces[-1][-1] == p_u:
            pieces[-1].append(p_v)
        else:
            pieces.append([p_u, p_v])
    return pieces


def trip_map(nodes, arcs, track, routes: dict, network_arcs=None, title: str = "") -> folium.Map:
    """routes: {layer name: (arc id array, colour key)}. network_arcs: arcs to draw as the road layer."""
    fmap = folium.Map(location=[float(track[:, 1].mean()), float(track[:, 0].mean())], zoom_start=14,
                      tiles="OpenStreetMap", control_scale=True)
    if network_arcs is not None and len(network_arcs):
        g = folium.FeatureGroup(name="Road network (near the track)", show=True)
        for a in network_arcs:
            u, v = arcs[int(a)]
            folium.PolyLine([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
                            color=COLORS["network"], weight=1, opacity=0.6).add_to(g)
        g.add_to(fmap)
    g = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(track) // 1500)
    for lon, lat in track[::step, :2]:
        folium.CircleMarker([float(lat), float(lon)], radius=1.5, color=COLORS["gps"], fill=True,
                            fill_opacity=0.5, weight=0).add_to(g)
    g.add_to(fmap)
    for name, (route, key) in routes.items():
        g = folium.FeatureGroup(name=name, show=True)
        for piece in _line(nodes, arcs, route):
            folium.PolyLine(piece, color=COLORS[key], weight=9 if key == "reference" else (5 if key == "classifier" else 3),
                            opacity=0.5 if key == "reference" else 0.9, tooltip=name).add_to(g)
        g.add_to(fmap)
    folium.Marker([float(track[0, 1]), float(track[0, 0])], tooltip="start",
                  icon=folium.Icon(color="green", icon="play")).add_to(fmap)
    folium.Marker([float(track[-1, 1]), float(track[-1, 0])], tooltip="end",
                  icon=folium.Icon(color="red", icon="stop")).add_to(fmap)
    if title:
        fmap.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:50px;z-index:9999;background:#fff;padding:6px 10px;'
            f'border-radius:4px;font:14px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)">{title}</div>'))
    items = "".join(f'<div><span style="display:inline-block;width:18px;height:4px;background:{COLORS[key]};'
                    f'vertical-align:middle;margin-right:6px"></span>{name}</div>' for name, (_, key) in routes.items())
    items += (f'<div><span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
              f'background:{COLORS["gps"]};margin:0 11px 0 5px"></span>GPS observations</div>')
    fmap.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:30px;left:10px;z-index:9999;background:#fff;padding:6px 10px;'
        f'border-radius:4px;font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)">{items}</div>'))
    folium.LayerControl(collapsed=False).add_to(fmap)
    lats, lons = track[:, 1], track[:, 0]
    fmap.fit_bounds([[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]])
    return fmap


def nearby_arcs(features: dict):
    return np.unique(features["cand"])
