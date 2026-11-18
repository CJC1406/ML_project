import { useEffect } from "react";
import { MapContainer, TileLayer, GeoJSON, LayersControl, useMap } from "react-leaflet";
import L from "leaflet";

function FitBounds({ data }) {
  const map = useMap();
  useEffect(() => {
    if (!data?.features?.length) return;
    const bounds = L.geoJSON(data).getBounds();
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [24, 24] });
  }, [data, map]);
  return null;
}

// Route = what the backend reconstructed; GPS = the input readings. No ground truth is ever sent.
export default function RouteMap({ route, gps, routeLabel = "Predicted route" }) {
  const key = (g) => (g ? `${g.features.length}:${JSON.stringify(g).length}` : "none");
  return (
    <MapContainer center={[20, 0]} zoom={2} className="map">
      <TileLayer
        attribution='&copy; OpenStreetMap contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <LayersControl position="topright" collapsed={false}>
        {gps && (
          <LayersControl.Overlay checked name="GPS readings">
            <GeoJSON key={"gps" + key(gps)} data={gps}
              style={{ color: "#e34948", weight: 2, opacity: 0.8, dashArray: "4 4" }} />
          </LayersControl.Overlay>
        )}
        {route && (
          <LayersControl.Overlay checked name={routeLabel}>
            <GeoJSON key={"route" + key(route)} data={route}
              style={{ color: "#2a78d6", weight: 5, opacity: 0.85 }}
              onEachFeature={(f, layer) => layer.bindTooltip(`#${f.properties.order} arc ${f.properties.arc_id}`)} />
          </LayersControl.Overlay>
        )}
      </LayersControl>
      <FitBounds data={gps || route} />
    </MapContainer>
  );
}
