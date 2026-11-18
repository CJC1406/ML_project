// All calls go through the Vite proxy (see vite.config.js), never directly to Laptop 1.
const BASE = "/api";

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new Error("Cannot reach the Vite server.");
  }
  const text = await res.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = null;
  }
  if (!res.ok) {
    // Vite returns 500/502/504 with no JSON body when Laptop 1 is unreachable.
    const detail = body?.detail ?? (res.status >= 500 && !body ? "Backend unreachable (is Laptop 1 running uvicorn on 0.0.0.0:8000?)" : text);
    throw new Error(`${res.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
  return body;
}

export const api = {
  health: () => request("/health"),
  trips: () => request("/trips"),
  models: () => request("/models"),
  metrics: () => request("/metrics"),
  predict: (tripId, model, decoder) =>
    request("/predict", { method: "POST", body: JSON.stringify({ trip_id: tripId, model, decoder }) }),
};

// The Folium map is served by the backend; open it through the same proxy.
export const mapHref = (mapUrl) => `${BASE}${mapUrl}`;

export const BACKEND_TARGET = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
