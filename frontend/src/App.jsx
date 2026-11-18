import { useEffect, useState } from "react";
import { api, mapHref, BACKEND_TARGET } from "./api.js";
import RouteMap from "./RouteMap.jsx";

const pct = (v) => (typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—");
const num = (v, d = 3) => (typeof v === "number" ? v.toFixed(d) : "—");
const DECODER_LABELS = { hmm_viterbi: "Classifier + HMM/Viterbi", classifier_only: "Classifier only" };

export default function App() {
  const [health, setHealth] = useState(null);
  const [healthError, setHealthError] = useState(null);
  const [trips, setTrips] = useState([]);
  const [models, setModels] = useState([]);
  const [labels, setLabels] = useState({});
  const [decoders, setDecoders] = useState(["hmm_viterbi", "classifier_only"]);
  const [metrics, setMetrics] = useState([]);
  const [tripId, setTripId] = useState("");
  const [model, setModel] = useState("");
  const [decoder, setDecoder] = useState("hmm_viterbi");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function connect() {
    setHealthError(null);
    try {
      const [h, t, m, mt] = await Promise.all([api.health(), api.trips(), api.models(), api.metrics()]);
      setHealth(h);
      setTrips(t.trips);
      setModels(m.models);
      setLabels(m.labels || {});
      if (m.decoders?.length) setDecoders(m.decoders);
      setMetrics(mt.metrics);
      setTripId((cur) => cur || t.trips[0] || "");
      setModel((cur) => cur || m.default || m.models[0] || "");
    } catch (e) {
      setHealth(null);
      setHealthError(e.message);
    }
  }

  useEffect(() => {
    connect();
  }, []);

  async function runPrediction() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await api.predict(tripId, model, decoder));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const connected = health?.status === "ok";
  const ev = result?.evaluation;

  return (
    <div className="layout">
      <aside className="panel">
        <h1>PathSnap Evaluator</h1>

        <section className="card">
          <div className="row">
            <span className={`dot ${connected ? "ok" : "bad"}`} />
            <strong>{connected ? "Backend connected" : "Backend offline"}</strong>
          </div>
          <code className="target">{BACKEND_TARGET}</code>
          {connected && (
            <ul className="facts">
              <li>Models loaded: {health.available_model_count}</li>
              <li>Trips available: {health.trip_count ?? trips.length}</li>
              {health.bundle_error && <li className="error">{health.bundle_error}</li>}
            </ul>
          )}
          {healthError && <p className="error">{healthError}</p>}
          <button className="secondary" onClick={connect}>Retry connection</button>
        </section>

        <section className="card">
          <label>
            Trip
            <select value={tripId} onChange={(e) => setTripId(e.target.value)} disabled={!trips.length}>
              {trips.length === 0 && <option value="">No trips available</option>}
              {trips.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
          <label>
            Model
            <select value={model} onChange={(e) => setModel(e.target.value)} disabled={!models.length}>
              {models.length === 0 && <option value="">No models loaded</option>}
              {models.map((m) => <option key={m} value={m}>{labels[m] || m}</option>)}
            </select>
          </label>
          <label>
            Sequence method
            <select value={decoder} onChange={(e) => setDecoder(e.target.value)}>
              {decoders.map((d) => <option key={d} value={d}>{DECODER_LABELS[d] || d}</option>)}
            </select>
          </label>
          <button onClick={runPrediction} disabled={!connected || !tripId || !model || loading}>
            {loading ? "Reconstructing route…" : "Predict route"}
          </button>
          {error && <p className="error">{error}</p>}
        </section>

        {result && (
          <section className="card">
            <h2>Prediction</h2>
            <ul className="facts">
              <li>Trip {result.trip_id}, {labels[result.model] || result.model}</li>
              <li>{DECODER_LABELS[result.sequence_method] || result.sequence_method}
                {result.hmm && ` (α ${num(result.hmm.alpha, 2)}, β ${num(result.hmm.beta, 1)} m)`}</li>
              <li>GPS readings: {result.observations}</li>
              <li>Route arcs: {result.route_arc_ids.length}</li>
              <li>Chain restarts: {result.chain_breaks}; unjoined gaps: {result.unjoined_gaps}</li>
              <li>Total time: {result.inference_time_ms.toFixed(0)} ms
                {result.timings_ms && ` (features ${result.timings_ms.features_ms.toFixed(0)}, model ${result.timings_ms.model_ms.toFixed(0)}, decode ${result.timings_ms.decode_ms.toFixed(0)})`}</li>
            </ul>
            {result.map_url && (
              <a className="link" href={mapHref(result.map_url)} target="_blank" rel="noreferrer">
                Open interactive Folium map
              </a>
            )}
            {result.warnings?.length > 0 && (
              <ul className="warnings">
                {result.warnings.map((w) => <li key={w}>{w}</li>)}
              </ul>
            )}
          </section>
        )}

        {result && (
          <section className="card">
            <h2>Evaluation</h2>
            {ev ? (
              <>
                <table>
                  <tbody>
                    <tr><td>Reading-level Top-1</td><td>{pct(ev.observation_top1)}</td></tr>
                    <tr><td>Route precision</td><td>{pct(ev.route_precision)}</td></tr>
                    <tr><td>Route recall</td><td>{pct(ev.route_recall)}</td></tr>
                    <tr><td>Route F1</td><td>{pct(ev.route_f1)}</td></tr>
                    <tr><td>Route mismatch fraction</td><td>{num(ev.route_mismatch_fraction)}</td></tr>
                    <tr><td>Route continuity</td><td>{pct(ev.route_continuity)}</td></tr>
                  </tbody>
                </table>
                <p className="muted">Computed on Laptop 1. Reference routes are never sent to this UI.</p>
              </>
            ) : (
              <p className="muted">No reference route on the backend for this trip.</p>
            )}
          </section>
        )}

        {metrics.length > 0 && (
          <section className="card">
            <h2>Model comparison (student test)</h2>
            <table>
              <thead>
                <tr><th>Model</th><th>Top-1</th><th>+HMM</th><th>Route F1</th><th>Log loss</th></tr>
              </thead>
              <tbody>
                {metrics.map((m) => (
                  <tr key={m.model}>
                    <td>{m.label || m.model}</td><td>{pct(m.top1_classifier_only)}</td><td>{pct(m.top1_hmm)}</td>
                    <td>{pct(m.route_f1_hmm)}</td><td>{num(m.log_loss)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted">{metrics[0].protocol}</p>
          </section>
        )}
      </aside>

      <main className="mapwrap">
        <RouteMap route={result?.route_geojson} gps={result?.gps_geojson}
          routeLabel={result ? DECODER_LABELS[result.sequence_method] : "Predicted route"} />
      </main>
    </div>
  );
}
