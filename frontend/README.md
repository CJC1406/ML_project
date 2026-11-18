# PathSnap evaluator UI (Laptop 2)

A React + Leaflet UI for the FastAPI backend on Laptop 1. It never receives route labels.

## Setup

```bash
cd frontend
cp .env.example .env.local      # then set Laptop 1's IP
npm install
npm run dev                     # http://localhost:5173  (also on this laptop's LAN IP)
```

`.env.local`:

```
VITE_API_BASE_URL=http://<laptop-1-ip>:8000
```

Restart `npm run dev` after changing it.

## How it connects

The browser only talks to the Vite server on Laptop 2. Vite forwards `/api/*` to
`VITE_API_BASE_URL` (see `vite.config.js`), so no CORS configuration is needed on Laptop 1.

## Troubleshooting

- "Backend offline": run `curl http://<laptop-1-ip>:8000/health` from Laptop 2. If it fails, check
  that Laptop 1 started with `scripts/run_backend.sh` (binds `0.0.0.0`), that both laptops are on
  the same network, and that Laptop 1's firewall allows port 8000. Guest or university Wi-Fi often
  blocks device-to-device traffic; a phone hotspot works as a fallback.
- No models listed: the model artifact path on Laptop 1 doesn't exist (`/health` shows a count of 0).
- The map tiles come from OpenStreetMap, so Laptop 2 needs internet access.
