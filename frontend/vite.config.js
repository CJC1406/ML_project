import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Laptop 2 serves the UI; Laptop 1 runs the FastAPI backend.
// The browser only ever talks to this Vite server, which forwards /api/* to Laptop 1.
// That avoids CORS entirely, so the UI works from localhost or from Laptop 2's LAN IP.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_API_BASE_URL || "http://localhost:8000";
  const proxy = {
    "/api": {
      target,
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ""),
    },
  };
  console.log(`[pathsnap] proxying /api -> ${target}`);
  return {
    plugins: [react()],
    server: { host: true, port: 5173, strictPort: true, proxy },
    preview: { host: true, port: 5173, strictPort: true, proxy },
  };
});
