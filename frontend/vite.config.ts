import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite dev server runs on :5173; the FastAPI backend is configured with CORS
// for http://localhost:5173 (see backend/app.py).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
