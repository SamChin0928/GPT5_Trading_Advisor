// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND = 'http://127.0.0.1:8000' // FastAPI inside Codespace container

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,           // allow external access (useful in Codespaces/Docker)
    port: 5173,
    proxy: {
      // API -> FastAPI
      '/api': {
        target: BACKEND,
        changeOrigin: true,
      },
      // Static files served by FastAPI at /data -> proxy as well
      '/data': {
        target: BACKEND,
        changeOrigin: true,
      },
    },
  },
})
