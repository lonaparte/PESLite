import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: './',
  build: { outDir: '../web_dist', emptyOutDir: true, assetsDir: 'assets' },
  server: { host: '127.0.0.1' },
});
