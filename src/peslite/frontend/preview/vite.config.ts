import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    lib: {
      entry: fileURLToPath(new URL('./index.ts', import.meta.url)),
      name: 'PESLitePreview',
      formats: ['es', 'iife'],
      fileName: format => `pes-preview.${format === 'es' ? 'es' : 'iife'}.js`,
    },
  },
});
