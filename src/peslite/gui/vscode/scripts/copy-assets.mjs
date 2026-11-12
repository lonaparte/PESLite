import { copyFileSync, cpSync, existsSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const source = resolve(root, '../web_dist');
if (!existsSync(resolve(source, 'index.html'))) {
  throw new Error('Build the shared PESLite GUI before packaging the VS Code extension.');
}
mkdirSync(resolve(root, 'media'), { recursive: true });
cpSync(source, resolve(root, 'media'), { recursive: true, force: true });
copyFileSync(resolve(root, '../../../../LICENSE'), resolve(root, 'LICENSE'));
