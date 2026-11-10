(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const workbench = byId('workbench');
  const source = byId('source');
  const files = byId('files');
  const sample = byId('sample').textContent.trimStart();
  let hosted = location.protocol === 'http:' || location.protocol === 'https:';
  let current = '';
  let dirty = false;
  const status = text => { byId('status').textContent = text; };
  const request = async (path, options) => {
    const response = await fetch(path, options);
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  };
  const preview = PESLitePreview.createPESPreview(byId('preview'), {
    title: 'sample',
    onSourceSelect: ({ line }) => {
      const rows = source.value.split('\n');
      const start = rows.slice(0, line - 1).reduce((sum, row) => sum + row.length + 1, 0);
      source.focus(); source.setSelectionRange(start, start + (rows[line - 1] || '').length);
    },
    onDiagnostics: entries => status(entries.length ? `${entries.length} diagnostic(s)` : 'Ready'),
    onLayoutChange: layout => {
      if (!hosted || !current) return;
      request(`/api/layout?path=${encodeURIComponent(current)}`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(layout),
      }).catch(error => status(error.message));
    },
  });
  const openSource = (name, text, layout = {}) => {
    byId('file-name').textContent = name;
    preview.setTitle(name.replace(/\.[^.]+$/, ''));
    preview.setLayout(layout);
    source.value = text;
    preview.update(text);
    dirty = false;
  };
  openSource('sample.pes', sample);
  source.addEventListener('input', () => { dirty = true; preview.update(source.value); });
  byId('save').addEventListener('click', async () => {
    if (!hosted || !current) { status('The standalone sample is not written to disk'); return; }
    try {
      await request(`/api/file?path=${encodeURIComponent(current)}`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: source.value }),
      });
      dirty = false; status('Saved');
    } catch (error) { status(error.message); }
  });
  function addEntries(entries, depth = 0) {
    for (const entry of entries) {
      if (entry.type === 'folder') addEntries(entry.children || [], depth + 1);
      else {
        const item = document.createElement('button');
        item.type = 'button'; item.textContent = entry.name; item.dataset.path = entry.path;
        item.style.paddingLeft = `${10 + depth * 14}px`;
        item.addEventListener('click', async () => {
          if (dirty && !confirm('Discard unsaved changes?')) return;
          try {
            const opened = await request(`/api/file?path=${encodeURIComponent(entry.path)}`);
            current = opened.path;
            [...files.children].forEach(child => child.classList.remove('active'));
            item.classList.add('active');
            openSource(opened.name, opened.source, opened.layout);
          } catch (error) { status(error.message); }
        });
        files.appendChild(item);
      }
    }
  }
  if (hosted) request('/api/workspace').then(workspace => {
    byId('workspace-name').textContent = workspace.name;
    addEntries(workspace.files);
    const target = [...files.children].find(item => item.dataset.path === workspace.open);
    target?.click();
  }).catch(() => { hosted = false; });
  function resize(sash, property, base) {
    sash.addEventListener('pointerdown', event => {
      const start = event.clientX;
      const width = parseFloat(getComputedStyle(workbench).getPropertyValue(property)) || base;
      sash.setPointerCapture(event.pointerId);
      const move = next => workbench.style.setProperty(property, `${Math.max(120, width + next.clientX - start)}px`);
      sash.addEventListener('pointermove', move);
      sash.addEventListener('pointerup', () => sash.removeEventListener('pointermove', move), { once: true });
    });
  }
  resize(byId('explorer-sash'), '--explorer-width', 220);
  resize(byId('editor-sash'), '--editor-width', 390);
  document.addEventListener('keydown', event => {
    if (!(event.ctrlKey || event.metaKey)) return;
    if (event.key.toLowerCase() === 'b') { event.preventDefault(); workbench.classList.toggle('explorer-hidden'); }
    if (event.key.toLowerCase() === 's') { event.preventDefault(); byId('save').click(); }
  });
  addEventListener('pagehide', () => {
    preview.destroy();
    if (hosted) navigator.sendBeacon('/api/session/close');
  }, { once: true });
})();
