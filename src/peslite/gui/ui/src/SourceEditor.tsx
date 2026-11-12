import { useEffect, useRef } from 'react';
import { basicSetup } from 'codemirror';
import { yaml } from '@codemirror/lang-yaml';
import { EditorState } from '@codemirror/state';
import { EditorView } from '@codemirror/view';

export function SourceEditor({ text, onChange, jumpLine }: {
  text: string; onChange: (value: string) => void; jumpLine: number | null;
}) {
  const container = useRef<HTMLDivElement>(null);
  const editor = useRef<EditorView | null>(null);
  const callback = useRef(onChange);
  const updatingFromHost = useRef(false);
  callback.current = onChange;

  useEffect(() => {
    if (!container.current) return;
    const view = new EditorView({
      parent: container.current,
      state: EditorState.create({
        doc: text,
        extensions: [basicSetup, yaml(), EditorView.lineWrapping,
          EditorView.updateListener.of(update => {
            if (update.docChanged && !updatingFromHost.current) {
              callback.current?.(update.state.doc.toString());
            }
          })],
      }),
    });
    editor.current = view;
    return () => { editor.current = null; view.destroy(); };
  }, []);

  useEffect(() => {
    const view = editor.current;
    if (!view || view.state.doc.toString() === text) return;
    updatingFromHost.current = true;
  try {
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } });
  } finally {
    updatingFromHost.current = false;
  }
  }, [text]);

  useEffect(() => {
    const view = editor.current;
    if (!view || jumpLine === null) return;
    const line = view.state.doc.line(Math.max(1, Math.min(view.state.doc.lines, jumpLine)));
    view.dispatch({ selection: { anchor: line.from }, scrollIntoView: true });
    view.focus();
  }, [jumpLine]);

  return <div className="source-editor" ref={container} />;
}
