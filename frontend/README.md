# Citefold Web

The React/TypeScript client for Citefold's supported browser workflow. It talks
to the local FastAPI backend for source import, persisted task progress,
grounded chat, citation inspection, and notes.

Run commands from `frontend/`:

```powershell
npm.cmd ci
npm.cmd run dev
```

Quality gates:

```powershell
npm.cmd test -- --run
npm.cmd run lint
npm.cmd run build
```

The Tauri directory is retained as an unsupported desktop preview. See the
root README for the product boundary and full local setup.
