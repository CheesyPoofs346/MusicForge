@echo off
rem MusicForge: local server + Cloudflare tunnel (URL prints below)
start "musicforge-server" /min cmd /c "python -m uvicorn server:app --app-dir C:\musicforge --port 8137"
cloudflared tunnel --url http://127.0.0.1:8137
