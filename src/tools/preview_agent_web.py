"""Read-only web preview; never starts camera, robot worker, or control routes.

SO101_PREVIEW_DATA_ROOT=/path/to/datasets/agent PYTHONPATH=src \
  python -m uvicorn tools.preview_agent_web:app --host 0.0.0.0 --port 8110
"""
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from agent.episodes import list_runs, list_episodes, video_file

WEB = Path(__file__).resolve().parents[1] / "agent/web"
DATA = Path(os.environ.get("SO101_PREVIEW_DATA_ROOT", "datasets/agent")).resolve()
ASSETS = {"app.css", "shadcn.css", "episodes.css", "episodes.js", "preview.js"}
app = FastAPI(title="SO-101 read-only UI preview")


@app.get("/")
def index():
    html = (WEB / "index.html").read_text()
    # The live app owns the robot bus. The preview runs only local UI behavior.
    html = re.sub(r"<script src=\"[^\"]+\"></script>", "", html)
    html = html.replace('<div class="status">', '<div class="status"><span class="badge badge-muted" role="status">읽기 전용 미리보기</span>', 1)
    html = html.replace('id="camera-missing" class="camera-missing" hidden=""', 'id="camera-missing" class="camera-missing"')
    html = html.replace("카메라 영상을 불러올 수 없습니다 (so101-camera 실행 확인)", "임시 UI 미리보기 · 카메라 미연결")
    for name, marker in (("app.css", "__APP_CSS_VERSION__"), ("shadcn.css", "__SHADCN_CSS_VERSION__"),
                         ("episodes.css", "__EPISODES_CSS_VERSION__")):
        html = html.replace(marker, str((WEB / name).stat().st_mtime_ns))
    html = html.replace("</body>", '<script src="/episodes.js"></script><script src="/preview.js"></script></body>', 1)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/episodes")
def episodes():
    return RedirectResponse("/?tab=episodes", status_code=307)


@app.get("/{asset}")
def asset(asset: str):
    if asset not in ASSETS:
        raise HTTPException(404)
    return FileResponse(WEB / asset, headers={"Cache-Control": "no-store"})


@app.get("/api/episodes")
def runs():
    return {"runs": list_runs(DATA)}


@app.get("/api/episodes/{run_id}")
def episode_list(run_id: str):
    try:
        return {"episodes": list_episodes(DATA, run_id)}
    except (ValueError, FileNotFoundError):
        raise HTTPException(404) from None


@app.get("/api/episodes/{run_id}/{episode_index}/video/{key}")
def video(run_id: str, episode_index: int, key: str):
    try:
        path = video_file(DATA, run_id, episode_index, key)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404) from None
    return FileResponse(path, media_type="video/mp4")
