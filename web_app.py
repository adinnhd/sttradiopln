"""Web recorder lokal. Jalankan: python web_app.py (satu proses)."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from radio_recorder.live import LiveRecorder


ROOT = Path(__file__).resolve().parent


class StartRequest(BaseModel):
    device_id: int = Field(ge=0)
    channels: int = Field(default=1, ge=1, le=2)


def create_app(recorder=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.recorder = recorder if recorder is not None else LiveRecorder(
            ROOT / "data" / "recordings")
        try:
            yield
        finally:
            app.state.recorder.close()

    app = FastAPI(title="Radio PLN", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware,
                       allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_control(request: Request, call_next):
        # Browser dari situs lain tidak boleh menyalakan mikrofon lokal.
        if request.method == "POST":
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "Kontrol hanya dari halaman lokal"}, 403)
            if request.headers.get("x-recorder-control") != "1":
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": "Header kontrol diperlukan"}, 403)
        return await call_next(request)

    @app.get("/")
    def index():
        return FileResponse(ROOT / "web" / "index.html")

    @app.get("/api/devices")
    def devices(request: Request):
        try:
            return {"items": request.app.state.recorder.devices()}
        except Exception as exc:
            raise HTTPException(503, f"Tidak dapat membaca perangkat audio: {exc}") from exc

    @app.get("/api/status")
    def status(request: Request):
        return request.app.state.recorder.status()

    @app.post("/api/start")
    def start(body: StartRequest, request: Request):
        try:
            return request.app.state.recorder.start(body.device_id, body.channels)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, f"Perangkat gagal dibuka: {exc}") from exc

    @app.post("/api/stop")
    def stop(request: Request):
        try:
            return request.app.state.recorder.stop()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/recordings")
    def recordings(request: Request,
                   status: Literal["speech", "uncertain", "non-speech", "pending", "all"] = "speech"):
        return request.app.state.recorder.store.listing(status)

    @app.get("/api/recordings/{identifier}/audio")
    def audio(identifier: str, request: Request):
        try:
            path = request.app.state.recorder.store.audio_path(identifier)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return FileResponse(path, media_type="audio/wav")

    app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
