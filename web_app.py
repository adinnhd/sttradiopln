"""Web recorder lokal. Jalankan: python web_app.py (satu proses)."""

from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
import tempfile
from typing import Literal
from zipfile import ZIP_STORED, ZipFile

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from radio_recorder.live import LiveRecorder


ROOT = Path(__file__).resolve().parent


class StartRequest(BaseModel):
    device_id: int = Field(ge=0)
    channels: int | None = Field(default=None, ge=1, le=2)


class RecordingSelection(BaseModel):
    status: Literal["speech", "uncertain", "non-speech", "pending", "all", "trash"] = "speech"
    recorded_date: date | None = None
    identifiers: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    select_all: bool = False
    expected_count: int = Field(ge=1)


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
                   status: Literal["speech", "uncertain", "non-speech", "pending", "all", "trash"] = "speech",
                   page: int = Query(default=1, ge=1),
                   page_size: int = Query(default=20, ge=1, le=100),
                   recorded_date: date | None = Query(default=None, alias="date")):
        return request.app.state.recorder.store.listing(
            status, limit=page_size, page=page, recorded_date=recorded_date)

    @app.post("/api/recordings/trash")
    def trash_recordings(body: RecordingSelection, request: Request):
        if body.status == "trash":
            raise HTTPException(422, "Rekaman sudah berada di Sampah")
        try:
            count = request.app.state.recorder.store.set_trashed(**body.model_dump())
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"count": count}

    @app.post("/api/recordings/restore")
    def restore_recordings(body: RecordingSelection, request: Request):
        if body.status != "trash":
            raise HTTPException(422, "Pemulihan hanya dari Sampah")
        try:
            count = request.app.state.recorder.store.set_trashed(restore=True, **body.model_dump())
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"count": count}

    @app.get("/api/recordings/download")
    def download_recordings(
            request: Request,
            status: Literal["speech", "uncertain", "non-speech", "pending", "all", "trash"] = "speech",
            recorded_date: date | None = Query(default=None, alias="date"),
            identifiers: list[str] | None = Query(default=None, alias="id"),
            excluded: list[str] | None = Query(default=None, alias="exclude")):
        try:
            files = request.app.state.recorder.store.archive_files(
                status, recorded_date, identifiers, excluded)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        if not files:
            raise HTTPException(404, "Tidak ada rekaman sesuai filter")

        def chunks():
            # Header unduhan dikirim sebelum ZIP besar selesai disiapkan. Arsip
            # kecil berada di RAM dan arsip besar otomatis memakai file sementara.
            archive = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)
            try:
                with ZipFile(archive, "w", compression=ZIP_STORED,
                             allowZip64=True) as output:
                    for item in files:
                        started = (item["started_at"].replace(":", "-")
                                   .replace("+", "_"))
                        member = (f"{item['status']}_{started}_"
                                  f"{item['id'][:8]}.wav")
                        output.write(item["path"], member)
                archive.seek(0)
                while data := archive.read(1024 * 1024):
                    yield data
            finally:
                archive.close()

        scope = recorded_date.isoformat() if recorded_date else "semua-tanggal"
        filename = f"rekaman_{status}_{scope}.zip"
        return StreamingResponse(
            chunks(), media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'})

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
