"""
AeroVLA Server — DJI FlightHub 风格地勤后端
用法: python aerovla_server.py [--host 0.0.0.0] [--port 6006] [--reload]
"""
import os, sys, time, json, asyncio, threading
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import (FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect,
                      status, Request, Query, File, Form)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel, Field
import uvicorn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aerovla_models import (SessionLocal, init_db, User, Drone, Mission, FlightLog)

# ======= 配置 =======
SECRET_KEY = os.environ.get("AEROVLA_SECRET", "aerovla-secret-key-change-in-production-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_H = 24  # 生产环境建议缩短到 2h + refresh token
pwd_context = None  # 用 bcrypt 代替 passlib
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# ======= WebSocket 客户端池 =======
_ws_clients: set[WebSocket] = set()
_latest_telemetry: dict = {}  # drone_id → telemetry dict


# ====== 生命周期 ======
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    asyncio.create_task(_broadcast_loop())
    # 启动时加载 VLA 引擎到 GPU（在后台线程中，避免阻塞 asyncio）
    import threading
    threading.Thread(target=_load_vla_engine, daemon=True).start()
    yield


app = FastAPI(title="AeroVLA Server", version="2.0.0",
              description="DJI FlightHub 风格地勤后端 — 无人机管理 / 任务规划 / 遥测推流 / 飞行日志",
              docs_url="/api/docs", redoc_url="/api/redoc", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ====== JWT ======
def create_token(data: dict, expires_h: int = ACCESS_TOKEN_EXPIRE_H):
    to_encode = data.copy()
    to_encode.update({"exp": datetime.now(timezone.utc) + timedelta(hours=expires_h)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()


def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username: raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError: raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User inactive")
    return user


def require_role(*roles: str):
    def checker(user=Depends(get_current_user)):
        if user.role not in roles:
            raise HTTPException(status_code=403, detail=f"Role {user.role} not allowed")
        return user
    return checker


# ====== Pydantic Schemas ======
class TokenOut(BaseModel):
    access_token: str; token_type: str = "bearer"; role: str; username: str

class UserInfo(BaseModel):
    username: str; role: str; display_name: str; email: str

class DroneCreate(BaseModel):
    drone_id: str = Field(..., min_length=1, max_length=64)
    name: str = ""; model: str = "AeroVLA"

class DroneOut(BaseModel):
    id: int; drone_id: str; name: str; model: str; firmware: str; status: str
    battery: float; position_lat: float; position_lng: float; altitude: float
    heading: float; speed: float; last_telemetry: Optional[str] = None

class Waypoint(BaseModel):
    lat: float; lng: float; alt: float = 50.0
    action: str = "fly_through"; speed: float = 5.0; heading: float = 0.0

class MissionCreate(BaseModel):
    name: str; description: str = ""
    mission_type: str = "waypoint"
    waypoints: list[Waypoint] = []
    vla_instruction: str = ""
    drone_id: Optional[int] = None

class MissionOut(BaseModel):
    id: int; name: str; status: str; mission_type: str
    waypoints: list = []; vla_instruction: str = ""
    drone_id: Optional[int]; created_by: Optional[int]
    created_at: Optional[str]; flight_time_s: float; distance_m: float

class FlightLogOut(BaseModel):
    id: int; mission_id: Optional[int]; drone_id: int
    timestamp: str; event_type: str; position_lat: float; position_lng: float
    altitude: float; heading: float; speed: float; battery: float
    vla_fwd: Optional[float]; vla_down: Optional[float]
    vla_yaw: Optional[float]; message: str

class TelemetryIn(BaseModel):
    drone_id: str; status: str = "online"; battery: float = 100.0
    position_lat: float = 0.0; position_lng: float = 0.0; altitude: float = 0.0
    heading: float = 0.0; speed: float = 0.0
    vla_fwd: Optional[float] = None; vla_down: Optional[float] = None
    vla_yaw: Optional[float] = None; vla_confidence: Optional[float] = None
    message: str = ""


# ====== 工具函数 ======
def _drone_out(d: Drone) -> dict:
    return {"id": d.id, "drone_id": d.drone_id, "name": d.name, "model": d.model,
            "firmware": d.firmware, "status": d.status, "battery": d.battery,
            "position_lat": d.position_lat, "position_lng": d.position_lng,
            "altitude": d.altitude, "heading": d.heading, "speed": d.speed,
            "last_telemetry": d.last_telemetry.isoformat() if d.last_telemetry else None}


# ==================== REST API ====================

# ---- Auth ----
@app.post("/api/auth/login", response_model=TokenOut)
def login(form: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not bcrypt.checkpw(form.password.encode(), user.hashed_password.encode()):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    user.last_login = datetime.now(timezone.utc); db.commit()
    token = create_token({"sub": user.username, "role": user.role})
    return TokenOut(access_token=token, role=user.role, username=user.username)


@app.get("/api/me")
def me(user=Depends(get_current_user)):
    return {"username": user.username, "role": user.role,
            "display_name": user.display_name or user.username, "email": user.email}


# ---- 用户管理 (admin only) ----
@app.get("/api/users")
def list_users(user=Depends(require_role("admin")), db=Depends(get_db)):
    return [{"id": u.id, "username": u.username, "role": u.role,
             "display_name": u.display_name, "is_active": u.is_active,
             "last_login": u.last_login.isoformat() if u.last_login else ""}
            for u in db.query(User).all()]


# ---- 无人机管理 ----
@app.get("/api/drones")
def list_drones(user=Depends(get_current_user), db=Depends(get_db)):
    return [_drone_out(d) for d in db.query(Drone).all()]


@app.post("/api/drones")
def register_drone(dc: DroneCreate, user=Depends(require_role("admin", "operator")), db=Depends(get_db)):
    if db.query(Drone).filter(Drone.drone_id == dc.drone_id).first():
        raise HTTPException(status_code=409, detail="Drone already registered")
    d = Drone(drone_id=dc.drone_id, name=dc.name or dc.drone_id, model=dc.model)
    db.add(d); db.commit(); db.refresh(d)
    return _drone_out(d)


@app.get("/api/drones/{drone_id}")
def get_drone(drone_id: int, user=Depends(get_current_user), db=Depends(get_db)):
    d = db.query(Drone).filter(Drone.id == drone_id).first()
    if not d: raise HTTPException(status_code=404, detail="Drone not found")
    return _drone_out(d)


@app.put("/api/drones/{drone_id}/status")
def update_drone_status(drone_id: int, status: str = Query(...),
                         user=Depends(require_role("admin", "operator")), db=Depends(get_db)):
    d = db.query(Drone).filter(Drone.id == drone_id).first()
    if not d: raise HTTPException(status_code=404)
    d.status = status; db.commit()
    return {"ok": True, "status": status}


# ---- 遥测上报 (供 AirSim 控制器调用) ----
@app.post("/api/telemetry/ingest")
def ingest_telemetry(telem: TelemetryIn, db=Depends(get_db)):
    global _latest_telemetry
    d = db.query(Drone).filter(Drone.drone_id == telem.drone_id).first()
    if not d:
        d = Drone(drone_id=telem.drone_id, name=telem.drone_id)
        db.add(d); db.flush()
    d.status = telem.status; d.battery = telem.battery
    d.position_lat = telem.position_lat; d.position_lng = telem.position_lng
    d.altitude = telem.altitude; d.heading = telem.heading; d.speed = telem.speed
    d.last_telemetry = datetime.now(timezone.utc)
    if telem.status == "online" and not d.connected_since:
        d.connected_since = datetime.now(timezone.utc)
    # 写入飞行日志
    log = FlightLog(drone_id=d.id, event_type=telem.status if telem.status != "online" else "telemetry",
                    position_lat=telem.position_lat, position_lng=telem.position_lng,
                    altitude=telem.altitude, heading=telem.heading, speed=telem.speed,
                    battery=telem.battery, vla_fwd=telem.vla_fwd, vla_down=telem.vla_down,
                    vla_yaw=telem.vla_yaw, vla_confidence=telem.vla_confidence, message=telem.message)
    db.add(log); db.commit()
    # 更新缓存
    _latest_telemetry[telem.drone_id] = telem.model_dump()
    _latest_telemetry[telem.drone_id]["updated"] = time.time()
    return {"ok": True}


# ---- 任务管理 ----
@app.get("/api/missions")
def list_missions(status: str = Query(None), user=Depends(get_current_user), db=Depends(get_db)):
    q = db.query(Mission)
    if status: q = q.filter(Mission.status == status)
    return [{"id": m.id, "name": m.name, "status": m.status, "mission_type": m.mission_type,
             "waypoints": m.waypoints or [], "vla_instruction": m.vla_instruction,
             "drone_id": m.drone_id, "created_by": m.created_by,
             "created_at": m.created_at.isoformat() if m.created_at else "",
             "flight_time_s": m.flight_time_s, "distance_m": m.distance_m,
             "description": m.description or ""}
            for m in q.order_by(Mission.created_at.desc()).all()]


@app.post("/api/missions")
def create_mission(mc: MissionCreate, user=Depends(require_role("admin", "operator")), db=Depends(get_db)):
    m = Mission(name=mc.name, description=mc.description, mission_type=mc.mission_type,
                waypoints=[w.model_dump() for w in mc.waypoints],
                vla_instruction=mc.vla_instruction, drone_id=mc.drone_id, created_by=user.id)
    if mc.drone_id and not db.query(Drone).filter(Drone.id == mc.drone_id).first():
        raise HTTPException(status_code=404, detail="Drone not found")
    db.add(m); db.commit(); db.refresh(m)
    return {"id": m.id, "name": m.name, "status": m.status}


@app.put("/api/missions/{mission_id}/status")
def update_mission_status(mission_id: int, status: str = Query(...),
                           user=Depends(require_role("admin", "operator")), db=Depends(get_db)):
    m = db.query(Mission).filter(Mission.id == mission_id).first()
    if not m: raise HTTPException(status_code=404)
    m.status = status
    if status == "executing": m.started_at = datetime.now(timezone.utc)
    elif status in ("completed", "aborted"): m.completed_at = datetime.now(timezone.utc)
    db.commit()
    return {"ok": True, "status": status}


# ---- 飞行日志 ----
@app.get("/api/flight-logs")
def list_flight_logs(drone_id: int = Query(None), mission_id: int = Query(None),
                      limit: int = Query(100, le=1000), offset: int = 0,
                      user=Depends(get_current_user), db=Depends(get_db)):
    q = db.query(FlightLog)
    if drone_id: q = q.filter(FlightLog.drone_id == drone_id)
    if mission_id: q = q.filter(FlightLog.mission_id == mission_id)
    total = q.count()
    rows = q.order_by(FlightLog.timestamp.desc()).offset(offset).limit(limit).all()
    return {"total": total, "logs": [
        {"id": r.id, "mission_id": r.mission_id, "drone_id": r.drone_id,
         "timestamp": r.timestamp.isoformat(), "event_type": r.event_type,
         "position_lat": r.position_lat, "position_lng": r.position_lng,
         "altitude": r.altitude, "heading": r.heading, "speed": r.speed,
         "battery": r.battery, "vla_fwd": r.vla_fwd, "vla_down": r.vla_down,
         "vla_yaw": r.vla_yaw, "vla_confidence": r.vla_confidence, "message": r.message or ""}
        for r in rows]}


# ====== VLA 推理引擎 ======
# RTX 4080 (16GB): 全精度 bf16, ~14.5GB 显存，刚好够用
# 环境变量: AEROVLA_BASE_MODEL, AEROVLA_LORA (可选, 默认用代码内置路径)
_vla_engine = None


def _load_vla_engine():
    """启动时加载 VLA 推理引擎到 GPU。4080 全精度 bf16，一次性加载，常驻显存。"""
    global _vla_engine
    import cv2
    base_path = os.environ.get("AEROVLA_BASE_MODEL", r"D:\aerovla-server\openvla-7b")
    lora_path = os.environ.get("AEROVLA_LORA", os.path.join(base_path, "weight-lora", "aerial_vla"))
    print(f"[SERVER] Loading VLA engine from: {base_path}")
    print(f"[SERVER] LoRA from: {lora_path}")
    print(f"[SERVER] 4080 全精度 bf16 — 无需 bitsandbytes")
    try:
        from aerovla_inference import AeroVLAInference
        _vla_engine = AeroVLAInference(
            base_model_path=base_path,
            lora_path=lora_path,
            load_in_4bit=False,
            load_in_8bit=False,   # 4080 全精度 bf16
            log_fn=print,
        )
        print("[SERVER] VLA engine loaded. Ready for inference.")
    except Exception as e:
        print(f"[SERVER] VLA engine load failed: {e}")
        _vla_engine = None


# ---- 健康检查 ----
@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "ws_clients": len(_ws_clients),
            "drones_online": len(_latest_telemetry),
            "model_loaded": _vla_engine is not None,
            "error": None if _vla_engine else "VLA engine not loaded"}


# ---- VLA 推理 ----
@app.post("/infer")
async def vla_infer(
    front_image: bytes = File(...),
    down_image: bytes = File(...),
    instruction: str = Form(""),
    direction_hint: str = Form(""),
):
    """VLA 推理端点。接收双视角 JPEG 图像和语言指令，返回 3-DoF 控制信号。

    请求: POST /infer
      multipart/form-data:
        front_image: JPEG 图像 (前视摄像头)
        down_image:  JPEG 图像 (下视摄像头)
        instruction: "a red car parked on the street"
        direction_hint: "straight ahead"

    响应: {"fwd": 2.3, "down": 0.0, "yaw": 0.1, "land": false, "infer_time": 3.94}
    """
    if _vla_engine is None:
        raise HTTPException(status_code=503, detail="VLA engine not loaded")
    import cv2, numpy as np

    # 解码图像
    front = cv2.imdecode(np.frombuffer(front_image, np.uint8), cv2.IMREAD_COLOR)
    down = cv2.imdecode(np.frombuffer(down_image, np.uint8), cv2.IMREAD_COLOR)
    if front is None or down is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # 推理
    t0 = time.time()
    try:
        result = _vla_engine.infer(
            front_img=front,
            down_img=down,
            instruction=instruction,
            direction_hint=direction_hint,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    infer_time = time.time() - t0
    return {
        "fwd": result.get("fwd", 0.0),
        "down": result.get("down", 0.0),
        "yaw": result.get("yaw", 0.0),
        "land": result.get("land", False),
        "infer_time": round(infer_time, 3),
    }


# ---- 前端入口 ----
FRONTEND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aerovla_webui.html")


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    if os.path.exists(FRONTEND_PATH):
        with open(FRONTEND_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>AeroVLA Server Running</h1><p>Frontend not found.</p>")


# ==================== WebSocket 遥测推流 ====================
async def _broadcast_loop():
    """每 1s 广播最新遥测给所有 WebSocket 客户端。"""
    while True:
        if _ws_clients and _latest_telemetry:
            payload = json.dumps({"type": "telemetry", "data": _latest_telemetry,
                                  "timestamp": time.time()}, ensure_ascii=False)
            dead = set()
            for ws in _ws_clients:
                try: await ws.send_text(payload)
                except: dead.add(ws)
            _ws_clients.difference_update(dead)
        await asyncio.sleep(1)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "")
            if msg_type == "telemetry":
                # 客户端也可以上报遥测
                _latest_telemetry[data.get("drone_id", "unknown")] = data.get("data", {})
            elif msg_type == "command":
                # 写入 command.json 给飞控
                try:
                    sd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vla_shared")
                    os.makedirs(sd, exist_ok=True)
                    json.dump(data.get("data", {}),
                              open(os.path.join(sd, "command.json"), "w", encoding="utf-8"),
                              indent=2, ensure_ascii=False)
                except: pass
            elif msg_type == "ping":
                await ws.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _ws_clients.discard(ws)


# ======= main =======
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    print(f"[AeroVLA Server] http://{args.host}:{args.port}")
    print(f"[AeroVLA Server] API docs: http://localhost:{args.port}/api/docs")
    print(f"[AeroVLA Server] Frontend: http://localhost:{args.port}/")
    uvicorn.run("aerovla_server:app", host=args.host, port=args.port, reload=args.reload)
