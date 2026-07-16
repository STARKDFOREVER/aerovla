"""AeroVLA 数据库模型 — SQLAlchemy + SQLite/PostgreSQL 兼容。"""
from datetime import datetime, timezone
from sqlalchemy import (Column, Integer, String, Float, Boolean, DateTime,
                        ForeignKey, Text, JSON, create_engine, event)
from sqlalchemy.orm import DeclarativeBase, relationship, Session, sessionmaker
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aerovla.db")
DB_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False}, echo=False)


# ---- SQLite 外键 PRAGMA ----
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


# ======== 用户表 ========
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(16), default="operator")  # admin / operator / viewer
    display_name = Column(String(128), default="")
    email = Column(String(128), default="")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime, nullable=True)

    missions = relationship("Mission", back_populates="creator", lazy="selectin")


# ======== 无人机表 ========
class Drone(Base):
    __tablename__ = "drones"
    id = Column(Integer, primary_key=True, autoincrement=True)
    drone_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(128), default="")
    model = Column(String(64), default="AeroVLA")
    firmware = Column(String(32), default="1.0.0")
    status = Column(String(16), default="offline")  # online/offline/flying/error
    battery = Column(Float, default=100.0)
    position_lat = Column(Float, default=0.0)
    position_lng = Column(Float, default=0.0)
    altitude = Column(Float, default=0.0)
    heading = Column(Float, default=0.0)
    speed = Column(Float, default=0.0)
    connected_since = Column(DateTime, nullable=True)
    last_telemetry = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    missions = relationship("Mission", back_populates="drone", lazy="selectin")
    logs = relationship("FlightLog", back_populates="drone", lazy="selectin")


# ======== 任务/航线表 ========
class Mission(Base):
    __tablename__ = "missions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(256), nullable=False)
    description = Column(Text, default="")
    mission_type = Column(String(32), default="waypoint")  # waypoint / search / patrol
    status = Column(String(16), default="draft")  # draft/planned/executing/completed/aborted
    waypoints = Column(JSON, default=list)  # [{lat, lng, alt, action, speed, heading}]
    vla_instruction = Column(Text, default="")  # open-ended VLA target description
    drone_id = Column(Integer, ForeignKey("drones.id"), nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    flight_time_s = Column(Float, default=0.0)
    distance_m = Column(Float, default=0.0)
    notes = Column(Text, default="")

    drone = relationship("Drone", back_populates="missions")
    creator = relationship("User", back_populates="missions")
    logs = relationship("FlightLog", back_populates="mission", lazy="selectin")


# ======== 飞行日志表 ========
class FlightLog(Base):
    __tablename__ = "flight_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    mission_id = Column(Integer, ForeignKey("missions.id"), nullable=True)
    drone_id = Column(Integer, ForeignKey("drones.id"), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    event_type = Column(String(32), default="telemetry")  # telemetry/vla_action/error/command
    data = Column(JSON, default=dict)
    position_lat = Column(Float, default=0.0)
    position_lng = Column(Float, default=0.0)
    altitude = Column(Float, default=0.0)
    heading = Column(Float, default=0.0)
    speed = Column(Float, default=0.0)
    battery = Column(Float, default=100.0)
    vla_fwd = Column(Float, nullable=True)
    vla_down = Column(Float, nullable=True)
    vla_yaw = Column(Float, nullable=True)
    vla_confidence = Column(Float, nullable=True)
    message = Column(Text, default="")

    drone = relationship("Drone", back_populates="logs")
    mission = relationship("Mission", back_populates="logs")


# ======== 初始化 ========
def init_db():
    Base.metadata.create_all(bind=engine)
    # 创建默认 admin 用户
    with SessionLocal() as db:
        if not db.query(User).filter(User.username == "admin").first():
            import bcrypt
            admin = User(
                username="admin",
                hashed_password=bcrypt.hashpw(b"admin888", bcrypt.gensalt()).decode(),
                role="admin",
                display_name="Administrator",
                email="admin@aerovla.local"
            )
            db.add(admin)
            db.commit()
