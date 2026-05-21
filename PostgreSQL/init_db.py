import os
import time
from sqlalchemy import create_engine
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import connect
from models import Base

# Environment setup
DB_USER = os.getenv("POSTGRES_USER", "user")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME = os.getenv("POSTGRES_DB", "analysis_db")
SOCKET_PATH = "/var/run/postgresql"

# FIXED: Removed the space after the @/
SQLALCHEMY_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASS}@/{DB_NAME}?host={SOCKET_PATH}"
PSYC_URI = f"postgresql://{DB_USER}:{DB_PASS}@/{DB_NAME}?host={SOCKET_PATH}"

def init_db():
    engine = create_engine(SQLALCHEMY_URL)
    for i in range(10):
        try:
            Base.metadata.create_all(bind=engine)
            print("Custom tables created successfully!")
            return
        except Exception as e:
            print(f"Waiting for Socket... attempt {i + 1}/10 | Error: {e}")
            time.sleep(1)

def init_langgraph():
    try:
        with connect(PSYC_URI, autocommit=True) as conn:
            checkpointer = PostgresSaver(conn)
            checkpointer.setup()
            print("LangGraph tables created!")
    except Exception as e:
        print(f"LangGraph setup failed: {e}")

if __name__ == "__main__":
    init_db()
    init_langgraph()