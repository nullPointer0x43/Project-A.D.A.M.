from datetime import datetime
from typing import List, Optional
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    thread_id = Column(String, primary_key=True)
    status = Column(String, default="initializing")
    current_page = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    report_path = Column(String, nullable=True)


class IntermediateFile(Base):
    __tablename__ = "intermediate_files"

    artifact_id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String, ForeignKey("analysis_sessions.thread_id"))
    artifact_type = Column(String)
    minio_path = Column(String)


class InsightMetadata(Base):
    __tablename__ = "insight_metadata"

    insight_id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String, ForeignKey("analysis_sessions.thread_id"))
    title = Column(String)
    chart_path = Column(JSON)
    code = Column(Text)
    code_output = Column(Text)
    summary = Column(Text)
    documents_retrieved = Column(JSON, nullable=True)
