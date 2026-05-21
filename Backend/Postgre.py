import os
import json
import psycopg
from datetime import datetime, timezone
from typing import List, Optional
from URLs import DEFAULT_POSTGRESQL_STRING


class PostgreSQLConnector:
    def __init__(self, conn_string: Optional[str] = None):
        self.conn_string = conn_string or os.getenv("DATABASE_URL", DEFAULT_POSTGRESQL_STRING)
        self.conn = psycopg.connect(self.conn_string)

    def add_session(self, thread_id: str, status: str = "analyzing", current_page: int = 0,
                    report_path: Optional[str] = None):
        query = """
        INSERT INTO analysis_sessions (thread_id, status, current_page, created_at, report_path)
        VALUES (%(thread_id)s, %(status)s, %(current_page)s, %(created_at)s, %(report_path)s)
        ON CONFLICT (thread_id) DO NOTHING;
        """

        data = {
            "thread_id": thread_id,
            "status": status,
            "current_page": current_page,
            "created_at": datetime.now(timezone.utc),
            "report_path": report_path
        }

        with self.conn.cursor() as cur:
            cur.execute(query, data)
            self.conn.commit()

    def add_upload(self, thread_id: str, minio_path: str, artifact_type: str = "ORIGINAL_CSV"):
        query = """
        INSERT INTO intermediate_files (thread_id, artifact_type, minio_path)
        VALUES (%s, %s, %s);
        """

        with self.conn.cursor() as cur:
            cur.execute(query, (thread_id, artifact_type, minio_path))
            self.conn.commit()

    def delete_uploads(self, thread_id: str):
        query = "DELETE FROM intermediate_files WHERE thread_id = %s;"

        with self.conn.cursor() as cur:
            cur.execute(query, (thread_id,))
            self.conn.commit()

    def add_insight(
            self,
            thread_id: str,
            title: str,
            chart_paths: List[str],
            code: str,
            code_output: str,
            summary: str,
            documents_retrieved: Optional[List[str]] = None
    ):
        query = """
        INSERT INTO insight_metadata (thread_id, title, chart_path, code, code_output, summary, documents_retrieved)
        VALUES (%(thread_id)s, %(title)s, %(chart_path)s, %(code)s, %(code_output)s, %(summary)s, %(documents_retrieved)s);
        """

        data = {
            "thread_id": thread_id,
            "title": title,
            "chart_path": json.dumps(chart_paths),
            "code": code,
            "code_output": code_output,
            "summary": summary,
            "documents_retrieved": json.dumps(documents_retrieved) if documents_retrieved else json.dumps([])
        }

        with self.conn.cursor() as cur:
            cur.execute(query, data)
            self.conn.commit()

    def get_insight(self, thread_id: str, title: str):
        query = """
        SELECT thread_id, title, chart_path, code, code_output, summary, documents_retrieved 
        FROM insight_metadata 
        WHERE thread_id = %s AND title = %s;
        """

        with self.conn.cursor() as cur:
            cur.execute(query, (thread_id, title))
            row = cur.fetchone()

            if not row:
                return None

            return {
                "thread_id": row[0],
                "title": row[1],
                "plots": row[2] if row[2] is not None else [],
                "code_generated": row[3],
                "code_output": row[4],
                "final_inference": row[5],
                "documents_retrieved": row[6] if row[6] is not None else []
            }

    def get_report_insights(self, thread_id: str):
        query = """
        SELECT thread_id, title, chart_path, code, code_output, summary, documents_retrieved 
        FROM insight_metadata 
        WHERE thread_id = %s;
        """

        with self.conn.cursor() as cur:
            cur.execute(query, (thread_id,))
            rows = cur.fetchall()

            if not rows:
                return None

            insight_data = []

            for row in rows:
                obj = {
                    "title": row[1],
                    "plots": row[2] if row[2] is not None else [],
                    "code_generated": row[3],
                    "code_output": row[4],
                    "final_inference": row[5],
                    "documents_retrieved": row[6] if row[6] is not None else []
                }
                insight_data.append(obj)

            return insight_data

    def get_uploads(self, thread_id: str):
        query = "SELECT minio_path FROM intermediate_files WHERE thread_id = %s;"

        with self.conn.cursor() as cur:
            cur.execute(query, (thread_id,))
            records = cur.fetchall()
            return [row[0] for row in records]

    def get_insight_titles(self, thread_id: str):
        query = """
        SELECT title 
        FROM insight_metadata 
        WHERE thread_id = %s;
        """

        with self.conn.cursor() as cur:
            cur.execute(query, (thread_id,))
            records = cur.fetchall()
            return [row[0] for row in records]

    def reset_session_data(self, thread_id: str):
        queries = [
            "DELETE FROM insight_metadata WHERE thread_id = %s;",
            "DELETE FROM intermediate_files WHERE thread_id = %s;",
            "DELETE FROM analysis_sessions WHERE thread_id = %s;"
        ]

        try:
            with self.conn.cursor() as cur:
                for query in queries:
                    cur.execute(query, (thread_id,))
                self.conn.commit()
            print(f"Successfully purged relational DB data matrices for thread {thread_id}")
        except Exception as e:
            print(f"Error resetting relational session data for thread {thread_id}: {e}")
            raise e

    def close(self):
        if self.conn:
            self.conn.close()