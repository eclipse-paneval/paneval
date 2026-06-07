import sqlite3
import threading
from typing import Optional
from contextlib import contextmanager

import json
import os
import hashlib
from typing import Any
from flagevalmm.common.logger import get_logger

logger = get_logger(__name__)


def calculate_hash(data: Any) -> str:
    string_representation = json.dumps(data, sort_keys=True)

    sha256 = hashlib.sha256()
    sha256.update(string_representation.encode("utf-8"))
    return sha256.hexdigest()


class ModelCache:
    def __init__(self, db_name="model_cache", cache_dir="./.cache"):
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        db_name = db_name.replace("/", "_")
        self.db_path = os.path.join(cache_dir, f"{db_name}.sqlite")
        self._local = threading.local()  # Thread-local storage
        self._lock = threading.Lock()  # Lock for initialization

        # Ensure database and table are created
        with self._get_conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS cache
                          (question TEXT PRIMARY KEY, answer TEXT)"""
            )
            conn.commit()

    @contextmanager
    def _get_conn(self):
        """Get a thread-safe database connection"""
        if not hasattr(self._local, "conn"):
            with self._lock:
                self._local.conn = sqlite3.connect(self.db_path)
                # Enable foreign key constraints
                self._local.conn.execute("PRAGMA foreign_keys = ON")

        try:
            yield self._local.conn
        except sqlite3.Error as e:
            # If a database error occurs, close and clean up the connection
            if hasattr(self._local, "conn"):
                self._local.conn.close()
                delattr(self._local, "conn")
            raise RuntimeError(f"Database error: {str(e)}")

    def insert(self, question: str, answer: str) -> None:
        """Thread-safe insert operation"""
        question_hash = calculate_hash(question)
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (question, answer) VALUES (?, ?)",
                    (question_hash, answer),
                )
                conn.commit()
        except Exception as e:
            raise RuntimeError(f"Failed to insert into cache: {str(e)}")

    def get(self, question: str) -> Optional[str]:
        """Thread-safe query operation"""
        question_hash = calculate_hash(question)
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT answer FROM cache WHERE question = ?", (question_hash,)
                )
                result = cursor.fetchone()
                if result is None:
                    return None
                answer: str = result[0]
                return answer
        except Exception as e:
            raise RuntimeError(f"Failed to get from cache: {str(e)}")

    def clear(self) -> None:
        """Clear the cache"""
        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM cache")
                conn.commit()
        except Exception as e:
            raise RuntimeError(f"Failed to clear cache: {str(e)}")

    def close(self) -> None:
        """Close the database connection for the current thread"""
        if hasattr(self._local, "conn"):
            try:
                self._local.conn.close()
                delattr(self._local, "conn")
            except Exception as e:
                raise RuntimeError(f"Failed to close connection: {str(e)}")

    def exists(self, question: str) -> bool:
        """Check if a question exists in the cache"""
        question_hash = calculate_hash(question)
        try:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT 1 FROM cache WHERE question = ? LIMIT 1", (question_hash,)
                )
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check cache existence: {str(e)}")
            return False

    def delete(self, question: str) -> None:
        question_hash = calculate_hash(question)
        try:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM cache WHERE question = ?", (question_hash,))
                conn.commit()
        except Exception as e:
            raise RuntimeError(f"Failed to delete from cache: {str(e)}")

    def __del__(self):
        try:
            self.close()
        except BaseException:
            pass  # Ignore cleanup errors


# Example usage
if __name__ == "__main__":
    cache = ModelCache()

    question = "What is the capital of France?"
    answer = "Paris"

    # Insert into cache
    cache.insert(question, answer)
    print(cache.exists(question))
    # Retrieve from cache
    cached_answer = cache.get(question)
    print(f"Cached answer: {cached_answer}")

    # non-existent question
    non_existent_question = "What is the capital of Mars?"
    print(cache.exists(non_existent_question))
