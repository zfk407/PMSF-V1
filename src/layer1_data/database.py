"""
PMSF_DLT_HISTORY 大乐透历史数据库管理
基于 SQLite，支持期号、开奖号码、衍生特征存储
"""
import sqlite3
import os
import pandas as pd
from datetime import datetime


class DltDatabase:
    """大乐透历史数据库"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_tables()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        """初始化数据表"""
        conn = self._get_conn()
        cursor = conn.cursor()
        # 原始开奖表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dlt_history (
                issue TEXT PRIMARY KEY,
                date TEXT,
                front01 INTEGER, front02 INTEGER, front03 INTEGER,
                front04 INTEGER, front05 INTEGER,
                back01 INTEGER, back02 INTEGER,
                sum_front INTEGER,
                span_front INTEGER,
                odd_even TEXT,
                big_small TEXT,
                zone TEXT,
                tail TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # 特征表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dlt_features (
                issue TEXT,
                number INTEGER,
                feature_json TEXT,
                PRIMARY KEY (issue, number)
            )
        """)
        # 状态表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dlt_states (
                issue TEXT PRIMARY KEY,
                state_a_prob REAL,
                state_b_prob REAL,
                state_c_prob REAL,
                predicted_state TEXT,
                actual_state TEXT
            )
        """)
        # 回测结果表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue TEXT,
                group_label TEXT,
                predicted_front TEXT,
                predicted_back TEXT,
                actual_front TEXT,
                actual_back TEXT,
                front_hit INTEGER,
                back_hit INTEGER,
                coverage REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def insert_draw(self, record: dict):
        """插入单期开奖记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cols = ["issue", "date", "front01", "front02", "front03",
                "front04", "front05", "back01", "back02",
                "sum_front", "span_front", "odd_even", "big_small",
                "zone", "tail"]
        placeholders = ",".join(["?"] * len(cols))
        values = [record.get(c) for c in cols]
        cursor.execute(
            f"INSERT OR REPLACE INTO dlt_history ({','.join(cols)}) VALUES ({placeholders})",
            values
        )
        conn.commit()
        conn.close()

    def insert_batch(self, df: pd.DataFrame):
        """批量插入开奖数据"""
        conn = self._get_conn()
        df.to_sql("dlt_history", conn, if_exists="replace", index=False)
        conn.commit()
        conn.close()

    def load_all(self) -> pd.DataFrame:
        """加载全部历史数据，按期号升序"""
        conn = self._get_conn()
        df = pd.read_sql("SELECT * FROM dlt_history ORDER BY issue ASC", conn)
        conn.close()
        return df

    def load_by_issue_range(self, start_issue: str, end_issue: str) -> pd.DataFrame:
        """按期号范围加载"""
        conn = self._get_conn()
        df = pd.read_sql(
            "SELECT * FROM dlt_history WHERE issue BETWEEN ? AND ? ORDER BY issue ASC",
            conn, params=(start_issue, end_issue)
        )
        conn.close()
        return df

    def get_latest_issue(self) -> str:
        """获取最新期号"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(issue) FROM dlt_history")
        result = cursor.fetchone()[0]
        conn.close()
        return result

    def count(self) -> int:
        """记录数"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM dlt_history")
        result = cursor.fetchone()[0]
        conn.close()
        return result

    def save_state(self, issue: str, state_probs: dict, predicted_state: str):
        """保存状态预测结果"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO dlt_states
            (issue, state_a_prob, state_b_prob, state_c_prob, predicted_state)
            VALUES (?, ?, ?, ?, ?)
        """, (issue, state_probs.get("A", 0), state_probs.get("B", 0),
              state_probs.get("C", 0), predicted_state))
        conn.commit()
        conn.close()

    def save_backtest(self, issue: str, group_label: str,
                      pred_front: list, pred_back: list,
                      actual_front: list, actual_back: list):
        """保存回测记录"""
        front_hit = len(set(pred_front) & set(actual_front))
        back_hit = len(set(pred_back) & set(actual_back))
        coverage = front_hit / 5.0
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO backtest_results
            (issue, group_label, predicted_front, predicted_back,
             actual_front, actual_back, front_hit, back_hit, coverage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (issue, group_label,
              ",".join(map(str, pred_front)),
              ",".join(map(str, pred_back)),
              ",".join(map(str, actual_front)),
              ",".join(map(str, actual_back)),
              front_hit, back_hit, coverage))
        conn.commit()
        conn.close()
