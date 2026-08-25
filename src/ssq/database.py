"""
双色球历史数据库管理
基于 SQLite，存储双色球开奖数据与衍生特征
"""
import sqlite3
import os
import pandas as pd


class SsqDatabase:
    """双色球历史数据库"""

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
        # 双色球原始开奖表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ssq_history (
                issue TEXT PRIMARY KEY,
                date TEXT,
                red01 INTEGER, red02 INTEGER, red03 INTEGER,
                red04 INTEGER, red05 INTEGER, red06 INTEGER,
                blue INTEGER,
                sum_red INTEGER,
                span_red INTEGER,
                odd_even TEXT,
                big_small TEXT,
                zone TEXT,
                tail TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # 双色球预测记录表（复盘用）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ssq_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                issue TEXT,
                group_label TEXT,
                predicted_red TEXT,
                predicted_blue INTEGER,
                actual_red TEXT,
                actual_blue INTEGER,
                red_hit INTEGER,
                blue_hit INTEGER,
                prize_level INTEGER,
                prize_name TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def insert_batch(self, df: pd.DataFrame):
        """批量插入开奖数据"""
        conn = self._get_conn()
        df.to_sql("ssq_history", conn, if_exists="replace", index=False)
        conn.commit()
        conn.close()

    def load_all(self) -> pd.DataFrame:
        """加载全部历史数据，按期号升序"""
        conn = self._get_conn()
        df = pd.read_sql("SELECT * FROM ssq_history ORDER BY issue ASC", conn)
        conn.close()
        return df

    def get_latest_issue(self) -> str:
        """获取最新期号"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(issue) FROM ssq_history")
        result = cursor.fetchone()[0]
        conn.close()
        return result

    def count(self) -> int:
        """记录数"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ssq_history")
        result = cursor.fetchone()[0]
        conn.close()
        return result

    def save_prediction(self, issue: str, group_label: str,
                        pred_red: list, pred_blue: int,
                        actual_red: list = None, actual_blue: int = None,
                        red_hit: int = None, blue_hit: int = None):
        """保存预测/复盘记录"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO ssq_predictions
            (issue, group_label, predicted_red, predicted_blue,
             actual_red, actual_blue, red_hit, blue_hit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (issue, group_label,
              ",".join(map(str, pred_red)),
              pred_blue,
              ",".join(map(str, actual_red)) if actual_red else None,
              actual_blue,
              red_hit if red_hit is not None else 0,
              blue_hit if blue_hit is not None else 0))
        conn.commit()
        conn.close()

    def load_predictions(self) -> pd.DataFrame:
        """加载预测记录"""
        conn = self._get_conn()
        df = pd.read_sql("SELECT * FROM ssq_predictions ORDER BY issue DESC", conn)
        conn.close()
        return df
