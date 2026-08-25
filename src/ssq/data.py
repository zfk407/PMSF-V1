"""
双色球历史数据获取器
支持：1) 从500彩票网抓取  2) 从CSV加载  3) 生成模拟数据（兜底）
"""
import os
import re
import time
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_WEB = True
except ImportError:
    HAS_WEB = False


class SsqDataFetcher:
    """双色球数据获取器"""

    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = raw_dir
        os.makedirs(raw_dir, exist_ok=True)

    # ---------- 公开数据源抓取 ----------
    def fetch_from_500(self, max_pages: int = 50) -> pd.DataFrame:
        """
        从500彩票网抓取双色球历史数据
        表格结构：tr > td(t_tr1期号) + td(cfont2红球x6) + td(cfont2蓝球x1) + ... + td(t_tr1日期)
        红球前6个cfont2，第7个cfont2为蓝球
        """
        if not HAS_WEB:
            print("[警告] 未安装 requests/bs4，无法网络抓取")
            return pd.DataFrame()

        all_records = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }

        for page in range(1, max_pages + 1):
            url = f"https://datachart.500.com/ssq/history/newinc/history.php?start=03001&end=99999&page={page}"
            try:
                resp = requests.get(url, headers=headers, timeout=20)
                resp.encoding = "gbk"
                soup = BeautifulSoup(resp.text, "lxml")

                page_count = 0
                for tr in soup.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) < 8:
                        continue
                    # 第一个td是期号
                    issue_td = tds[0]
                    issue_text = issue_td.get_text(strip=True)
                    if not re.match(r'^\d{5}$', issue_text):
                        continue
                    # 验证期号合理性
                    try:
                        year_prefix = int(issue_text[:2])
                        seq = int(issue_text[2:])
                        if not (3 <= year_prefix <= 26 and 1 <= seq <= 154):
                            continue
                    except ValueError:
                        continue

                    # 所有cfont2号码（红6 + 蓝1）
                    num_tds = tr.find_all("td", class_="cfont2")
                    if len(num_tds) < 7:
                        continue
                    nums = []
                    for nt in num_tds[:7]:
                        val = nt.get_text(strip=True)
                        if re.match(r'^\d{2}$', val):
                            nums.append(int(val))
                    if len(nums) < 7:
                        continue
                    reds = nums[:6]
                    blue = nums[6]
                    if not all(1 <= n <= 33 for n in reds) or not (1 <= blue <= 16):
                        continue

                    # 日期
                    date_str = ""
                    for td in reversed(tds):
                        txt = td.get_text(strip=True)
                        if re.match(r'^\d{4}-\d{2}-\d{2}$', txt):
                            date_str = txt
                            break

                    all_records.append({
                        "issue": issue_text, "date": date_str,
                        "red01": reds[0], "red02": reds[1], "red03": reds[2],
                        "red04": reds[3], "red05": reds[4], "red06": reds[5],
                        "blue": blue
                    })
                    page_count += 1

                print(f"[抓取] 第{page}页: {page_count}期")
                if page_count == 0:
                    break
                time.sleep(0.3)
            except Exception as e:
                print(f"[警告] 第{page}页抓取失败: {e}")
                break

        if all_records:
            df = pd.DataFrame(all_records).drop_duplicates(subset=["issue"])
            df = df.sort_values("issue").reset_index(drop=True)
            self._save_csv(df, "ssq_500.csv")
            print(f"[成功] 从500彩票网获取 {len(df)} 期双色球真实数据")
            print(f"  数据范围: {df['issue'].iloc[0]} ~ {df['issue'].iloc[-1]}")
            return df
        return pd.DataFrame()

    # ---------- CSV 加载 ----------
    def load_from_csv(self, filename: str = "ssq_history.csv") -> pd.DataFrame:
        """从CSV文件加载"""
        path = os.path.join(self.raw_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path, dtype={"issue": str})
            print(f"[成功] 从CSV加载 {len(df)} 期双色球数据")
            return df
        return pd.DataFrame()

    # ---------- 模拟数据生成（兜底） ----------
    def generate_mock_data(self, n_periods: int = 3000, start_issue: str = "03001") -> pd.DataFrame:
        """
        生成模拟双色球数据用于系统联调
        注意：这是随机模拟数据，仅用于测试系统流程，不代表真实开奖
        """
        print(f"[信息] 生成 {n_periods} 期模拟双色球数据（用于系统联调）")
        np.random.seed(2023)
        records = []
        year = int(start_issue[:2])
        seq = int(start_issue[2:])
        base_date = datetime(2003, 2, 1)

        for i in range(n_periods):
            cur_year = year + (seq + i - 1) // 154
            cur_seq = (seq + i - 1) % 154 + 1
            issue = f"{cur_year:02d}{cur_seq:03d}"
            draw_date = base_date + timedelta(days=i * 2.3)
            reds = sorted(np.random.choice(range(1, 34), 6, replace=False).tolist())
            blue = int(np.random.choice(range(1, 17), 1)[0])
            records.append({
                "issue": issue,
                "date": draw_date.strftime("%Y-%m-%d"),
                "red01": reds[0], "red02": reds[1], "red03": reds[2],
                "red04": reds[3], "red05": reds[4], "red06": reds[5],
                "blue": blue
            })

        df = pd.DataFrame(records)
        self._save_csv(df, "ssq_mock.csv")
        return df

    # ---------- 衍生特征计算 ----------
    @staticmethod
    def enrich_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """计算和值、跨度、奇偶、大小、三区、尾数等衍生特征"""
        df = df.copy()
        red_cols = ["red01", "red02", "red03", "red04", "red05", "red06"]

        # 和值
        df["sum_red"] = df[red_cols].sum(axis=1)
        # 跨度
        df["span_red"] = df[red_cols].max(axis=1) - df[red_cols].min(axis=1)

        # 奇偶结构 (奇,偶)
        def odd_even(row):
            odds = sum(1 for c in red_cols if row[c] % 2 == 1)
            return f"{odds}奇{6 - odds}偶"
        df["odd_even"] = df.apply(odd_even, axis=1)

        # 大小结构 (大>=17, 小<17)
        def big_small(row):
            bigs = sum(1 for c in red_cols if row[c] >= 17)
            return f"{bigs}大{6 - bigs}小"
        df["big_small"] = df.apply(big_small, axis=1)

        # 三区结构（1-11, 12-22, 23-33）
        def zone_dist(row):
            nums = [row[c] for c in red_cols]
            z1 = sum(1 for n in nums if 1 <= n <= 11)
            z2 = sum(1 for n in nums if 12 <= n <= 22)
            z3 = sum(1 for n in nums if 23 <= n <= 33)
            return f"{z1}-{z2}-{z3}"
        df["zone"] = df.apply(zone_dist, axis=1)

        # 尾数结构
        def tail_dist(row):
            nums = [row[c] for c in red_cols]
            tails = sorted(n % 10 for n in nums)
            return "-".join(map(str, tails))
        df["tail"] = df.apply(tail_dist, axis=1)

        return df

    def _save_csv(self, df: pd.DataFrame, filename: str):
        path = os.path.join(self.raw_dir, filename)
        df.to_csv(path, index=False, encoding="utf-8-sig")

    def get_data(self, use_web: bool = True, use_mock_fallback: bool = True,
                 n_mock: int = 3000) -> pd.DataFrame:
        """
        统一数据获取入口：优先网络 -> CSV -> 模拟
        返回已 enriched 的 DataFrame
        """
        df = pd.DataFrame()
        if use_web and HAS_WEB:
            try:
                df = self.fetch_from_500(max_pages=30)
            except Exception as e:
                print(f"[警告] 网络抓取异常: {e}")
        if df.empty:
            df = self.load_from_csv("ssq_history.csv")
        if df.empty:
            df = self.load_from_csv("ssq_500.csv")
        if df.empty and use_mock_fallback:
            df = self.generate_mock_data(n_periods=n_mock)

        if not df.empty:
            df = self.enrich_derived_features(df)
        return df
