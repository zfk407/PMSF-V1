"""
大乐透历史数据获取器
支持：1) 从公开网站抓取  2) 从CSV加载  3) 生成模拟数据（兜底）
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


class DltDataFetcher:
    """大乐透数据获取器"""

    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = raw_dir
        os.makedirs(raw_dir, exist_ok=True)

    # ---------- 公开数据源抓取 ----------
    def fetch_from_500(self, max_pages: int = 50) -> pd.DataFrame:
        """
        从500彩票网抓取大乐透历史数据
        表格结构：tr > td(t_tr1期号) + td(cfont2前区x5) + td(后区x2) + ... + td(t_tr1日期)
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
            url = f"https://datachart.500.com/dlt/history/newinc/history.php?start=07001&end=99999&page={page}"
            try:
                resp = requests.get(url, headers=headers, timeout=20)
                resp.encoding = "gbk"
                soup = BeautifulSoup(resp.text, "lxml")

                # 找所有包含期号的行：期号td的class是t_tr1，且内容为5位数字
                page_count = 0
                for tr in soup.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) < 9:
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
                        if not (7 <= year_prefix <= 26 and 1 <= seq <= 150):
                            continue
                    except ValueError:
                        continue

                    # 前区5个号码：class为cfont2的td
                    front_tds = tr.find_all("td", class_="cfont2")
                    if len(front_tds) < 5:
                        continue
                    front = []
                    for ft in front_tds[:5]:
                        val = ft.get_text(strip=True)
                        if re.match(r'^\d{2}$', val):
                            front.append(int(val))
                    if len(front) != 5 or not all(1 <= n <= 35 for n in front):
                        continue

                    # 后区2个号码：在前区之后的td中找2位数字
                    back = []
                    # 后区通常在前区td之后，找非cfont2的2位数字td
                    all_td_texts = [td.get_text(strip=True) for td in tds]
                    # 从第6个td开始找后区（前5个cfont2之后）
                    found_front = False
                    front_count = 0
                    for td_text in all_td_texts:
                        if re.match(r'^\d{2}$', td_text):
                            num = int(td_text)
                            if not found_front:
                                front_count += 1
                                if front_count >= 5:
                                    found_front = True
                            else:
                                if 1 <= num <= 12 and len(back) < 2:
                                    back.append(num)
                                if len(back) >= 2:
                                    break
                    if len(back) != 2:
                        continue

                    # 日期：最后一个t_tr1且匹配日期格式
                    date_str = ""
                    for td in reversed(tds):
                        txt = td.get_text(strip=True)
                        if re.match(r'^\d{4}-\d{2}-\d{2}$', txt):
                            date_str = txt
                            break

                    all_records.append({
                        "issue": issue_text, "date": date_str,
                        "front01": front[0], "front02": front[1],
                        "front03": front[2], "front04": front[3],
                        "front05": front[4],
                        "back01": back[0], "back02": back[1]
                    })
                    page_count += 1

                print(f"[抓取] 第{page}页: {page_count}期")
                if page_count == 0:
                    break
                time.sleep(0.3)
            except Exception as e:
                print(f"[警告] 第{page}页抓取失败: {e}")
                import traceback
                traceback.print_exc()
                break

        if all_records:
            df = pd.DataFrame(all_records).drop_duplicates(subset=["issue"])
            df = df.sort_values("issue").reset_index(drop=True)
            self._save_csv(df, "dlt_500.csv")
            print(f"[成功] 从500彩票网获取 {len(df)} 期真实数据")
            print(f"  数据范围: {df['issue'].iloc[0]} ~ {df['issue'].iloc[-1]}")
            return df
        return pd.DataFrame()

    # ---------- CSV 加载 ----------
    def load_from_csv(self, filename: str = "dlt_history.csv") -> pd.DataFrame:
        """从CSV文件加载"""
        path = os.path.join(self.raw_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path, dtype={"issue": str})
            print(f"[成功] 从CSV加载 {len(df)} 期数据")
            return df
        return pd.DataFrame()

    # ---------- 模拟数据生成（兜底） ----------
    def generate_mock_data(self, n_periods: int = 2000, start_issue: str = "07001") -> pd.DataFrame:
        """
        生成模拟大乐透数据用于系统联调
        注意：这是随机模拟数据，仅用于测试系统流程，不代表真实开奖
        """
        print(f"[信息] 生成 {n_periods} 期模拟数据（用于系统联调）")
        np.random.seed(42)
        records = []
        # 解析起始期号
        year = int(start_issue[:2])
        seq = int(start_issue[2:])
        base_date = datetime(2007, 1, 1)

        for i in range(n_periods):
            # 期号递增
            cur_year = year + (seq + i - 1) // 150
            cur_seq = (seq + i - 1) % 150 + 1
            issue = f"{cur_year:02d}{cur_seq:03d}"
            # 开奖日期（约每周3期）
            draw_date = base_date + timedelta(days=i * 2.3)
            # 前区 5 个不重复 1-35
            front = sorted(np.random.choice(range(1, 36), 5, replace=False).tolist())
            # 后区 2 个不重复 1-12
            back = sorted(np.random.choice(range(1, 13), 2, replace=False).tolist())
            records.append({
                "issue": issue,
                "date": draw_date.strftime("%Y-%m-%d"),
                "front01": front[0], "front02": front[1],
                "front03": front[2], "front04": front[3],
                "front05": front[4],
                "back01": back[0], "back02": back[1]
            })

        df = pd.DataFrame(records)
        self._save_csv(df, "dlt_mock.csv")
        return df

    # ---------- 衍生特征计算 ----------
    @staticmethod
    def enrich_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        """计算和值、跨度、奇偶、大小、四区、尾数等衍生特征"""
        df = df.copy()
        front_cols = ["front01", "front02", "front03", "front04", "front05"]

        # 和值
        df["sum_front"] = df[front_cols].sum(axis=1)
        # 跨度
        df["span_front"] = df[front_cols].max(axis=1) - df[front_cols].min(axis=1)

        # 奇偶结构 (奇,偶)
        def odd_even(row):
            odds = sum(1 for c in front_cols if row[c] % 2 == 1)
            return f"{odds}奇{5 - odds}偶"
        df["odd_even"] = df.apply(odd_even, axis=1)

        # 大小结构 (大>=18, 小<18)
        def big_small(row):
            bigs = sum(1 for c in front_cols if row[c] >= 18)
            return f"{bigs}大{5 - bigs}小"
        df["big_small"] = df.apply(big_small, axis=1)

        # 四区结构
        def zone_dist(row):
            nums = [row[c] for c in front_cols]
            z1 = sum(1 for n in nums if 1 <= n <= 9)
            z2 = sum(1 for n in nums if 10 <= n <= 18)
            z3 = sum(1 for n in nums if 19 <= n <= 27)
            z4 = sum(1 for n in nums if 28 <= n <= 35)
            return f"{z1}-{z2}-{z3}-{z4}"
        df["zone"] = df.apply(zone_dist, axis=1)

        # 尾数结构
        def tail_dist(row):
            nums = [row[c] for c in front_cols]
            tails = [n % 10 for n in nums]
            return "-".join(map(str, sorted(tails)))
        df["tail"] = df.apply(tail_dist, axis=1)

        return df

    def _save_csv(self, df: pd.DataFrame, filename: str):
        path = os.path.join(self.raw_dir, filename)
        df.to_csv(path, index=False, encoding="utf-8-sig")

    def get_data(self, use_web: bool = True, use_mock_fallback: bool = True,
                 n_mock: int = 2000) -> pd.DataFrame:
        """
        统一数据获取入口：优先网络 -> CSV -> 模拟
        返回已 enriched 的 DataFrame
        """
        df = pd.DataFrame()
        # 1. 尝试网络
        if use_web and HAS_WEB:
            try:
                df = self.fetch_from_500(max_pages=30)
            except Exception as e:
                print(f"[警告] 网络抓取异常: {e}")
        # 2. 尝试CSV
        if df.empty:
            df = self.load_from_csv("dlt_history.csv")
        if df.empty:
            df = self.load_from_csv("dlt_500.csv")
        # 3. 模拟兜底
        if df.empty and use_mock_fallback:
            df = self.generate_mock_data(n_periods=n_mock)

        if not df.empty:
            df = self.enrich_derived_features(df)
        return df
