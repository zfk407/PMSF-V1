"""
2组输出生成器
A组 - 模型共识组：最高概率
B组 - 彭湃强化组：规则匹配最高
"""
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json


class OutputGenerator:
    """2组输出生成器"""

    GROUP_NAMES = {
        "A": "模型共识组",
        "B": "彭湃强化组"
    }

    GROUP_DESCRIPTIONS = {
        "A": "多模型融合概率最高，稳定性最强",
        "B": "彭湃规则匹配度最高，双线/纠缠/配对强化"
    }

    def __init__(self, config: dict):
        self.cfg = config
        self.result_dir = config["output"]["result_dir"]
        self.n_groups = config["output"].get("n_groups", 2)
        os.makedirs(self.result_dir, exist_ok=True)

    def generate(self, optimized_groups: list, fused_probs: dict,
                 state: str, state_probs: dict,
                 structure_filter, risk_report: dict = None,
                 target_issue: str = "NEXT") -> dict:
        """
        生成最终4组输出
        optimized_groups: 遗传算法优化后的 [(front, back, fitness), ...]
        fused_probs: 融合概率 {number: prob}
        state: 当前状态 A/B/C
        state_probs: 状态概率
        structure_filter: 结构过滤器实例
        risk_report: 风控报告
        target_issue: 目标期号
        返回: 完整输出报告
        """
        # 确保有n_groups组
        groups = []
        for i in range(self.n_groups):
            if i < len(optimized_groups):
                front, back, fitness = optimized_groups[i]
            else:
                # 兜底：从概率最高的号码中生成
                top_nums = sorted(fused_probs.items(), key=lambda x: x[1], reverse=True)
                front = tuple(sorted([n for n, _ in top_nums[i*5:(i+1)*5]]))
                back = tuple(sorted(np.random.choice(range(1, 13), 2, replace=False).tolist()))
                fitness = 0.0
            groups.append({"front": front, "back": back, "fitness": fitness})

        # 为每组分配标签并生成详情
        output = {
            "system": "ZHIHUI-DLT",
            "version": "1.0",
            "target_issue": target_issue,
            "generate_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_state": {
                "state": state,
                "state_name": self._state_name(state),
                "probabilities": state_probs
            },
            "groups": [],
            "top10_numbers": self._get_top10(fused_probs),
            "risk_report": risk_report,
            "disclaimer": "本系统基于历史数据统计分析，彩票开奖为独立随机事件，结果仅供参考，不构成投注建议。"
        }

        labels = ["A", "B"]
        for i, (label, group) in enumerate(zip(labels, groups)):
            front = group["front"]
            back = group["back"]
            struct_info = structure_filter.get_structure_info(front)

            group_detail = {
                "label": label,
                "name": self.GROUP_NAMES[label],
                "description": self.GROUP_DESCRIPTIONS[label],
                "front": list(front),
                "front_str": " ".join(f"{n:02d}" for n in front),
                "back": list(back),
                "back_str": " ".join(f"{n:02d}" for n in back),
                "fitness": float(group["fitness"]),
                "probability_score": float(sum(np.log(fused_probs.get(n, 1/35) + 1e-10) for n in front)),
                "structure": struct_info,
                "number_probs": {str(n): float(fused_probs.get(n, 1/35)) for n in front}
            }
            output["groups"].append(group_detail)

        return output

    def _state_name(self, state: str) -> str:
        names = {"A": "纠缠热态", "B": "终止冷态", "C": "拓展回补态"}
        return names.get(state, "未知")

    def _get_top10(self, probs: dict) -> list:
        """获取概率Top10号码"""
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return [{"number": num, "probability": float(p)} for num, p in sorted_probs[:10]]

    def format_text(self, output: dict) -> str:
        """格式化为文本输出"""
        lines = []
        lines.append("=" * 60)
        lines.append("  PMSF-V1 彭湃大乐透多尺度状态融合系统")
        lines.append("=" * 60)
        lines.append(f"  目标期号: {output['target_issue']}")
        lines.append(f"  生成时间: {output['generate_time']}")
        lines.append("")

        # 状态
        state_info = output["current_state"]
        lines.append(f"  当前状态: {state_info['state_name']} ({state_info['state']})")
        sp = state_info["probabilities"]
        lines.append(f"    纠缠热态(A): {sp.get('A', 0):.2%}")
        lines.append(f"    终止冷态(B): {sp.get('B', 0):.2%}")
        lines.append(f"    拓展回补态(C): {sp.get('C', 0):.2%}")
        lines.append("")

        # Top10
        lines.append("  概率Top10号码:")
        for i, item in enumerate(output["top10_numbers"], 1):
            lines.append(f"    {i:2d}. {item['number']:02d}  概率: {item['probability']:.4f}")
        lines.append("")

        # 2组
        lines.append("-" * 60)
        lines.append("  推荐组合 (2组5+2):")
        lines.append("-" * 60)
        for group in output["groups"]:
            lines.append("")
            lines.append(f"  【{group['label']}组】{group['name']}")
            lines.append(f"    {group['description']}")
            lines.append(f"    前区: {group['front_str']}")
            lines.append(f"    后区: {group['back_str']}")
            struct = group["structure"]
            lines.append(f"    结构: {struct['odd_even']} | {struct['big_small']} | "
                         f"四区:{struct['zone']} | 和值:{struct['sum']} | 跨度:{struct['span']}")

        lines.append("")
        lines.append("-" * 60)
        lines.append(f"  免责声明: {output['disclaimer']}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save(self, output: dict, filename: str = None) -> str:
        """保存输出结果到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pmsf_output_{output['target_issue']}_{timestamp}"

        # 保存JSON
        json_path = os.path.join(self.result_dir, f"{filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        # 保存文本
        txt_path = os.path.join(self.result_dir, f"{filename}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(self.format_text(output))

        return json_path, txt_path
