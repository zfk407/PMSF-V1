# PMSF-V1 彭湃大乐透多尺度状态融合智能筛选系统

> Version: 1.0 | Architecture: Rule + State + Graph + Ensemble + Optimization

## 系统定位

PMSF-V1 不是传统"彩票预测模型"，而是一个基于**历史时序、状态转换、号码关系网络、多模型融合与组合优化**的概率筛选系统。

核心目标：在历史数据空间、彭湃经验规则空间、统计概率空间、机器学习空间中，寻找综合评分最高的有限组合集合。

## 五层架构

```
                 最终4组5+2
                      ↑
             第五层：组合优化层
        Monte Carlo / Genetic Algorithm / Risk Control
                      ↑
             第四层：概率融合层
 XGBoost / CatBoost / TFT / GNN / Copula / Bayesian Fusion
                      ↑
             第三层：状态识别层
          HSMM / HMM / Markov Switching
                      ↑
             第二层：关系网络层
                GNN / Copula / 彭湃规则
                      ↑
             第一层：数据基础层
          数据库 / 数据获取 / 特征工程
                      ↑
                原始开奖数据
```

## 核心创新

| 传统彩票分析 | PMSF-V1 |
|------------|---------|
| 看号码频率 | 看状态 |
| 号码独立 | 号码关系网络 |
| 单模型 | 多模型融合 |
| 固定规则 | 规则作为先验 |
| 直接选号 | 概率空间优化 |
| 凭经验 | 滚动回测验证 |

## 三态系统

- **STATE-A 纠缠热态**：热号延续、配对活跃、结构稳定
- **STATE-B 终止冷态**：热号退出、关系断裂、遗漏增加
- **STATE-C 拓展回补态**：冷号进入、区域扩散、新关系形成

## 4组输出

- **A组 模型共识组**：最高概率，稳定性最强
- **B组 彭湃强化组**：规则匹配最高，双线/纠缠/配对强化
- **C组 冷态拓展组**：捕捉状态转换，冷号回补导向
- **D组 探索组**：防止模型过拟合，覆盖低概率区域

## 快速开始

### 1. 安装依赖

```bash
cd E:\PMSF-V1
pip install -r requirements.txt
```

### 2. 运行预测

```bash
# 预测下一期（自动尝试网络获取数据）
python main.py --mode predict

# 指定目标期号
python main.py --mode predict --issue 25101

# 不使用网络（用本地数据或模拟数据）
python main.py --mode predict --no-web
```

### 3. 运行回测

```bash
# 滚动回测30期
python main.py --mode backtest --n-test 30
```

## 项目结构

```
E:\PMSF-V1\
├── main.py                    # 主入口
├── config.yaml                # 全局配置
├── requirements.txt           # 依赖
├── README.md                  # 说明文档
├── data/
│   ├── raw/                   # 原始数据（CSV）
│   └── processed/             # 处理后数据（SQLite）
├── src/
│   ├── layer1_data/           # 第一层：数据基础层
│   ├── layer2_rules/          # 第二层：彭湃规则层
│   ├── layer3_state/          # 第三层：状态识别层
│   ├── layer4_probability/    # 第四层：概率模型层
│   ├── layer5_optimization/   # 第五层：组合优化层
│   ├── backtest/              # 滚动回测系统
│   └── output/                # 输出生成器
├── results/                   # 输出结果
└── logs/                      # 日志
```

## 评价指标

不使用单纯"中了几次"，采用：

1. **覆盖率**：预测号码集合覆盖实际号码的比例
2. **Top-K命中率**：模型前K号码中实际出现几个
3. **结构命中**：预测的奇偶/大小/四区结构是否成立
4. **状态命中**：预测的状态是否与实际表现一致

## 融合权重

```
最终评分 = 0.25*XGBoost + 0.20*CatBoost + 0.15*GNN
         + 0.15*TFT + 0.10*HSMM + 0.10*Bayesian
         + 0.05*彭湃规则修正
```

权重通过滚动回测动态调整。

## 免责声明

本系统基于历史数据统计分析，彩票开奖为独立随机事件，结果仅供参考，不构成投注建议。请理性购彩，量力而行。

---

**PMSF-V1.0** | Technical White Paper V1
