# PMSF-V1 可视化Web界面

彭湃大乐透多尺度状态融合系统的可视化前端，支持历史数据查询、走势图、预测中心、模型介绍、详细报告、系统运行监控。

## 功能特性

- 📊 **仪表盘** - 关键指标、号码热度、和值走势、状态分布、最新4组推荐
- 📋 **历史数据** - 2872+期完整开奖记录、号码频率/和值/跨度/奇偶/大小/四区分布图
- 🔮 **预测中心** - 最新4组推荐、中奖台账（自动标注奖项等级）、命中率统计
- 🤖 **模型介绍** - 五层架构可视化、融合权重、号码关系网络、8个模型详解
- 📑 **详细报告** - 完整技术报告展示
- ⚙️ **系统运行** - 运行流程可视化、最新复盘分析、运行日志、数据更新说明

## 技术栈

- 前端：原生 HTML + CSS + JavaScript（无构建工具，直接部署）
- 图表：ECharts 5.5（CDN引入）
- 数据：静态 JSON 文件
- 自动化：GitHub Actions + Python

## 部署方式

### 方式一：GitHub Pages（推荐，免费）

1. 将 `web/` 目录推送到 GitHub 仓库
2. 仓库 Settings → Pages → Source 选择 `main` 分支，目录选择 `/web`（或根目录）
3. 等待部署完成，访问 `https://<用户名>.github.io/<仓库名>/web/`

### 方式二：Vercel / Netlify / Cloudflare Pages（免费）

1. 导入 GitHub 仓库
2. 构建命令：无（静态站点）
3. 输出目录：`web`
4. 部署完成后获得免费域名

### 方式三：本地运行

```bash
cd web
python -m http.server 8080
# 访问 http://localhost:8080
```

## 数据更新

### 自动更新（GitHub Actions）

已配置 `.github/workflows/update.yml`，每周一/三/六 21:30（北京时间）自动执行：
1. 从500彩票网抓取最新开奖数据
2. 对比上期推荐进行复盘分析
3. 重新运行PMSF系统预测下一期
4. 更新所有JSON数据文件
5. 自动提交并部署

也可在 Actions 页面手动触发。

### 手动更新

```bash
cd web
python scripts/update_data.py
```

## 目录结构

```
web/
├── index.html              # 主页面
├── css/
│   └── style.css           # 样式（深色科技风）
├── js/
│   ├── data.js             # 数据加载与工具函数
│   ├── charts.js           # ECharts图表配置
│   └── app.js              # 主应用逻辑
├── data/                   # JSON数据文件（自动生成）
│   ├── history.json        # 历史开奖数据
│   ├── stats.json          # 统计分析
│   ├── latest_prediction.json  # 最新预测
│   ├── predictions.json    # 历史预测记录
│   ├── report.json         # 详细报告
│   └── runtime.json        # 运行时信息
├── scripts/
│   └── update_data.py      # 数据更新脚本
└── .github/workflows/
    └── update.yml          # GitHub Actions自动化
```

## 数据来源

历史开奖数据来自 500彩票网（datachart.500.com），仅供学习研究使用。

## 免责声明

本系统基于历史数据统计分析，彩票开奖为独立随机事件，结果仅供参考，不构成投注建议。彩票有风险，投注需理性。
