<div align="center">

# 🧠 Information Diet Manager

**信息摄取质量评估与“信息茧房”分析工具**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-4285F4?style=flat-square&logo=googlechrome&logoColor=white)](https://developer.chrome.com/docs/extensions/)
[![Vite](https://img.shields.io/badge/Vite-Frontend-646CFF?style=flat-square&logo=vite&logoColor=white)](https://vitejs.dev/)

</div>

---

## 📖 项目简介

Information Diet Manager 用于评估用户的信息摄取结构与“信息茧房”程度，支持：

- Chrome 插件自动采集浏览信息（URL、标题、正文摘要等）
- FastAPI 后端接收、清洗、存储数据
- 分析引擎执行分类、情感、相似度与综合评估
- 可视化摘要与导出（JSON / JSONL / CSV）

---

## ✨ 核心功能

- **数据采集**：浏览器插件自动抓取网页信息并上报
- **数据接入**：`/collect`、`/import` 支持实时与批量导入
- **数据存储**：SQLite（默认）+ 标准化哈希去重
- **分析评估**：
  - 内容分类（category）
  - 情感分析（sentiment / polarity）
  - 相似度计算（similarity）
  - 信息摄取质量评估（summary/report）
- **结果输出**：
  - 仪表盘摘要 `/dashboard/summary`
  - 可视化数据 `/dashboard/visualization`
  - 导出给分析脚本 `/export/lsj`、`/export/lsj/training`

---

## 🏗️ 当前项目结构（重构后）

```text
INFORMATION-DIET-MANAGER/
├── chrome-extension/              # Chrome 插件（采集端）
├── docs/                          # 文档
├── frontend/                      # 前端（Vite）
├── scripts/
│   ├── run_backend.py             # 后端启动脚本（推荐）
│   └── run_analysis.py            # 分析入口脚本（推荐）
├── src/
│   ├── backend_api/               # 语义化后端入口（推荐）
│   ├── analysis_engine/           # 语义化分析入口（推荐）
│   ├── hyh/                       # legacy 后端实现（兼容保留）
│   └── lsj/                       # legacy 分析实现（兼容保留）
├── requirements.txt
└── README.md
```

> 说明：
> `backend_api` / `analysis_engine` 是新的语义化入口层；
> `hyh` / `lsj` 暂保留用于兼容旧代码与历史脚本。

---

## 🚀 快速开始

## 1) 安装依赖

建议在项目根目录操作：

```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## 2) 启动后端（推荐方式）

```bash
python scripts/run_backend.py
```

启动后访问：

- Swagger 文档: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 3) 启动前端（可选）

```bash
cd frontend
npm install
npm run dev
```

---

## 4) 安装 Chrome 插件

1. 打开 `chrome://extensions/`
2. 开启“开发者模式”
3. 点击“加载已解压的扩展程序”
4. 选择项目下 `chrome-extension/`
5. 浏览网页后，插件将采集数据并上报后端

---

## 5) 运行分析（命令行）

```bash
python scripts/run_analysis.py --help
```

示例（analyze 模式）：

```bash
python scripts/run_analysis.py \
  --mode analyze \
  --input_file data.json \
  --output_file result.json
```

---

## 🔌 主要 API

- `POST /collect`：单条采集入库
- `POST /import`：文件导入（csv/json/jsonl）
- `GET /items`：分页查询原始数据
- `POST /analyze/run`：轻量分析
- `POST /analyze/run_full`：全量分析任务
- `GET /analyze/jobs/{job_id}`：任务状态
- `GET /analyze/result/{job_id}`：任务结果
- `GET /dashboard/summary`：摘要指标
- `GET /dashboard/visualization`：可视化数据
- `GET /export/lsj`：导出分析数据
- `GET /export/lsj/training`：导出训练数据

---

## 🔄 数据流说明

1. Chrome 插件采集网页内容
2. POST 到后端 `/collect`
3. 后端规范化、去重、入库
4. 调用 `/analyze/run_full` 触发分析管道
5. 前端或脚本读取 `/dashboard/*` 与 `/analyze/result/*` 结果

---

## 🧩 常见问题

### 1) `ModuleNotFoundError: No module named 'src'`

请确保在**项目根目录**运行命令，优先使用：

```bash
python scripts/run_backend.py
```

### 2) 插件没有数据上报

- 确认后端已启动（`127.0.0.1:8000`）
- 确认插件配置的 API 地址正确
- 在扩展管理页查看 service worker 日志排错

---

## 📄 License

仅供课程/研究与学习用途。
如需对外发布，请补充明确的 LICENSE 文件与隐私说明。
