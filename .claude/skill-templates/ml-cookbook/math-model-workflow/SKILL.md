---
name: "math-model-workflow"
description: "数学建模全流程工程化辅助系统—7阶段工作流(DISCOVERY→FORMULATION→COMPUTATION→EVIDENCE→SCHEMATICS→MANUSCRIPT→ASSURANCE)+冠军审稿+图表排版硬约束。适用于CUMCM/51MCM/MCM-ICM。触发词：开始数模研究、处理赛题、建模工作流、冠军审稿、论文排版检查"
---

# Meta-model-agent

Meta-model-agent 是面向数学建模研究与竞赛论文生产的工程化辅助系统。它将题意研判、数学建模、程序求解、结果验证、图形表达、论文组织和提交检查连接为一条可执行、可恢复、可审计的研究链。

## 项目说明

本项目的核心目标是辅助完成从研究问题到最终论文的完整质量闭环：先建立可信的问题与模型契约，再通过真实计算获得结果，以图形、表格和结构图组织证据，最后形成符合竞赛规范的论文并接受多轮质量审稿。

## 使用方式

### 初始化工作区

```bash
python scripts/workspace_init.py --workspace . --competition cumcm --output-format pdf
```

竞赛类型可选 `cumcm`、`51mcm`、`mcm-icm`。输出格式可选 `pdf` 或 `docx`。

### 工作流程（7 阶段）

| 阶段 | 目标 | 主要产物 |
| --- | --- | --- |
| `DISCOVERY` | 读取题面、附件与数据，拆解问题 | `问题分析.md` |
| `FORMULATION` | 建立数学机制、假设、公式与验证方案 | `建模报告.md` |
| `COMPUTATION` | 编写程序并开展真实计算 | `程序/主程序.py`、`计算结果.md` |
| `EVIDENCE` | 将结果转化为论文图表与数据证据 | 图形、表格、`图表/figure_manifest.json` |
| `SCHEMATICS` | 绘制技术路线和系统逻辑图 | DrawIO/TikZ 源文件及论文引用 |
| `MANUSCRIPT` | 集成模型、实验、图表与引用 | LaTeX 或 Markdown 论文源稿 |
| `ASSURANCE` | 编译并检查最终材料 | `论文/数模论文.pdf` 或 `.docx` |

### 运行阶段命令

```bash
# 查看当前阶段
python scripts/stage_executor.py current --workspace .

# 开始当前阶段
python scripts/stage_executor.py begin DISCOVERY --workspace .

# 验证 → 门禁 → 完成
python scripts/stage_executor.py validate DISCOVERY --workspace .
python scripts/stage_executor.py gate_check DISCOVERY --workspace .
python scripts/stage_executor.py complete DISCOVERY --workspace . --artifacts "问题分析.md"
```

### 质量模式

- `baseline`（默认）：保持阶段、交付物和门禁稳定
- `enhancement`：针对薄弱项实施受控返工
- `championship`：论文完成后加入多轮独立模拟审稿和全文修订

切换命令：
```bash
python scripts/pipeline_manager.py set-mode championship --workspace .
```

### 论文排版硬约束

- LaTeX 图表使用 `[htbp]` 并在章节边界用 `\FloatBarrier` 收束，禁止全篇强制 `[H]`
- 图中文字在最终 PDF 100% 显示比例下必须可读
- CUMCM 摘要不超 1 页，正文不超 30 页，附录单独计数
- 代码附录必须从当前源码自动生成

### 图表与流程图硬约束

- 所有 DrawIO/TikZ 流程图中的普通矩形、容器和表头必须使用直角矩形
- 每张图必须登记明确论点、数据来源和读者任务
- AI Image 默认关闭，禁止用于技术路线图、流程图和模型架构图

### 冠军审稿

冠军模式下执行不少于 3 轮独立审稿，终版需满足：P0 为 0、P1 不超过 2、综合分不低于 85。

## 文档索引

- 工作流总图：`references/workflow-map.md`
- 阶段门禁矩阵：`references/gate-matrix.md`
- 阶段控制规则：`references/phase-control.md`
- 增强操作指南：`references/enhancement-operations.md`
- 冠军审稿方法：`references/championship-review-method.md`
- 竞赛机器配置：`assets/competition_profiles.json`
- 各阶段实施协议：`references/stage_protocols/*/SKILL.md`
- 环境安装说明：`ENVIRONMENT.md`

## 注意事项

- 仅提供研究辅助，题意、数据、模型、程序、引用、结果和最终材料必须由使用者核验
- 内部评分不代表竞赛结果承诺
- 自动门禁只能检查已编码的合同，不能证明模型或结论一定正确

