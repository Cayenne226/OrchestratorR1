# Week 1 Summary: 数据准备 + 基础设施

> **日期**: 3/24 - 3/30
> **状态**: [~] 进行中

## 任务完成情况

| ID | 任务 | 状态 | 备注 |
|----|------|------|------|
| T1.1 | QA 数据集准备 | [x] | 6 datasets × 1000/500, 全部加载成功 |
| T1.2 | 代码数据集准备 | [x] | HumanEval 164 + MBPP full 500/374 |
| T1.3 | GPQA + LiveCodeBench | [x] | GPQA 198条 + LCB 202条 |
| T1.4 | SFT 热身数据生成 | [x] | 200条, 8种路径模式, 6种agent全覆盖, 0失败 |
| T1.5 | A100 环境配置 | [ ] | |
| T1.6 | Base model baseline | [ ] | |
| T1.7 | Direct-GPT-4o baseline | [ ] | |

## 关键数字

| 指标 | 值 |
|------|-----|
| 训练数据总量 | 6,374 条 (QA 6000 + Code 374) |
| 测试数据总量 | 4,064 条 (QA 3000 + Code 664 + GPQA 198 + LCB 202) |
| SFT 热身数据 | 200 条 (8种路径模式, 6种agent全覆盖) |
| Base model F1 (QA) | 待测 |
| Direct-GPT-4o F1 (QA) | 待测 |
| API 花费 | $0 (数据准备无需 API) |

## 数据集详情

### 训练集分布 (train_mixed.jsonl = 6,374)
| Source | Count | Difficulty |
|--------|-------|-----------|
| NQ | 1,000 | simple |
| TriviaQA | 1,000 | simple |
| PopQA | 1,000 | simple |
| HotpotQA | 1,000 | multi_hop |
| 2WikiMultihopQA | 1,000 | multi_hop |
| MuSiQue | 1,000 | multi_hop |
| MBPP | 374 | code |

### 测试集分布 (test_qa=3,000 + test_code=664)
| Source | Count | Difficulty |
|--------|-------|-----------|
| NQ/TriviaQA/PopQA | 各500 | simple |
| HotpotQA/2Wiki/MuSiQue | 各500 | multi_hop |
| HumanEval | 164 | code |
| MBPP | 500 | code |

## 遇到的问题与解决

| 问题 | 解决方案 |
|------|---------|
| PopQA 无 train split | 训练/测试均从 test split 取样 (通过 seed+shuffle 区分) |
| xanhho/2WikiMultihopQA 脚本弃用 | 替换为 ohjoonhee/2WikiMultihopQA |
| drt/musique 不存在 | 替换为 bdsaglam/musique |
| MBPP sanitized 样本量不足 (test=257) | 切换为 MBPP full config (test=500) |
| MBPP full 字段名不同 (text vs prompt) | 代码兼容处理: `ex.get("prompt") or ex.get("text")` |
| GPQA Diamond 需要 gated access | 使用公开镜像 hendrydong/gpqa_diamond_mc |
| LiveCodeBench code_generation 9.3GB 太大 | 使用精选子集 cassanof/livecodebench_lite_filtered (202条) |

## 数据验证结果 (2026-03-25)

所有文件已通过自动化验证:

| 文件 | 行数 | Source 分布 | 空答案 | 缺失字段 | 状态 |
|------|------|------------|--------|---------|------|
| train_qa.jsonl | 6,000 | 6×1000 均匀 | 0 | 无 | OK |
| test_qa.jsonl | 3,000 | 6×500 均匀 | 0 | 无 | OK |
| train_code.jsonl | 374 | MBPP 374 | 0 | 无 | OK |
| test_code.jsonl | 664 | HumanEval 164 + MBPP 500 | 0 | 无 | OK |
| train_mixed.jsonl | 6,374 | QA 6000 + Code 374 | 0 | 无 | OK |

| test_gpqa.jsonl | 198 | GPQA Diamond (physics/chem/bio) | 0 | 无 | OK |
| test_livecode.jsonl | 202 | LCB easy/medium/hard | 0 | 无 | OK |

## 对后续计划的影响

- 是否需要调整 W2 计划? [ ] 否
- T1.1/T1.2 完成顺利，无阻塞
- T1.3 (GPQA + LiveCodeBench) 完成 → test_gpqa.jsonl (198条) + test_livecode.jsonl (202条)
- T1.4 (SFT 热身) 200条生成完成，GPT-4o API 200/200 成功率 100%
- T1.5-T1.7 依赖 A100 硬件环境

## 下周预览
- W2 核心目标: SFT 热身训练 + GRPO 训练 × 3 seeds
- 阻塞风险: T1.5 (A100 环境) 若未完成会阻塞 W2 全部任务
