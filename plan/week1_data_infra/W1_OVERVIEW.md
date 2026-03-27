# Week 1: 数据准备 + 基础设施 (3/24 - 3/30)

## 本周目标
搭建完整的数据流水线和训练环境，确保端到端 pipeline 能跑通，并获得未训练模型的 baseline 数据。

## 验收标准
- [x] 全部训练/测试数据 .jsonl 文件就位
- [ ] A100 环境配置完成，能启动 7B 模型 FSDP 训练
- [x] SFT 热身数据 200 条，覆盖 6 种 agent × 3 种任务类型
- [ ] Base model baseline 数字到位
- [x] Direct-GPT-4o baseline 数字到位 ✅ (已通过 eval/run_direct_gpt4o.py 完成)

## 任务列表

| ID | 任务 | 日期 | 优先级 | 依赖 | 状态 |
|----|------|------|--------|------|------|
| T1.1 | QA 数据集准备 | 3/24-25 | P0 | 无 | [x] |
| T1.2 | 代码数据集准备 | 3/24-25 | P0 | 无 | [x] |
| T1.3 | GPQA + LiveCodeBench 新增 | 3/25-26 | P0 | 无 | [x] |
| T1.4 | SFT 热身数据生成 | 3/26-27 | P0 | T1.1 | [x] |
| T1.5 | A100 训练环境配置 | 待定 | P0 | 无 | [ ] 等待 GPU |
| T1.6 | Base model baseline | 待定 | P1 | T1.1-T1.5 | [ ] 等待 GPU |
| T1.7 | Direct-GPT-4o baseline | 3/25-26 | P1 | T1.1-T1.3 | [x] ✅ |

## ⚠️ 双配置策略说明 (2026-03-26)

已完成任务（T1.1-T1.4, T1.7）**均不需要重做**。原因：
- 数据集与 worker pool 无关
- SFT 数据中 `executor_cheap`/`executor_strong` 的区分已隐含成本分层
- 双配置仅影响 `AgentRegistry(worker_pool="cheap"|"matched")` 的运行时 model 映射
- T1.5/T1.6 计划不变：环境配置仍然是单模型训练（训练只用 cheap pool）

## 日程安排

```
周一 3/24: T1.1(QA数据) + T1.2(代码数据) + T1.5(A100环境) 并行启动
周二 3/25: T1.1/T1.2 收尾 + T1.3(GPQA/LiveCodeBench) 启动
周三 3/26: T1.3 收尾 + T1.4(SFT数据生成) 启动 + T1.5 收尾
周四 3/27: T1.4 收尾 + 数据质量检查
周五 3/28: T1.6(Base model baseline) 启动评估
周六 3/29: T1.7(Direct-GPT-4o) 启动
周日 3/30: T1.7 收尾 + 本周汇总
```

## 产出文件清单

```
data/
├── train_qa.jsonl          ← T1.1: 6000条 (NQ+TriviaQA+PopQA+HotpotQA+2Wiki+MuSiQue, 各1000) ✓
├── train_qa_matched.jsonl  ← 对齐Router-R1: 14000条 (NQ 7000 + HotpotQA 7000) ✓
├── test_qa.jsonl           ← T1.1: 3000条 (各500) ✓
├── train_code.jsonl        ← T1.2: 374条 (MBPP train) ✓
├── test_code.jsonl         ← T1.2: 664条 (HumanEval 164 + MBPP 500) ✓
├── train_mixed.jsonl       ← 混合训练集 6374条 (QA 6000 + Code 374, shuffled) ✓
├── test_gpqa.jsonl         ← T1.3: 198条 GPQA Diamond (physics/chemistry/biology) ✓
├── test_livecode.jsonl     ← T1.3: 202条 LiveCodeBench (easy/medium/hard) ✓
└── sft_warmup.jsonl        ← T1.4: 200条 (8种路径模式, 覆盖6种agent) ✓

eval/results/
├── orch_base_7b.json       ← T1.6 (待完成)
└── direct_gpt4o.json       ← T1.7 (待完成)
```
