# Week 1: 数据准备 + 基础设施 (3/24 - 3/30)

## 本周目标
搭建完整的数据流水线和训练环境，确保端到端 pipeline 能跑通，并获得未训练模型的 baseline 数据。

## 验收标准
- [ ] 全部训练/测试数据 .jsonl 文件就位
- [ ] A100 环境配置完成，能启动 7B 模型 FSDP 训练
- [ ] SFT 热身数据 200 条，覆盖 6 种 agent × 3 种任务类型
- [ ] Base model baseline 数字到位
- [ ] Direct-GPT-4o baseline 数字到位

## 任务列表

| ID | 任务 | 日期 | 优先级 | 依赖 | 状态 |
|----|------|------|--------|------|------|
| T1.1 | QA 数据集准备 | 3/24-25 | P0 | 无 | [ ] |
| T1.2 | 代码数据集准备 | 3/24-25 | P0 | 无 | [ ] |
| T1.3 | GPQA + LiveCodeBench 新增 | 3/25-26 | P0 | 无 | [ ] |
| T1.4 | SFT 热身数据生成 | 3/26-27 | P0 | T1.1 | [ ] |
| T1.5 | A100 训练环境配置 | 3/24-26 | P0 | 无 | [ ] |
| T1.6 | Base model baseline | 3/28 | P1 | T1.1-T1.5 | [ ] |
| T1.7 | Direct-GPT-4o baseline | 3/29-30 | P1 | T1.1-T1.3 | [ ] |

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
├── train_qa.jsonl          ← T1.1: NQ+HotpotQA+2Wiki+MuSiQue+TriviaQA+PopQA
├── test_qa.jsonl           ← T1.1: 各500条
├── train_code.jsonl        ← T1.2: HumanEval+MBPP
├── test_code.jsonl         ← T1.2
├── test_gpqa.jsonl         ← T1.3: GPQA Diamond
├── test_livecode.jsonl     ← T1.3: LiveCodeBench
├── sft_warmup.jsonl        ← T1.4: 200条
└── train_mixed.jsonl       ← 混合训练集（QA+代码）

eval/results/
├── orch_base_7b.json       ← T1.6
└── direct_gpt4o.json       ← T1.7
```
