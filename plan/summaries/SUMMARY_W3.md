# Week 3 Summary: 基线 + 消融

> **日期**: 4/07 - 4/13
> **状态**: [ ] 未开始

## 任务完成情况

| ID | 任务 | 状态 | 备注 |
|----|------|------|------|
| T3.1 | ReAct baseline | [ ] | |
| T3.2 | w/o reactive 消融 ⚠️ | [ ] | |
| T3.3 | Adversarial failure | [ ] | |
| T3.4 | 其余消融 (×5) | [ ] | |
| T3.5 | Conductor/Router-R1 结果 | [ ] | |
| T3.6 | α 敏感性 Pareto | [ ] | |

## 关键实验结果

### Reactive vs Non-Reactive (⚠️ 最重要)
| 指标 | Reactive | Non-Reactive | Δ |
|------|----------|--------------|---|
| F1 (简单QA) | | | |
| F1 (多跳推理) | | | |
| F1 (代码) | | | |

**结论**: reactive 比 non-reactive 好 ___% (多跳推理)

### Adversarial Failure
| Noise Rate | Reactive F1 | Non-Reactive F1 | Δ |
|------------|-------------|-----------------|---|
| 0.0 | | | |
| 0.1 | | | |
| 0.3 | | | |
| 0.5 | | | |

### 完整 Table 2 (消融)
| Method | Avg F1 | Avg Cost | Avg Turns |
|--------|--------|----------|-----------|
| Full | | | |
| w/o reactive | | | |
| w/o critic | | | |
| w/o decomposer | | | |
| w/o refiner | | | |
| SFT-only | | | |
| Fixed-Pipeline | | | |
| α=0 | | | |

### Pareto 数据
| α | F1 | Avg Cost |
|---|-----|----------|
| 0.0 | | |
| 0.3 | | |
| 0.9 | | |

## 论文 Story 判断

基于本周数据，论文的核心 story 是否站得住?
- [ ] Reactive > Non-Reactive (差异显著)
- [ ] 涌现自适应行为 (消融各有贡献)
- [ ] 成本可控 (Pareto 前沿清晰)
- [ ] 需要调整 story (说明原因):

## 遇到的问题与解决

| 问题 | 解决方案 |
|------|---------|
| | |

## API 成本累计: $___
