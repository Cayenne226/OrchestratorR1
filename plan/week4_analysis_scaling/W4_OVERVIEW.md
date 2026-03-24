# Week 4: 分析 + Scaling (4/14 - 4/20)

## 本周目标
完成全部论文图表的数据准备和初版绘制，产出 Scaling 分析结果。这是论文 "Analysis" section 的核心素材。

## 验收标准
- [ ] Agent 调用分布热力图初版
- [ ] 训练动态分析图初版
- [ ] Pareto 前沿图初版
- [ ] 3 个 reactive case study 撰写完成
- [ ] 3B vs 7B scaling 数据到位
- [ ] 简单/复杂题效率分组统计完成

## 任务列表

| ID | 任务 | 日期 | 优先级 | 依赖 | 状态 |
|----|------|------|--------|------|------|
| T4.1 | Agent 调用分布热力图 | 4/14 | P0 | W2 eval 数据 | [ ] |
| T4.2 | 训练动态分析 | 4/15 | P0 | W2 wandb 日志 | [ ] |
| T4.3 | Pareto 前沿图 | 4/16 | P0 | T3.6 α 数据 | [ ] |
| T4.4 | Reactive case study | 4/17 | P0 | W2 GRPO ckpt | [ ] |
| T4.5 | Scaling 分析 (3B vs 7B) | 4/18-19 | P1 | T2.5 3B 数据 | [ ] |
| T4.6 | 简单/复杂题效率分组 | 4/20 | P1 | W2 eval 数据 | [ ] |

## 日程安排

```
周一 4/14: T4.1 热力图（数据提取 + matplotlib 绘制）
周二 4/15: T4.2 训练动态（wandb export + 绘制）
周三 4/16: T4.3 Pareto 图（汇总 α 实验数据 + 绘制）
周四 4/17: T4.4 Case study（手动挑选 + 格式化展示）
周五 4/18: T4.5 Scaling 启动（14B 训练 或 整理 3B/7B 对比）
周六 4/19: T4.5 收尾 + T4.6 效率分组统计
周日 4/20: 所有图表初版完成 → 本周汇总
```

## 产出文件清单

```
figures/
├── heatmap_agent_distribution.pdf     ← T4.1
├── training_dynamics.pdf              ← T4.2
├── pareto_curve.pdf                   ← T4.3
├── adversarial_robustness.pdf         ← T3.3 数据绘图
├── scaling_3b_7b.pdf                  ← T4.5
└── efficiency_grouping.pdf            ← T4.6

analysis/
├── case_study_1_reactive_recovery.md  ← T4.4
├── case_study_2_simple_shortcut.md    ← T4.4
└── case_study_3_complex_pipeline.md   ← T4.4
```
