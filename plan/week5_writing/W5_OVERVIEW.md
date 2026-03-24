# Week 5: 论文撰写 (4/21 - 4/27)

## 本周目标
完成 NeurIPS 格式的完整初稿（10 页主文 + references + appendix）。

## 验收标准
- [ ] 主文 10 页，完整可读
- [ ] 所有 Table 和 Figure 嵌入
- [ ] 数学符号统一
- [ ] Related Work 覆盖 Conductor, AgentConductor, Router-R1, Prompt-R1
- [ ] 初稿 PDF 生成

## 任务列表

| ID | 任务 | 日期 | 优先级 | 状态 |
|----|------|------|--------|------|
| T5.1 | Method section | 4/21 | P0 | [ ] |
| T5.2 | Experiments section | 4/22 | P0 | [ ] |
| T5.3 | Analysis section | 4/23 | P0 | [ ] |
| T5.4 | Intro + Related Work | 4/24 | P0 | [ ] |
| T5.5 | Abstract + Conclusion + Appendix | 4/25-27 | P0 | [ ] |

## 日程安排

```
周一 4/21: T5.1 Method (MDP, policy, agent pool, generation loop, reward, training)
周二 4/22: T5.2 Experiments (setup, Table 1, Table 2)
周三 4/23: T5.3 Analysis (heatmap, dynamics, pareto, case study, scaling)
周四 4/24: T5.4 Introduction + Related Work
周五 4/25: T5.5 Abstract + Conclusion
周六 4/26: Appendix (agent prompts, full results, training details)
周日 4/27: 通读 + 统一记号 + 查错 → 完整初稿
```

## 论文结构 (页数预算)

```
Abstract                              0.3 页
1. Introduction                       1.5 页
2. Related Work                       1.0 页
3. Method                             2.5 页
   3.1 Problem Formulation (MDP)
   3.2 Orchestrator Policy
   3.3 Agent Pool Design
   3.4 Reactive Generation Loop
   3.5 Reward Function
   3.6 Training: SFT Warmup + GRPO
4. Experiments                        3.0 页
   4.1 Setup
   4.2 Main Results (Table 1)
   4.3 Ablation Studies (Table 2)
   4.4 Cost-Quality Tradeoff
5. Analysis                           1.5 页
   5.1 Emergent Adaptive Behavior
   5.2 Reactive vs Open-Loop
   5.3 Scaling Analysis
6. Conclusion                         0.2 页
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total main text:                     10.0 页

References                            ~1 页
Appendix A: Agent System Prompts
Appendix B: Full Results per Dataset
Appendix C: Training Hyperparameters
Appendix D: Case Studies
Appendix E: Compute Cost Breakdown
```

## 核心 Tables & Figures 清单

| ID | 类型 | 内容 | 位置 |
|----|------|------|------|
| Table 1 | 主结果 | 3 tracks × 基线 × 4 指标 | §4.2 |
| Table 2 | 消融 | 7 消融 × 3 指标 | §4.3 |
| Figure 1 | 架构图 | 系统总览 | §1 or §3 |
| Figure 2 | 热力图 | Agent 调用分布 | §5.1 |
| Figure 3 | 折线图 | 训练动态 | §5.1 |
| Figure 4 | 散点图 | Pareto 前沿 | §4.4 |
| Figure 5 | 折线图 | Adversarial robustness | §5.2 |
| Figure 6 | 柱状图 | Scaling 3B/7B | §5.3 |
