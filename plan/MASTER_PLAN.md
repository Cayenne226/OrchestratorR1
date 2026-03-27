# OrchestratorR1 — NeurIPS 2026 Master Plan

> **Abstract deadline**: 2026-05-04 AOE
> **Full paper deadline**: 2026-05-06 AOE
> **硬件**: 4×A100/V100 80GB
> **状态标记**: [ ] 未开始 | [~] 进行中 | [x] 完成 | [!] 阻塞
> ⚠️ **2026-03-26 更新**: 修正 Conductor 信息（GPT-5 级 worker，87.5% GPQA，83.93% LCB），调整实验策略

---

## 目录总览

```
plan/
├── MASTER_PLAN.md                  ← 本文件（总控）
│
├── week1_data_infra/               ← 第1周: 数据准备 + 基础设施 (3/24 - 3/30)
│   ├── W1_OVERVIEW.md              ← 本周总览 + 验收标准
│   ├── T1.1_qa_datasets.md         ← 任务: QA 数据集准备
│   ├── T1.2_code_datasets.md       ← 任务: 代码数据集准备
│   ├── T1.3_gpqa_livecode.md       ← 任务: GPQA + LiveCodeBench 新增
│   ├── T1.4_sft_warmup_data.md     ← 任务: SFT 热身数据生成
│   ├── T1.5_a100_env.md            ← 任务: A100 训练环境配置
│   ├── T1.6_base_baseline.md       ← 任务: Base model baseline
│   └── T1.7_direct_strong.md       ← 任务: Direct-GPT-4o baseline
│
├── week2_training/                 ← 第2周: 核心训练 (3/31 - 4/06)
│   ├── W2_OVERVIEW.md
│   ├── T2.1_sft_warmup_train.md    ← 任务: SFT 热身训练
│   ├── T2.2_grpo_seed1.md          ← 任务: GRPO 主训练 Seed 1
│   ├── T2.3_grpo_seed2_3.md        ← 任务: GRPO Seed 2, 3
│   ├── T2.4_eval_all_seeds.md      ← 任务: 全数据集评估
│   └── T2.5_3b_parallel.md         ← 任务: 3B 对比训练
│
├── week3_baselines_ablation/       ← 第3周: 基线 + 消融 (4/07 - 4/13)
│   ├── W3_OVERVIEW.md
│   ├── T3.1_react_baseline.md      ← 任务: ReAct baseline
│   ├── T3.2_no_reactive_ablation.md← 任务: w/o reactive 消融 (⚠️关键)
│   ├── T3.3_adversarial_failure.md ← 任务: Adversarial intermediate failure
│   ├── T3.4_ablations.md           ← 任务: 其余消融实验
│   ├── T3.5_conductor_router.md    ← 任务: Conductor/Router-R1 结果
│   └── T3.6_alpha_sensitivity.md   ← 任务: α 敏感性 Pareto
│
├── week4_analysis_scaling/         ← 第4周: 分析 + Scaling (4/14 - 4/20)
│   ├── W4_OVERVIEW.md
│   ├── T4.1_heatmap.md             ← 任务: Agent 调用分布热力图
│   ├── T4.2_training_dynamics.md   ← 任务: 训练动态分析
│   ├── T4.3_pareto_curve.md        ← 任务: Pareto 前沿图
│   ├── T4.4_case_study.md          ← 任务: Reactive case study
│   ├── T4.5_scaling_7b_14b.md      ← 任务: Scaling 分析
│   └── T4.6_efficiency_grouping.md ← 任务: 简单/复杂题效率分组
│
├── week5_writing/                  ← 第5周: 论文撰写 (4/21 - 4/27)
│   ├── W5_OVERVIEW.md
│   ├── T5.1_method.md              ← 任务: Method section
│   ├── T5.2_experiments.md         ← 任务: Experiments section
│   ├── T5.3_analysis.md            ← 任务: Analysis section
│   ├── T5.4_intro_related.md       ← 任务: Intro + Related Work
│   └── T5.5_abstract_conclusion.md ← 任务: Abstract + Conclusion + Appendix
│
├── week6_polish_submit/            ← 第6周: 打磨 + 提交 (4/28 - 5/06)
│   ├── W6_OVERVIEW.md
│   ├── T6.1_supplementary_exp.md   ← 任务: 补充实验
│   ├── T6.2_figures_polish.md      ← 任务: 图表美化
│   ├── T6.3_proofread.md           ← 任务: 校对 + 匿名化
│   └── T6.4_submit.md              ← 任务: 提交
│
└── summaries/                      ← 阶段汇总
    ├── SUMMARY_W1.md               ← 第1周汇总（完成后填写）
    ├── SUMMARY_W2.md
    ├── SUMMARY_W3.md
    ├── SUMMARY_W4.md
    ├── SUMMARY_W5.md
    └── SUMMARY_W6.md
```

---

## 进度总览

| 周 | 日期 | 主题 | 任务数 | 完成 | 状态 |
|----|------|------|--------|------|------|
| W1 | 3/24-3/30 | 数据 + 基础设施 | 7 | 5/7 | [~] T1.1✓ T1.2✓ T1.3✓ T1.4✓ T1.7✓(Direct-GPT-4o跑完) T1.5/T1.6待GPU |
| W2 | 3/31-4/06 | 核心训练 | 5 | 0/5 | [ ] 需更新: T2.4 加入双 pool 评估 |
| W3 | 4/07-4/13 | 基线 + 消融 | 6 | 0/6 | [ ] ✅ 已更新: T3.1-T3.6 均加入双 pool 评估注释 |
| W4 | 4/14-4/20 | 分析 + Scaling | 6 | 0/6 | [ ] ✅ 已更新: T4.1/T4.3 双版图表, T4.4-T4.6 双 pool 分析 |
| W5 | 4/21-4/27 | 论文撰写 | 5 | 0/5 | [ ] ✅ 已更新: T5.1 双 pool 表格, T5.2 双表设计 (Table1+Table2), T5.3-T5.5 transfer 叙述 |
| W6 | 4/28-5/06 | 打磨 + 提交 | 4 | 0/4 | [ ] ✅ 已更新: T6.1 补充实验, T6.2 双版图表 |
| **总计** | | | **33** | **5/33** | |

### 已完成任务是否需要重做？

| 任务 | 双配置影响 | 需要重做？ |
|------|-----------|-----------|
| T1.1 QA 数据集 | 无 — 数据与 worker pool 无关 | ❌ 不需要 |
| T1.2 代码数据集 | 无 — 数据格式通用 | ❌ 不需要 |
| T1.3 GPQA+LCB | 无 — 测试集通用 | ❌ 不需要 |
| T1.4 SFT 热身数据 | agent_type 已区分 cheap/strong — 不受 backend model 影响 | ❌ 不需要 |
| T1.7 Direct-GPT-4o | 无 — 不涉及 agent pool | ❌ 不需要 |

> ✅ **结论: 所有已完成任务都不需要重做。** 双配置仅影响运行时的 model 映射 (AgentRegistry.worker_pool)，不影响数据或 SFT 格式。

---

## 关键里程碑

| 日期 | 里程碑 | 验收物 |
|------|--------|--------|
| 3/30 | M1: 数据就绪 + 环境通 | 所有 .jsonl + base model 能跑通 |
| 4/06 | M2: 主模型训好 | 3 seeds checkpoints + 初步 eval 数字 |
| 4/13 | M3: 全部实验完成 | Table 1 + Table 2 所有数据到位 |
| 4/20 | M4: 分析图表完成 | 所有 figures 的初版 |
| 4/27 | M5: 论文初稿 | 完整 10 页 PDF |
| 5/04 | M6: Abstract 提交 | OpenReview 提交确认 |
| 5/06 | **M7: Full Paper 提交** | **最终 PDF + supplementary** |

---

## 风险追踪

| 风险 | 状态 | 应对 |
|------|------|------|
| GRPO 不收敛 | 待观察 | 增大 SFT 数据 / 调低 lr / 先 LoRA 验证 |
| Conductor 绝对分数碾压 | **策略已调整** | ✅ 双配置策略: matched-worker 正面对标 (同等 worker 下 reactive ≥ open-loop) + cheap-worker 展示成本效率。不再回避数字战，而是在公平条件下正面对标 |
| Reviewer 要求 vs Conductor 直接对比 | **已准备** | 双配置策略: matched-worker Table 1 正面对标 + cheap-worker Table 2 展示跨模态+成本效率 |
| API 预算超支 | 待监控 | 训练全用 gpt-4o-mini / 限 max_turns |
| GPQA 数据获取 | ✅ 已解决 | 已获取 198 条 GPQA Diamond |
| 6 周写不完 | 中等风险 | Plan B: 最小可发表版本（见 RESEARCH_PLAN.md §7）|
