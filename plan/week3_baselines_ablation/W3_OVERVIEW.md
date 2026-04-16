# Week 3: 基线 + 消融实验 (4/07 - 4/13)

> ⚠️ **2026-04-16 修订**: 
> 1. T3.3 Adversarial 从"消融之一"升级为**论文核心实验**（P0 最高）
> 2. 新增 T3.7 w/o think 消融
> 3. 所有评估新增延迟统计（`--measure_latency`）
> 4. 建议在 W2 训练完成后立即做早期验证（100 条，验证核心假设）

## 本周目标
完成全部基线对比和消融实验，使论文所有 Table 的数据全部到位。

**核心优先级**:
- **T3.3 (Adversarial)** 是论文 Contribution #1 的唯一直接证据 → 最高优先级
- **T3.2 (w/o reactive)** 是 T3.3 的基础 → 第二优先级
- 其余消融为 Table 4 数据 → 第三优先级

## 验收标准
- [ ] ⚠️ **Adversarial 实验完成: noise=0.3 下 reactive vs open-loop 差距 > 5%**
- [ ] w/o reactive (open-loop) 消融完成
- [ ] ReAct baseline 数字到位
- [ ] 全部消融实验完成（含新增 w/o think）
- [ ] Conductor/Router-R1 对比数据（引用或复现）
- [ ] α 敏感性 Pareto 数据
- [ ] 所有评估结果含延迟统计

## 任务列表

| ID | 任务 | 日期 | 优先级 | 依赖 | 状态 |
|----|------|------|--------|------|------|
| T3.1 | ReAct baseline | 4/07-08 | P1 | W1 数据 | [ ] |
| T3.2 | w/o reactive 消融 | 4/07-09 | **P0** | W2 SFT ckpt | [ ] |
| T3.3 | **Adversarial failure 实验** | 4/09-10 | **P0 最高 ⚠️** | W2 GRPO ckpt | [ ] |
| T3.4 | 其余消融 (×5) | 4/10-12 | P0 | W2 SFT ckpt | [ ] |
| T3.5 | Conductor/Router-R1 结果 | 4/08-09 | P1 | 无 | [ ] |
| T3.6 | α 敏感性 Pareto | 4/12-13 | P1 | W2 SFT ckpt | [ ] |
| **T3.7** | **w/o think 消融** ⚠️ 新增 | 4/12 | **P0** | W2 SFT ckpt | [ ] |

## 日程安排

```
⚠️ 前置: W2 训练完成后立即做 T3.3 早期验证（100 条，~2h）
   → 如果 reactive vs open-loop 差距 < 3%: 需要重新考虑论文核心 claim
   → 如果差距 > 5%: 核心 claim 可行，继续 W3 完整实验

周一 4/07:
  T3.1 ReAct baseline 实现 + 启动评估
  T3.2 "w/o reactive" 训练启动

周二 4/08:
  T3.2 训练运行中
  T3.5 收集 Conductor/Router-R1 的已发表结果

周三 4/09:
  T3.2 训练完成 → 评估（含延迟统计）
  T3.3 ⚠️ Adversarial failure 完整实验启动（5 个 noise rate × reactive/open-loop）

周四 4/10:
  T3.3 完成 → 绘制 F1 vs noise_rate 图 → 确认核心 claim 是否成立
  T3.4 消融训练批量启动 (w/o critic, w/o decomposer, w/o refiner, α=0, Fixed-Pipeline)

周五 4/11:
  T3.4 消融训练运行中 / 评估

周六 4/12:
  T3.4 收尾
  T3.7 w/o think 消融（修改 system prompt 移除 <think> 指令 → 重新 GRPO → 评估）
  T3.6 α 敏感性实验启动

周日 4/13:
  T3.6 完成 + T3.7 完成
  整理所有 Table 数据（含延迟列）
  本周汇总
```

## 产出文件清单

```
eval/results/
├── react_7b.json                          ← T3.1
├── orch_no_reactive_7b.json               ← T3.2 (⚠️ 关键)
├── adversarial_reactive_noise0.0.json     ← T3.3 (⚠️ 核心)
├── adversarial_reactive_noise0.1.json     ← T3.3
├── adversarial_reactive_noise0.2.json     ← T3.3
├── adversarial_reactive_noise0.3.json     ← T3.3
├── adversarial_reactive_noise0.5.json     ← T3.3
├── adversarial_openloop_noise0.0.json     ← T3.3
├── adversarial_openloop_noise0.1.json     ← T3.3
├── adversarial_openloop_noise0.2.json     ← T3.3
├── adversarial_openloop_noise0.3.json     ← T3.3
├── adversarial_openloop_noise0.5.json     ← T3.3
├── ablation_no_critic.json                ← T3.4
├── ablation_no_decomp.json                ← T3.4
├── ablation_no_refiner.json               ← T3.4
├── ablation_alpha0.json                   ← T3.4
├── ablation_fixed_pipeline.json           ← T3.4
├── ablation_no_think.json                 ← T3.7 (⚠️ 新增)
├── conductor_reference.json               ← T3.5 (引用数据)
├── router_r1_reference.json               ← T3.5
├── alpha_0.0.json                         ← T3.6
├── alpha_0.1.json
├── alpha_0.3.json
├── alpha_0.5.json
├── alpha_0.7.json
└── alpha_0.9.json

figures/
├── adversarial_robustness.pdf             ← T3.3 → Figure 5 (论文核心图)
```

## API 成本预估
| 任务 | 成本 |
|------|------|
| ReAct baseline | ~$5 |
| w/o reactive 训练+评估 | ~$8 |
| **Adversarial failure (10 runs)** | **~$6** |
| 5 个消融训练+评估 | ~$25 |
| w/o think 消融 | ~$5 |
| α 敏感性 (6 个值) | ~$20 |
| **本周总计** | **~$69** |
