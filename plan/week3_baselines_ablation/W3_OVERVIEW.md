# Week 3: 基线 + 消融实验 (4/07 - 4/13)

## 本周目标
完成全部基线对比和消融实验，使 Table 1（主结果）和 Table 2（消融）的数据全部到位。

## 验收标准
- [ ] ReAct baseline 数字到位
- [ ] "w/o reactive" 消融完成（⚠️ 最重要）
- [ ] Adversarial intermediate failure 实验完成
- [ ] 全部 7 个消融实验完成
- [ ] Conductor/Router-R1 对比数据（引用或复现）
- [ ] α 敏感性 Pareto 数据

## 任务列表

| ID | 任务 | 日期 | 优先级 | 依赖 | 状态 |
|----|------|------|--------|------|------|
| T3.1 | ReAct baseline | 4/07-08 | P1 | W1 数据 | [ ] |
| T3.2 | w/o reactive 消融 | 4/07-09 | **P0** | W2 SFT ckpt | [ ] |
| T3.3 | Adversarial failure 实验 | 4/09-10 | **P0** | W2 GRPO ckpt | [ ] |
| T3.4 | 其余消融 (×5) | 4/10-12 | P0 | W2 SFT ckpt | [ ] |
| T3.5 | Conductor/Router-R1 结果 | 4/08-09 | P1 | 无 | [ ] |
| T3.6 | α 敏感性 Pareto | 4/12-13 | P1 | W2 SFT ckpt | [ ] |

## 日程安排

```
周一 4/07:
  T3.1 ReAct baseline 实现 + 启动评估
  T3.2 "w/o reactive" 训练启动 (需要重新训练一个 GRPO)

周二 4/08:
  T3.2 训练运行中
  T3.5 收集 Conductor/Router-R1 的已发表结果

周三 4/09:
  T3.2 训练完成 → 评估
  T3.3 Adversarial failure 实验（用已有 GRPO ckpt + 修改版评估）

周四 4/10:
  T3.3 完成
  T3.4 消融训练批量启动 (w/o critic, w/o decomposer, α=0, Fixed-Pipeline)

周五 4/11:
  T3.4 消融训练运行中 / 评估

周六 4/12:
  T3.4 收尾
  T3.6 α 敏感性实验启动

周日 4/13:
  T3.6 完成 + 本周汇总
  整理 Table 1 和 Table 2 的完整数字
```

## 产出文件清单

```
eval/results/
├── react_7b.json               ← T3.1
├── orch_no_reactive_7b.json    ← T3.2 (⚠️ 关键)
├── adversarial_failure.json    ← T3.3 (⚠️ 关键)
├── ablation_no_critic.json     ← T3.4
├── ablation_no_decomp.json     ← T3.4
├── ablation_no_refiner.json    ← T3.4
├── ablation_alpha0.json        ← T3.4
├── ablation_fixed_pipeline.json← T3.4
├── conductor_reference.json    ← T3.5 (引用数据)
├── router_r1_reference.json    ← T3.5
├── alpha_0.0.json              ← T3.6
├── alpha_0.1.json
├── alpha_0.3.json
├── alpha_0.5.json
├── alpha_0.7.json
└── alpha_0.9.json
```

## API 成本预估
| 任务 | 成本 |
|------|------|
| ReAct baseline | ~$5 |
| w/o reactive 训练+评估 | ~$8 |
| Adversarial failure | ~$3 |
| 5 个消融训练+评估 | ~$25 |
| α 敏感性 (6 个值) | ~$20 |
| **本周总计** | **~$60** |
