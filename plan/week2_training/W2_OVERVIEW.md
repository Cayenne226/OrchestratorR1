# Week 2: 核心训练 (3/31 - 4/06)

## 本周目标
完成 SFT 热身 + GRPO 主训练（7B, 3 seeds），获得可评估的 checkpoint，并初步确认训练是否收敛。

## 验收标准
- [ ] SFT 热身 checkpoint 可正确生成 `<call>` 标签
- [ ] 3 个 seed 的 GRPO 训练完成，loss 曲线收敛
- [ ] 所有 checkpoint 的评估数字到位
- [ ] 3B 对比训练至少 1 个 seed 完成

## 任务列表

| ID | 任务 | 日期 | 优先级 | 依赖 | 状态 |
|----|------|------|--------|------|------|
| T2.1 | SFT 热身训练 (7B) | 3/31 | P0 | W1全部 | [ ] |
| T2.2 | GRPO 主训练 Seed 1 | 3/31-4/02 | P0 | T2.1 | [ ] |
| T2.3 | GRPO Seed 2, 3 | 4/02-4/05 | P0 | T2.1 | [ ] |
| T2.4 | 全数据集评估 (3 seeds) | 4/05-4/06 | P0 | T2.2-T2.3 | [ ] |
| T2.5 | 3B 对比训练 (并行) | 3/31-4/04 | P1 | W1全部 | [ ] |

## 日程安排

```
周一 3/31:
  上午: T2.1 SFT 热身 (~30min on A100)
  下午: 验证 SFT checkpoint → 启动 T2.2 GRPO Seed 1
  同时: T2.5 3B SFT + GRPO 在空余 GPU 上启动

周二-三 4/01-02:
  T2.2 GRPO Seed 1 运行中 (~8-12h)
  监控 wandb: loss 曲线、reward 变化、format valid rate
  Seed 1 完成后立即启动 Seed 2

周三-四 4/02-04:
  T2.3 Seed 2 运行 → 完成 → 启动 Seed 3
  （如果有多机: Seed 2 和 Seed 3 可并行）

周五 4/05:
  T2.4 开始全数据集评估（3 seeds × 所有数据集）
  初步整理 Table 1 数字

周六-日 4/05-06:
  T2.4 评估完成
  T2.5 3B 评估
  本周汇总: 初步判断结果是否符合预期
```

## 关键决策点

### 训练收敛判断 (T2.2 完成后)
观察 wandb 上的指标:
- **reward 均值**: 应从负值逐步上升到正值区间
- **format valid rate**: 应从 SFT 的 ~90% 保持或提升到 95%+
- **F1/EM 趋势**: 应在前 500 steps 内明显上升

**如果不收敛的应对**:
1. 检查 SFT checkpoint 质量（人工看几条输出）
2. 降低 learning rate (1e-6 → 5e-7)
3. 增大 G (16 → 24，如果显存允许)
4. 减小 max_turns (6 → 4)
5. 先用 LoRA 快速验证再全参

## 产出文件清单

```
checkpoints/
├── sft_warmup_7b/              ← T2.1
├── orch_grpo_7b_seed1/final/   ← T2.2
├── orch_grpo_7b_seed2/final/   ← T2.3
├── orch_grpo_7b_seed3/final/   ← T2.3
├── sft_warmup_3b/              ← T2.5
└── orch_grpo_3b_seed1/final/   ← T2.5

eval/results/
├── orch_grpo_7b_seed1.json     ← T2.4
├── orch_grpo_7b_seed2.json     ← T2.4
├── orch_grpo_7b_seed3.json     ← T2.4
└── orch_grpo_3b_seed1.json     ← T2.5

wandb/                          ← 训练日志
```

## API 成本预估

| 任务 | 计算 | 成本 |
|------|------|------|
| GRPO Seed 1 (7B) | 5000 prompts × G=16 × 3 calls × $0.15/1M | ~$3-5 |
| GRPO Seed 2 | 同上 | ~$3-5 |
| GRPO Seed 3 | 同上 | ~$3-5 |
| GRPO 3B | 5000 × G=16 × 3 × $0.15/1M | ~$3-5 |
| 评估（全部） | ~5000 queries × ~3 calls | ~$5-10 |
| **本周总计** | | **~$17-30** |
