> Archived document: this file is kept for historical reference and may not match the current codebase.
> For current guidance, start from `docs/README.md` and the active docs under `docs/`.

全局耗时分布（rank 2，总计 4604 ms）

  | 类别 | 耗时 | 占比 | 调用次数 | 说明 |
  |------|------|------|---------|------|
  | SAE for-k 循环（index_add/slice/index/copy） | 1866 ms | 40.5% | 55,344 | 🔴 最大瓶颈 |
  | SAE for-k 循环（mul/reduceSum） | 614 ms | 13.3% | 28,717 | 🔴 同上，逐元素计算 |
  | AI_CPU BitwiseXor（did_fire 掩码） | 676 ms | 14.7% | 96 | 🔴 意外瓶颈！回退到 CPU |
  | hcomAicpuInit（一次性初始化） | 333 ms | 7.2% | 1 | 可忽略，只跑一次 |
  | SAE EmbeddingBag 前向 | 277 ms | 6.0% | 96 | 🟡 中等 |
  | SAE IndexPut（did_fire 赋值） | 144 ms | 3.1% | 48 | 🟡 |
  | AI_CPU IndexPut | 118 ms | 2.6% | 4 | 🟡 CPU 回退 |
  | Model MatMul（Qwen3 前向） | 128 ms | 2.8% | 1844 | ✅ AI_CORE，正常 |
  | Model FlashAttention | 74 ms | 1.6% | 256 | ✅ 正常 |
  | Communication | 67 ms | 1.5% | 500 | ✅ 正常 |
  | SAE TopK | 57 ms | 1.2% | 48 | ✅ 正常 |

  三个核心发现

  1. for-k 循环仍是最大瓶颈（53.8%）

  和之前 profile_sae.py 的结论一致。FusedEncoder + FusedDecoder 反向中 k=128 的 Python 循环，总共 55,344 次 kernel launch，占了一半以上的时间。

  2. BitwiseXor 是新发现的意外瓶颈（14.7%）

  trainer.py:364 的 did_fire[sae_key][out.latent_indices.flatten()] = True 触发了 aclnnInplaceBitwiseXorTensor，而这个算子回退到了 AI_CPU，每次调用平均 7ms。这在之前的纯 SAE profiling 中看不到，因为那里没有跑
  trainer 的 did_fire 逻辑。

  3. Qwen3 模型前向本身非常快（4.4%）

  MatMul + FlashAttention 合计只占 4.4%，说明瓶颈完全在 SAE 侧。

  优化优先级（更新版）

  | 优先级 | 目标 | 占比 | 方案 |
  |--------|------|------|------|
  | 🔴 P0 | for-k 循环融合 | 53.8% | AscendC 融合 scatter-add kernel |
  | 🔴 P0 | BitwiseXor CPU 回退 | 14.7% | 改 did_fire 实现方式，避免 bool scatter（例如用 index_fill_ 或 NPU 原生的 scatter 算子替代） |
  | 🟡 P1 | EmbeddingBag 前向 | 6.0% | AscendC 向量化 kernel |
  | 🟡 P1 | IndexPut（did_fire 赋值） | 3.1% | 与 BitwiseXor 一起改掉 |


   这是 profile_sae.py 的结果（PID 2493316，单进程，device_0，纯 SAE 训练 3 步）

  耗时分布（总计 164.2 ms / 3 步 = 每步约 54.7 ms）

  | 算子 | 耗时 | 占比 | 调用次数 | 来源 |
  |------|------|------|---------|------|
  | InplaceIndexAdd | 67.8 ms | 41.3% | 771 | for-k 循环 index_add_ |
  | Slice | 31.8 ms | 19.3% | 1920 | for-k 循环切片 indices/acts |
  | Mul | 19.6 ms | 12.0% | 1158 | for-k 循环外积+逐元素乘 |
  | EmbeddingBag | 15.6 ms | 9.5% | 6 | 解码器+编码器前向 |
  | Index (gather) | 14.4 ms | 8.8% | 384 | FusedDecoder 反向取行 |
  | ReduceSum | 5.9 ms | 3.6% | 396 | FusedDecoder 反向点积 |
  | ViewCopy | 4.6 ms | 2.8% | 384 | for-k 循环内 contiguous 拷贝 |
  | TopKV2 | 1.7 ms | 1.1% | 3 | top-k 选择 |
  | MatMul + ReLU + 优化器 + 损失 | 2.6 ms | 1.6% | — | 都很快 |

  for-k 循环合计占 87.8%（index_add + slice + mul + index + reduceSum + viewCopy），与之前的分析完全吻合。

  两次 profiling 的对比总结：

  共同点

  - for-k 循环都是 #1 瓶颈，两次测试的每次调用耗时高度一致（index_add ~85us, Slice ~16us, Mul ~17-22us），证明数据可靠
  - 核心算子（EmbeddingBag、TopK、MatMul）占比都很小（~8-10%），说明"有效计算"的比例极低

  关键差异

  | | SAE-only | 真实训练 |
  |--|---------|---------|
  | for-k 循环 | 87.8% | 53.9% |
  | AI_CPU 回退 | 0% | 20.4% |
  | 模型前向 | 0% | 3.2% |
  | 通信 + 初始化 | 0% | 8.5% |

  - for-k 循环比例从 88% 降到 54%：不是它变快了，而是真实训练里多了模型前向、通信、CPU 回退等其他开销，把占比"稀释"了。绝对耗时依然是 #1
  - AI_CPU 回退是 SAE-only 测试中看不到的隐藏瓶颈（20.4%）：trainer.py 中 did_fire[indices] = True 的 bool scatter 触发了 BitwiseXor 回退到 AI_CPU，每次调用 ~7ms。这个在纯 SAE 测试里完全不存在
  - 每步 SAE 耗时对比：SAE-only ~55ms/步 vs 真实训练中 SAE 部分 ~79ms/步，差距就来自 did_fire 的 CPU 回退

  优化优先级（综合两次结果）

  | 优先级 | 目标 | SAE-only 占比 | 真实训练占比 | 方案 |
  |--------|------|-------------|------------|------|
  | 🔴 P0 | for-k 循环融合 | 87.8% | 53.9% | AscendC 融合 kernel |
  | 🔴 P0 | did_fire CPU 回退 | 不可见 | 20.4% | 改用 index_fill_ 或重写为 NPU 原生算子 |
  | 🟡 P1 | EmbeddingBag | 9.5% | 6.0% | AscendC 向量化 kernel |

  两个 P0 加起来占真实训练的 74.3%，解决后预计整体提速 3-4 倍。