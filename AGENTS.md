# AGENTS.md

## 1. 项目定位

本项目为“南特704”胚胎发育阶段分类研究项目，核心任务是利用 IVF 胚胎的多焦平面显微图像，对胚胎发育阶段进行 16 类分类。

项目当前的核心研究问题不是简单提高一个 CNN 分类器的准确率，而是解决两个问题：

1. 南特704数据包含 7 个 focal planes，模型需要充分利用多焦平面信息；
2. 16 类发育阶段具有明显层级结构，但传统 coarse-to-fine hard routing 会产生级联误差，并使部分样本无法进入正确 Fine Expert。

当前方案采用：3D Focal Backbone、Focus Attention、Phase 1 coarse classification、Phase 2 三个 Fine Expert、Soft Routing 以及概率化 hierarchical prediction。

## 2. 工作原则

### 2.1 资料库优先级

本项目相关问题必须优先按照以下顺序建立认知：

1. `论文/`：建立医学背景、数据特点、相关方法和研究动机；
2. `南特704/code/`：确认当前真实代码、接口、数据流和配置；
3. 当前对话：确认最近确定的实验方案和未完成修改。

不得仅根据文件名、README 或旧代码推测当前状态。

### 2.2 修改代码前

必须先：

1. 读取当前文件实际内容；
2. 检查调用方和被调用方接口；
3. 检查 Tensor shape；
4. 检查 train / validation / test 数据流；
5. 检查配置项；
6. 检查是否引入新依赖；
7. 修改后检查 import、函数签名和返回值；
8. 尽可能执行语法检查和最小 forward smoke test；
9. 修改依赖时同步更新 `requirements.txt`；
10. 不得为了修一个问题同时改变无关实验变量。

### 2.3 实验变量隔离

一次实验原则上只改变一个核心变量。例如研究 Soft Routing 时，不应同时修改输入尺寸、数据增强、Backbone、optimizer、loss 权重或数据划分。

## 3. 当前模型架构

### 3.1 输入

标准输入：

```text
[B, 7, 1, 500, 500]
```

当前 Backbone 明确要求 7 个 focal planes、单通道、500×500 原始图像。除非设计独立实验，否则不要擅自改为 224×224。

### 3.2 Focal3DBackbone

数据流：

```text
[B,7,1,500,500]
        ↓
[B,1,7,500,500]
        ↓
Stem
[B,64,7,125,125]
        ↓
Stage 1
[B,128,7,63,63] → P2 [B,512,7,63,63]
        ↓
Stage 2
[B,256,7,32,32] → P3 [B,512,7,32,32]
        ↓
Stage 3
[B,512,7,16,16] → P4 [B,512,7,16,16]
        ↓
Patch Embedding
[B,512,7,8,8]
        ↓
[B,448,512]
```

其中 `448 = 7 × 8 × 8`。焦平面维度必须作为 3D depth 维度处理，不能错误地当作普通 channel。

## 4. Phase 1

```text
Backbone
 ↓
Focus Attention
 ↓
Coarse Head
 ↓
3 classes
```

Coarse classes：

```text
0: Pronuclear
1: Cleavage
2: Blastocyst
```

16 类到 3 类的映射：

```text
0~2   → 0
3~10  → 1
11~15 → 2
```

Phase 1 best checkpoint 作为 Phase 2 初始化权重。

## 5. Phase 2

冻结：

```text
Encoder / Backbone
Focus Attention
Coarse Head
```

可训练：

```text
CoarseEmbedding
PN MSFD
Cleavage MSFD
Blastocyst MSFD
PN Head
Cleavage Head
Blastocyst Head
```

### 5.1 Soft conditioning

Coarse probability：

```text
P(PN|x), P(CL|x), P(BL|x)
```

通过：

```text
E_coarse = Σ P(c|x) E_c
```

形成 soft coarse embedding，并加入 Fine Query。

### 5.2 三个 Fine Expert

```text
Pronuclear: 3 classes
Cleavage:   8 classes
Blastocyst: 5 classes
```

三个 Expert 对所有样本均进行计算。禁止恢复 `coarse_pred → hard mask → 单 Expert` 的 routing。

### 5.3 Fine Loss

训练时用 GT coarse 选择对应 Expert：

```text
GT PN → PN Expert
GT CL → CL Expert
GT BL → BL Expert
```

GT coarse 只能用于计算 Fine supervision，不能进入 validation/test inference routing。

### 5.4 Final Loss

最终概率：

```text
P(y|x) = P(c|x) P(y|c,x)
```

推荐在 log-space 实现：

```text
log P(y|x) = log P(c|x) + log P(y|c,x)
```

输出：

```text
final_log_probs: [B,16]
final_probs:     [B,16]
```

## 6. MSFD 接口规范

输入：

```text
query: [B,448,512]
P2: [B,512,7,63,63]
P3: [B,512,7,32,32]
P4: [B,512,7,16,16]
```

建议保持：

```python
return_attention=True  -> (fused, attention_maps)
return_attention=False -> fused
```

不要让 `return_attention=False` 返回 `(fused, None)`，否则调用方容易把 tuple 错误地传入 classification head。

## 7. 数据集与划分

每个 embryo 有多个 RUN。训练/验证/测试必须以 embryo 为单位划分，禁止将同一 embryo 的不同 RUN 随机拆到不同 subset，以避免时序相关导致的数据泄漏。

16 类：

```text
0 tPB2   1 tPNa  2 tPNf
3 t2     4 t3    5 t4    6 t5    7 t6    8 t7    9 t8    10 t9+
11 tM    12 tSB  13 tB   14 tEB   15 tHB
```

## 8. 数据增强

当前包含 rotation、translation、scale、horizontal flip、vertical flip、brightness、contrast、noise。

几何增强必须对整个 7-plane stack 使用同一组参数，不能对不同 focal planes 独立旋转、平移或缩放，否则会破坏焦平面之间的空间对应关系。

`NUM_AUG_VIEWS=2` 时，dataset sample 数与实际 forward view 数应在实验记录中区分。

## 9. 不允许的修改

禁止：

1. 未读取当前文件就生成替换代码；
2. 假定旧接口仍然存在；
3. 将 GT coarse 传入 validation/test；
4. 恢复 hard routing；
5. 一次消融实验同时修改多个核心变量；
6. 未检查 `requirements.txt` 就引入第三方依赖；
7. 直接删除 checkpoint、日志或实验结果；
8. 用未经资料库或文献支持的医学结论描述模型效果。

## 10. 实验记录规范

至少记录：Experiment ID、日期、代码版本、输入尺寸、focal planes、augmentation、NUM_AUG_VIEWS、Backbone、Phase 1 checkpoint、冻结模块、routing strategy、loss、FINAL_LOSS_WEIGHT、optimizer、learning rate、scheduler、batch size、train/val/test split、best validation accuracy、test accuracy、per-class metrics、confusion matrix。

Soft Routing 至少报告：

```text
Coarse Accuracy
Conditional Fine Accuracy
Final 16-class Accuracy
```

## 11. 最小验证要求

模型修改后至少验证：

```text
input
 ↓
backbone shapes
 ↓
focus attention
 ↓
coarse logits [B,3]
 ↓
fine logits [B,3], [B,8], [B,5]
 ↓
final_log_probs [B,16]
 ↓
final_probs [B,16]
 ↓
sum(probabilities)=1
 ↓
loss.backward()
 ↓
optimizer.step()
```

没有完成最小验证，不应宣称代码可以进行正式实验。
