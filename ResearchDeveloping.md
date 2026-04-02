# 研究开发记录

本文记录当前围绕 `InternVLA-N1 / DualVLN` 做的两条研发线：

- 训练数据结构与自建采集流程
- `VLN-CE` 中 `STOP` 决策失败现象与可改进方向

---

## 1. 当前结论概览

当前已经确认：

1. 我们可以自己在 Habitat + MP3D 中构造一套 `N1-like` 数据。
2. 这套数据已经不只是“格式相似”，而是已经能被正式 `N1` 的 dataset / collator / model forward 读入。
3. `VLN-CE` 当前 `STOP` 决策与其他动作属于同级输出，没有单独的到点校验模块。
4. 这意味着如果模型较早输出 `STOP`，系统会直接执行，再由 Habitat 的 success metric 事后判定失败。
5. 因此，后续很值得单独加入一个 `STOP check` 或 `STOP gate` 模块。

---

## 2. 训练数据的发现

### 2.1 N1 训练数据不是原始 benchmark json

`N1` 真正训练时读的，不是普通 `VLN-CE` 的 `train.json.gz / val_unseen.json.gz` 这种 episode 定义文件，而是轨迹形式的数据。

其核心结构是：

- `meta/episodes.jsonl`
- `meta/tasks.jsonl`
- `meta/info.json`
- `data/chunk-000/episode_xxxxxx.parquet`
- `videos/chunk-000/observation.images.rgb.*/*`
- `videos/chunk-000/observation.images.depth.*/*`

也就是说：

- 原始 benchmark 数据定义任务
- N1 训练数据记录任务执行过程

这类数据本质上是：

- instruction
- 逐帧 RGB / depth
- 逐帧 action
- 逐帧 pose
- 逐帧像素目标 supervision

---

### 2.2 N1 训练 supervision 的关键不只是 action

我们确认 loader 真正依赖的关键字段包括：

- `action`
- `pose.125cm_30deg`
- `goal.125cm_30deg`
- `relative_goal_frame_id.125cm_30deg`

其中最重要的发现是：

- `goal.*` 不是最终 goal 点，而是**当前帧对应的 future pixel goal**
- `relative_goal_frame_id.*` 表示这个像素目标对应未来第几帧

这说明 `N1` 的训练重点不是传统 imitation learning 里单纯预测下一个离散动作，而是：

- 让高层模型学会输出视觉空间中的中间目标
- 再由低层模块把这个目标转成局部轨迹/动作

---

### 2.3 官方 N1 instruction 是分层生成的

从论文和本地 sample 可以确认，官方 N1 的文本不是简单模板。

其结构大致是：

- `sub_instruction`
- `revised_sub_instruction`
- `sum_instruction`

我们目前的理解是：

1. 先根据轨迹切出若干 `sub-clip`
2. 为每个 `sub-clip` 生成局部 instruction
3. 对局部 instruction 做 rewrite
4. 再生成一条总 instruction

我们已经复现了一个近似版本：

- 先做关键帧切片
- 再用 `gpt-4o` 生成 `fine_grained / revised / long_instruction`

虽然模型与论文不同，但流程结构已经接近。

---

### 2.4 官方 N1 的段落一般没有我们最初切得那么碎

通过对本地 sample 的统计，发现：

- 大多数轨迹只切成 `1~3` 段
- 常见的单段长度约在 `80~100` 帧量级

这说明我们最初那种：

- 频繁切很多短 sub-clip
- 最后 summary 很长很碎

并不符合官方 N1 风格。

因此后来做了两类修正：

1. 更保守的关键帧规则
2. summary 明确做去重和合并

这样生成出来的 instruction 更接近官方 N1 样式。

---

## 3. 我们已经完成的自建数据流程

### 3.1 当前已经具备的能力

目前已经实现：

1. 在 Habitat + MP3D 中随机采样起点终点
2. 生成 shortest path 风格专家轨迹
3. 采集 RGB / depth / pose / action
4. 生成 `goal.125cm_30deg`
5. 生成 `relative_goal_frame_id.125cm_30deg`
6. 转换成 `N1-like` 数据结构
7. 自动生成 instruction
8. 成功喂入 N1 的正式训练链路做 smoke test

这说明：

- 我们已经不是停留在“分析官方数据”
- 而是具备了初步的“自己造数据并接训练”的能力

---

### 3.2 当前自建数据的不足

虽然当前链路已经跑通，但和正式 N1 数据相比仍然有差距：

1. 轨迹质量仍然偏简化
   - 当前更接近 shortest-path / 重采样
   - 不是论文里 `A* + ESDF + waypoint optimization + smoothing` 那一套

2. instruction 还不够稳定
   - 尤其 `left/right` 容易出错
   - 局部 landmark 的空间关系仍可能 hallucinate

3. pixel goal 规则仍是近似版
   - 当前是固定 horizon 内的最远可见/可投影 future frame
   - 更接近论文的版本应是 `farthest visible waypoint`

4. 数据规模仍很小
   - 当前只构造了小规模验证集
   - 还不足以直接支撑正式训练

---

## 4. 喂入训练的验证与发现

### 4.1 数据加载已经跑通

目前已经确认：

- dataset 构建成功
- collator 成功
- batch shape 正常

这说明我们的目录结构、字段命名、instruction 写回方式已经对齐到正式 N1 loader 的要求。

---

### 4.2 forward smoke test 已经跑通

我们已经用本地 checkpoint 做了最小 forward 验证。

结果：

- 成功 forward
- 得到有效 `loss`

这说明：

- 这份数据不是“看起来像”
- 而是已经进入到了正式模型 forward

这是一个非常关键的里程碑。

---

### 4.3 训练接入过程中暴露出的工程问题

这轮工作里已经暴露并修复过多类问题，包括：

1. `json / parquet` 兼容问题
2. `rgb.125cm_0deg` 路径兼容问题
3. `pixel_goal_only` 下 sample 选择问题
4. `bf16 / float32` dtype mismatch

这些问题说明：

- 自建数据不仅要“采出来”
- 还要严格对齐训练代码的隐含假设

---

## 5. VLN-CE 中 STOP 的发现

### 5.1 STOP 目前和其他动作是同级输出

当前 `VLN-CE` 代码中，模型输出文本后统一解析成动作：

- `STOP`
- `FORWARD`
- `LEFT`
- `RIGHT`
- `LOOKDOWN`

也就是说：

- `STOP` 不是特殊控制分支
- 它只是模型输出的一个普通动作符号

当前实现里并没有：

- 在执行 `STOP` 前单独判断“是否已经足够接近目标”
- 在执行 `STOP` 前做二次确认

---

### 5.2 当前的成功判定是事后进行的

当模型输出 `STOP` 后，系统会：

1. 执行动作
2. episode 结束
3. Habitat success metric 再检查当前位置是否满足成功条件

也就是说：

- `STOP` 是否正确，不是执行前决定
- 而是执行后由 metric 事后判定

这会带来一个直接问题：

- 如果模型在离目标仍然较远时输出 `STOP`
- 系统不会拦截
- 最终只会在结果里看到 `success = 0`

---

### 5.3 失败案例已经明确说明 STOP 是个单独薄弱点

在 `VLN-CE` 的失败样本里，`STOP` 问题不是偶发噪声，而是一个可重复观测到的独立薄弱点：

- 一部分 failure 是已经到过目标附近，但停得不准。
- 更大一部分 failure 是前序路线已偏，但最后仍以错误 `STOP` 结束 episode。

这说明后续改进不应只围绕 planner 或 `pixel goal`，还应单独建模“最终停靠条件是否已经满足”。

---

## 6. Qwen verifier 方向的探索结论

我们已经做过一条不需要额外训练的最小实验：

- 当主模型输出 `STOP` 时
- 再单独调用一次同一个 `Qwen2.5-VL`
- 让它判断当前视角是否真的满足最终停靠条件

这条线的探索结论已经比较明确：

1. 自由生成式 verifier 不稳定  
   它会继续输出导航 token（如 `←←←←`），而不是专心回答停靠判断。

2. `YES/NO` 单 token 打分仍然不可靠  
   分数长期几乎不变，说明这类判别很容易被语言先验主导，而不是被当前图像显著驱动。

3. 即使 reject `STOP`，恢复策略也很关键  
   如果只是拒停然后走一个固定 fallback，系统会持续回到 `STOP` 倾向。

因此：

- `Qwen prompt verifier` 可以作为探索工具
- 但不适合作为最终 `STOP` 解决方案

---

## 7. stop phrase 对齐方向

相比泛化的 `should_stop` 或 `YES/NO` verifier，更合理的任务定义是：

- 输入：当前视角图像
- 文本：instruction 中最终停靠描述，尤其是最终物体/地标短语
- 输出：当前图像是否与该最终停靠语义对齐

这个方向更接近问题本质：

- 当前系统里的 `STOP` 更像“流程走完了”
- 而不是“我已经视觉确认到了最终停靠位置”

因此，我们新建了一条更聚焦的监督线：

- 从公开 `VLN-CE video20` 结果中取最终停下帧和中间负样本
- 从 instruction 中抽取：
  - `stop_phrase`
  - `stop_object_phrase`
- 用图文对比学习来做终点语义对齐

---

## 8. stop phrase 数据构造

新增工具：

- [scripts/data_collect/stop_alignment_utils.py](/mnt/data/0923_Interndata/InternNav/scripts/data_collect/stop_alignment_utils.py)
- [scripts/data_collect/build_stop_phrase_manifest_from_eval.py](/mnt/data/0923_Interndata/InternNav/scripts/data_collect/build_stop_phrase_manifest_from_eval.py)

其中：

- `extract_stop_phrase(instruction)`  
  抽取最终停靠句子

- `extract_stop_object_phrase(instruction)`  
  尽量进一步抽取最终物体/地标短语  
  例如：
  - `Wait under wooden rafter` -> `wooden rafter`

当前 manifest：

- [/mnt/data/0923_Interndata/tmp_stop_phrase/video20_stop_phrase_manifest.jsonl](/mnt/data/0923_Interndata/tmp_stop_phrase/video20_stop_phrase_manifest.jsonl)

当前统计：

- `663` 条样本
- `40` 条正样本
- `623` 条负样本

说明：

- 目前正负极不平衡
- 且 `stop_object_phrase` 仍有一部分样本抽得不够理想，例如 `Stop there`
- 这部分后续仍需继续清洗

---

## 9. LongCLIP 对比学习版本

我们没有继续沿用“Qwen 联合编码 + MLP 二分类”作为主方向，而是新建了更贴近图文对齐的 `LongCLIP` 版本：

- [scripts/data_collect/train_stop_phrase_longclip.py](/mnt/data/0923_Interndata/InternNav/scripts/data_collect/train_stop_phrase_longclip.py)

### 9.1 初版问题

最初尝试直接全量微调 `LongCLIP`，很快出现：

- 相似度统计 `NaN`
- 训练后 embedding 不稳定

因此不再采用“全 backbone 直接小样本微调”的方式。

### 9.2 当前稳定版结构

当前稳定版改成：

- 冻结 `LongCLIP` 编码器
- 只训练两个小投影头：
  - `image_proj`
  - `text_proj`
- 再配一个可学习 `logit_scale`

也就是：

- 图像：`LongCLIP encode_image`
- 文本：`LongCLIP encode_text`
- 然后：
  - 投影
  - 归一化
  - 对比学习

这比前面的 MLP 分类头更符合“公开终止帧 vs 最终停靠短语”的任务定义。

### 9.3 LongCLIP 权重

按 repo 自带 README 的方式下载了：

- [/mnt/data/0923_Interndata/InternNav/checkpoints/clip-long/longclip-B.pt](/mnt/data/0923_Interndata/InternNav/checkpoints/clip-long/longclip-B.pt)

### 9.4 smoke test 结果

我们已完成一轮最小 smoke test：

- manifest：`video20_stop_phrase_manifest.jsonl`
- max samples：`128`
- batch size：`8`
- epoch：`1`

输出目录：

- [/mnt/data/0923_Interndata/tmp_stop_phrase/run_longclip_smoke2](/mnt/data/0923_Interndata/tmp_stop_phrase/run_longclip_smoke2)

结果：

- `train_loss = 1.9751`
- `train pos_mean = 0.0587`
- `train neg_mean = 0.0403`
- `train margin_acc = 0.9391`
- `val pos_mean = 0.0486`
- `val neg_mean = 0.0474`
- `val margin_acc = 0.9231`

这些数还不足以说明它已经可用于线上 `STOP gate`，但至少说明：

1. `LongCLIP` 对比学习链已经打通
2. 当前稳定版不会再像全量微调那样出现 `NaN`
3. 公开终止帧 + stop phrase/object phrase 这条监督路径已经可以继续扩展

---

## 10. 当前最合理的下一步

基于现阶段实验，后续应优先做：

1. 清洗 `stop_object_phrase`
   - 重点处理 `Stop there` / `wait there` / `inside the room` 这类弱文本

2. 重平衡 stop phrase 数据
   - 提升正样本占比
   - 加入更难的近负样本

3. 用 `LongCLIP` 对齐分数替代当前 `STOP verifier`
   - 只在主模型想 `STOP` 时调用
   - 判断“当前视角是否真的对齐最终停靠语义”

4. 再决定是否需要把 `LongCLIP` 对齐结果与 planner 恢复逻辑联动

### 5.3.1 `pixel goal -> planner -> STOP` 的真实执行链

在当前 `VLN-CE dual-system` 代码里，需要区分两层：

- 上游 `Qwen / System2`
- 下游 `planner / System1 (nextdit_async)`

当前主流执行模式并不是“Qwen 直接频繁输出左转右转前进”，而更接近：

1. `Qwen` 输出 `LOOKDOWN`
2. `Qwen` 在 look-down 视角下输出 `pixel goal`
3. `planner` 根据该 `pixel goal` 生成一小段局部动作
4. 执行若干步后，再回到 `Qwen`

也就是说：

- `Qwen` 主要负责：
  - `LOOKDOWN`
  - `pixel goal`
  - 任务级 `STOP`
- `planner` 主要负责：
  - 根据 `pixel goal` 展开 `LEFT / RIGHT / FORWARD`

这与日志里的典型模式一致：

- `actions [5]`
- `output text: 318 261`
- `predicted goal [261, 318]`
- 接着一串 `step_id ... action 1/2/3`

其中：

- `output text: 318 261`
  - 是 `Qwen` 直接生成的文本
- 后面的连续动作
  - 是 `planner` 输出的局部动作序列

---

### 5.3.2 planner 一次不是“走到 pixel goal 为止”，而是滚动短程规划

当前 evaluator 里有两个关键常量：

- `MAX_STEPS = 8`
- `MAX_LOCAL_STEPS = 4`

含义是：

- `planner` 会围绕同一个 `pixel goal` 做局部规划
- 但一次真正连续执行的动作，最多只拿前 `4` 步
- 如果当前 `pixel goal` 还未结束，可以再次围绕同一个 `pixel goal` 重新规划一批新的 `4` 步

所以一个 `pixel goal` 不是“一次规划然后完整执行到底”，而是：

```text
pixel goal
 -> 生成一批局部动作
 -> 实际只执行前 4 步
 -> 如果 pixel goal 仍有效，再重新规划
 -> 再执行下一批 4 步
```

这解释了日志中这种现象：

```text
predicted goal [257, 163]
step_id 49 action 1
step_id 50 action 1
step_id 51 action 3
step_id 52 action 1
local_actions [2, 1, 2, 1]
step_id 53 action 2
step_id 54 action 1
step_id 55 action 2
step_id 56 action 1
```

这里：

- `49-52`
  - 是第一批局部规划得到的 4 步
- `53-56`
  - 不是“剩余 4 步”
  - 而是对同一个 `pixel goal` 重新规划出的第二批 4 步

---

### 5.3.3 `pixel goal` 什么时候结束

当前 `pixel goal` 的结束并不是“显式到达该点”才结束，而是更偏工程式控制。

主要结束条件有：

1. 局部 planner 产出 `STOP`
2. 围绕该 `pixel goal` 的连续执行预算超过 `MAX_STEPS`
3. episode 本身结束

这意味着：

- `pixel goal` 更像一个短程局部控制上下文
- 不是一个必须被严格几何到达的硬 waypoint

也就是说：

- 当前系统没有一个单独的“已经到达这个 pixel goal”判定器
- 更多是靠步数预算和局部 planner 状态来结束当前 `pixel goal`

---

### 5.3.4 必须区分两种 `STOP`

当前日志里出现的 `STOP` 实际上有两种，不能混为一谈。

#### 1. planner 的局部 `STOP`

例如：

```text
local_actions [1, 1, 0, 0]
```

这里的 `0` 表示：

- 局部 planner 认为当前这段局部轨迹可以收尾
- 当前 `pixel goal` 应结束

但这**不是任务级终止**。

在代码里，这种局部 `STOP` 会导致：

- `pixel_goal = None`
- `local_actions = []`
- 回到外层主循环

然后系统重新请求 `Qwen`。

#### 2. Qwen 的任务级 `STOP`

例如日志里的：

```text
step_id: 96 output text: STOP
actions [0]
step_id 96 action 0
```

这才是：

- `Qwen / System2` 明确决定任务结束
- evaluator 直接执行 `env.step(0)`
- Habitat 再事后判定是否成功

因此应该把两者严格区分：

- `planner STOP`
  - 结束的是当前 `pixel goal`
- `Qwen STOP`
  - 结束的是整个 episode

---

### 5.3.5 一个典型成功样本的完整链路

日志：

```text
actions [5]
step_id 89 action 5
step_id: 89 output text: 318 261
predicted goal [261, 318]
step_id 89 action 1
step_id 90 action 1
step_id 91 action 1
step_id 92 action 1
local_actions [1, 1, 0, 0]
step_id 93 action 1
step_id 94 action 1
step_id: 96 output text: STOP
actions [0]
step_id 96 action 0
success: 1.0
```

这段真实过程是：

1. `Qwen` 先输出 `LOOKDOWN`
2. `Qwen` 再输出一个 `pixel goal`
3. `planner` 先执行第一批局部动作
4. `planner` 第二次局部规划里给出局部 `STOP`
5. 当前 `pixel goal` 被清空
6. 外层循环再次请求 `Qwen`
7. `Qwen` 这时输出真正的任务级 `STOP`
8. Habitat 判定成功

这个案例很重要，因为它清楚说明：

- `planner` 不直接决定任务完成
- 它只是帮助局部逼近目标区域
- 真正 episode 结束仍然主要靠上游 `Qwen` 输出 `STOP`

我们在 `VLN-CE` 的失败样本中已经看到多类问题：

1. 早期转向错误
2. 中途漂移
3. 接近目标但没有停好
4. 明显离目标还远时就输出 `STOP`

尤其像某些失败 episode：

- 路线前面已经有偏差
- 后面虽然还在走
- 最后仍然主动输出 `STOP`
- Habitat 再判定 `success=0`

这说明：

- `STOP` 决策本身很可能是一个独立薄弱点
- 它不一定和“局部动作生成”问题完全相同

---

### 5.4 在 278 条已完成 VLN-CE episode 上恢复出的 STOP 统计

基于：

- `/mnt/data/0923_Interndata/InternNav/logs/habitat/latest_dual_system_tuned_stdout.log`

可以恢复出 `278` 条已完成 episode，其中：

- 总失败：`97`
- 有明确 `STOP` 日志的失败：`94`

这里要分两个口径：

#### 严格口径：真正以终止精度为主因的失败

用 `os = 1, success = 0` 作为近似标准，也就是：

- 已经到过目标附近
- 但没有正确停成

这类共有：

- `16 / 97`
- 约 `16.5%`

这部分最典型，说明：

- 终止判断本身不够准
- 如果单独改进 `STOP` 判别，最有机会直接转成成功

#### 宽口径：最后是错误 STOP 直接结束了 episode

如果只看日志里失败前是否明确输出了 `STOP`，则：

- `94 / 97`
- 约 `96.9%`

这说明绝大多数失败最后都以错误终止结束。

但这 `94` 条里，又要分开理解：

- 一部分是“已经接近目标，但停得不准”
- 另一大部分是“前面路线已经错了，最后又过早停下”

也就是说：

- 不能简单把这 `94` 条都归因为纯 stop 模块问题
- 但也能看出 `STOP` 是一个非常高频的失败触发点

#### 对这批失败的更细拆分

在 `os = 0` 的 `78` 条失败里：

- `70` 条是在 `150` 步以内就提前错误 `STOP`
- `8` 条在 `150-449` 步之间错误 `STOP`
- 几乎没有“接近 500 步上限才停”的模式

这说明大量失败并不是“超步数找不到目标”，而是：

- 前面导航已经偏掉
- 然后模型在错误地点过早停下

因此更准确的结论是：

- `STOP` 不是唯一问题
- 但它经常是失败被正式触发的最后一步
- 改善终止判断很可能带来明显收益
- 但收益大小取决于“拒停后是否还有能力继续找到目标”

---

## 6. 为什么后续值得加入 STOP check 模块

### 6.1 理由

当前系统是：

- 所有动作统一由主模型输出
- `STOP` 不经过单独校验

而 `STOP` 的风险在于：

- 一旦停错，整条 episode 直接失败
- 且这种错误往往发生在“已经接近目标但不够近”时
- 这类错误对 `SR / SPL` 影响非常大

因此，在工程上引入 `STOP check` 是合理的。

---

### 6.2 可行方向

后续可以考虑以下几种方案：

#### 方案 A：规则型 STOP gate

当模型输出 `STOP` 时，额外检查：

- 当前 `distance_to_goal`
- 或当前 topdown / GPS 接近度

如果未达到阈值：

- 拒绝执行 `STOP`
- 改为继续导航或重采样动作

优点：

- 最简单
- 最容易验证是否有效

缺点：

- 依赖 privileged metric
- 更像工程修补，不完全端到端

---

#### 方案 B：单独训练 STOP checker

增加一个二分类模块，输入：

- 当前 observation
- instruction
- 当前历史状态

输出：

- 现在是否应该停止

优点：

- 更接近可学习方案
- 可以和主模型 decouple

缺点：

- 需要单独构造 stop label

---

#### 方案 C：两阶段决策

主模型先输出：

- navigation action

再单独由另一个模块决定：

- 是否允许 `STOP`

---

### 6.3 已做过的最小 STOP check 实验与结论

我们已经在 `VLN-CE` 的单条失败样本 `EU6Fwq7SyZv / episode 128` 上做过一次最小实验。

实验做法：

- 不改权重
- 不改训练
- 只在 evaluator 里拦截主模型输出的 `STOP`
- 若主模型想停，则再额外做一次视觉语言判别：
  - 当前图像是否已经到达 instruction 描述的终点
- 预期输出 `YES / NO`
- 若拒绝，则改成继续向前搜索

实验配置与日志：

- 配置：
  - `/mnt/data/0923_Interndata/InternNav/scripts/eval/configs/habitat_dual_system_ep128_stopcheck_cfg.py`
- 输出：
  - `/mnt/data/0923_Interndata/InternNav/logs/habitat/ep128_stopcheck`
- 日志：
  - `/mnt/data/0923_Interndata/InternNav/logs/habitat/ep128_stopcheck_stdout.log`

实验结果：

- baseline:
  - `success = 0`
  - `os = 0`
  - `ne = 4.414`
  - `steps = 71`
- stopcheck:
  - `success = 0`
  - `os = 1`
  - `ne = 5.908`
  - `steps = 501`
  - `stop_rejects = 220`

这次实验没有成功，原因也很明确：

- 复用了当前导航模型去做 `YES / NO` 判别
- 但它没有真正进入终止判别模式
- 实际仍然在输出导航风格 token
- 日志里出现的回答是：
  - `answer = →→→→`

所以这个实验说明：

- “把 STOP 单独拿出来管”这个方向是对的
- 但“直接复用当前导航输出头做 YES/NO 判别”是无效的

这也进一步支持后续更合理的方向：

- 需要单独的 `should_stop` 判别设计
- 或至少需要单独的数据对齐与单独调用方式
- 不能简单依赖现有导航输出头的 prompt 改写

这相当于把：

- “怎么走”
- “是否该停”

拆成两套机制。

---

## 7. 当前建议的研发优先级

### 第一优先级：继续提升数据质量

包括：

1. 更稳定的 `left/right` 几何生成
2. 更接近论文的 pixel goal 选择规则
3. 更高质量的专家轨迹生成

---

### 第二优先级：批量构造小规模训练集

目标是：

- 先构造可训练的小规模自建集
- 进行 ablation 与 smoke training

---

### 第三优先级：单独研究 STOP 机制

建议先做最便宜的版本：

- 在 `VLN-CE` 上加一个规则型 `STOP gate`
- 看是否显著降低“近目标但错误 STOP”这类失败

如果有效，再决定是否引入可学习模块。

---

## 8. 当前阶段的核心判断

这轮工作带来的关键变化不是某一个脚本，而是：

- 我们已经能从零开始在 Habitat 中构造 `N1-like` 数据
- 并把它接到正式训练链路
- 同时也发现了现有 `VLN-CE` 决策里 `STOP` 是一个非常值得单独拿出来研究的薄弱点

所以后续研发可以明确分成两条：

1. **数据侧**
   - 更像 N1
   - 更稳定
   - 可批量生成

2. **策略侧**
   - 尤其是 `STOP` 单独建模 / 校验

这两条线都已经具备继续推进的基础。

---

## 9. 分支与最新 STOP 方案记录

### 9.1 分支整理

为了把不同思路隔离开，当前做了两条实验分支：

- `should_stop_head`
  - 保存轻量 `stop_head` gate 的完整实验版本
- `qwen_should_stop_verifier`
  - 从基线恢复后新开的分支
  - 用来测试“候选 `STOP` 时，让 Qwen 自己再做一次终点视觉确认”

这样可以保证：

- `should_stop_head` 保留原始二分类 gate 方案
- `qwen_should_stop_verifier` 专门承接新的 prompt-based stop verifier 思想

### 9.2 新方案的核心思路

当前更合理的最小方案不是：

- `STOP` 被拒绝后继续 `FORWARD`

而是：

1. 主流程照常运行
2. 当 Qwen 输出候选 `STOP`
3. 再额外调用一次同一个 Qwen
4. 单独问：
   - 当前图像是否真的满足 instruction 里的最终停靠条件
5. 如果回答 `YES`
   - 执行真正的 `STOP`
6. 如果回答 `NO`
   - 不再 fallback 成 `FORWARD`
   - 而是强制回到：
     - `LOOKDOWN -> pixel goal -> planner`

也就是说，这版方案的目标是：

- 把终止判定和“最终 landmark / object 的视觉确认”绑定起来

而不是只做一个后置动作拦截器。

### 9.3 代码位置

当前实现接在：

- [habitat_vln_evaluator.py](/mnt/data/0923_Interndata/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py)

新增的关键项包括：

- `enable_qwen_stop_verify`
- `qwen_stop_verify_max_new_tokens`
- `qwen_stop_reject_action`
- `stop_verify_prompt`
- `_verify_stop_with_qwen(...)`

单条实验配置是：

- [habitat_dual_system_ep128_qwenstop_cfg.py](/mnt/data/0923_Interndata/InternNav/scripts/eval/configs/habitat_dual_system_ep128_qwenstop_cfg.py)

### 9.4 这轮实现里暴露出的工程问题

第一次把 verifier 接进去之后，程序在真正遇到候选 `STOP` 时崩掉了。

报错是：

- `Image features and image tokens do not match: tokens: 0, features 391`

原因不是思路错，而是 verifier prompt 里没有显式 `<image>` token：

- processor 收到了图像
- 但文本模板里没有图像占位
- 所以 Qwen 在那一步无法对齐图像 token 和图像特征

后来已经修正为：

- 若 verifier prompt 里没有 `<image>`
- 自动在前面补上 `DEFAULT_IMAGE_TOKEN`

### 9.5 当前初步观察

在 `ep128` 上的初步运行里，这版方案已经表现出和基线不同的行为：

- 它不再沿用“拒停后直接 `FORWARD`”的闭环
- 轨迹会被重新拉回：
  - `LOOKDOWN`
  - `pixel goal`
  - `planner`

并且在当前观察窗口内，`ep128` 的行为已经明显和原 baseline 分叉：

- 原 baseline 更早进入候选 `STOP`
- 这版 verifier 下，它在相同阶段仍持续走：
  - `↓`
  - `pixel goal`
  - `planner local actions`

说明至少有一点已经成立：

- 这版修改确实改变了终止前的行为节奏
- 它不再是简单地把 `STOP` 改成 `FORWARD`

### 9.6 当前阶段判断

到目前为止，这版 `Qwen STOP verifier` 比 `stop_head + FORWARD fallback` 更合理，原因有三点：

1. 它仍然利用 instruction + 当前图像做终止判别
2. 它把拒停后的控制重新送回 planner 主链，而不是盲走一步
3. 它更贴近当前失败根因：
   - 不是完全不会停
   - 而是没有显式确认最终停靠目标

后续如果继续推进，这条线值得优先于旧的 `stop_head` 方案。

### 9.7 当前最新测试现象

在 `ep128` 上继续调通之后，已经观察到一个关键现象：

- 候选 `STOP` 真正触发了 verifier
- 日志里出现：
  - `step_id: 47 output text: STOP`
  - `stop_verify step=47 answer='←←←←' accept=False`

这说明：

1. `STOP` 候选已经被单独拦出来
2. verifier 调用链已经真正跑通
3. 拒停后没有再崩
4. 并且拒停后确实回到了：
   - `LOOKDOWN`
   - 再输出动作/像素目标
   - 重新进入 planner

也就是说，目前最关键的闭环已经成立：

```text
candidate STOP
-> verifier
-> reject
-> LOOKDOWN
-> 重新进入 local planning
```

这和之前 `stop_head` 方案的根本区别是：

- 之前是：
  - `STOP -> reject -> FORWARD`
- 现在是：
  - `STOP -> reject -> LOOKDOWN -> planner`

当前还存在的问题是：

- verifier 给出的回答仍然不是 `YES/NO`
- 这次实际回答是：
  - `←←←←`

也就是说：

- 虽然链路已打通
- 但“让 Qwen 严格进入终止判别模式”这件事仍不稳定

不过相比旧方案，当前已经至少证明：

- `STOP` 拒绝后可以被重新送回规划链
- 不会再退化成单纯的 `FORWARD` 盲走

### 9.8 二选一打分版 verifier

由于上一版 verifier 仍然会输出导航 token，而不是严格的 `YES/NO`，进一步做了一个更强约束的修改：

- 不再调用 `generate()` 让 verifier 自由生成文本
- 改为对两个候选答案直接打分：
  - ` YES`
  - ` NO`
- 用当前图像和 verifier prompt 分别计算两个候选的条件对数似然
- 分数更高的一项作为最终判定

对应实现位置：

- `HabitatVLNEvaluator._score_vlm_choices`
- `HabitatVLNEvaluator._verify_stop_with_qwen`
- 文件：
  - `/mnt/data/0923_Interndata/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py`

在 `ep128` 上，这版第一次候选 `STOP` 的日志已经变成：

- `step_id: 47 output text: STOP`
- `stop_verify step=47 answer='NO scores=[-19.75, -18.75]' accept=False`

后续行为是：

- `actions [5]`
- 先重新执行 `LOOKDOWN`
- 然后重新回到导航链，而不是直接终止

这说明：

1. verifier 已经不再输出箭头或坐标
2. `STOP` 候选能够被稳定转成二分类判断
3. 拒停后仍然走：
   - `LOOKDOWN -> pixel goal / action -> planner`

当前这版相对前一版的关键进展是：

- 解决了 verifier 输出空间不受控的问题
- 至少在第一处候选 `STOP` 上，已经得到稳定的 `NO` 判定

当前还没有完成整条 `ep128` 的最终 A/B 结果，但方向上已经比“自由文本 verifier”更可靠。

### 9.9 `ep128` 完整运行中的新失败模式

继续把二选一打分版 verifier 跑到 `ep128` 的后半段后，出现了新的明确失败模式：

- 在第一次候选 `STOP` 时：
  - `step_id: 46 output text: STOP`
  - `stop_verify step=46 answer='NO scores=[-19.625, -18.75]' accept=False`
- verifier 正常拒绝了这次 `STOP`
- 之后执行了预期中的 `LOOKDOWN`

但随后没有稳定回到：

- `LOOKDOWN -> pixel goal -> planner`

而是变成了：

- `step_id: 46 output text: →→→→`
- 之后连续多轮都输出：
  - `→→→→`

表现为：

- 连续右转
- 不再重新产生有效 pixel goal
- episode 明显进入坏循环

也就是说，当前这版虽然解决了“verifier 本身输出箭头”的问题，但仍然没有解决一个更深层的问题：

- 拒停以后，Qwen 的后续策略并不稳定
- 它不一定会重新进入局部规划模式
- 可能直接退化成连续离散动作串

因此当前结论是：

1. `YES/NO` 候选打分比自由文本 verifier 更稳定
2. 但“拒停后如何强制回到正确导航子模式”仍然没有解决
3. 仅仅把 `STOP` 拒掉，还不足以形成可靠的纠偏闭环

### 9.10 `pixel-only` 恢复 prompt 失败

在 `STOP` 被拒后，又尝试加了一层更强的恢复约束：

- 下一轮 `LOOKDOWN` 不再走通用导航 prompt
- 改成专门的 `pixel_recover_prompt`
- 明确要求：
  - 只输出下一 waypoint 坐标
  - 不允许输出 `STOP`
  - 不允许输出箭头或动作

实现位置：

- `self.pixel_recover_prompt`
- `force_pixel_only_once`
- `pixel_recover_retry`
- 文件：
  - `/mnt/data/0923_Interndata/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py`

但在 `ep128` 上，这条恢复策略仍然失败。

关键日志：

- `step_id: 55 output text: STOP`
- `stop_verify step=55 answer='NO scores=[-19.5, -18.625]' accept=False`
- 进入拒停恢复后：
  - `step_id: 55 pixel_recover_retry output text: ←←←←`

随后行为继续退化为：

- 连续 `←←←←`
- 再转成连续 `→→→→`
- 不再回到有效的 pixel goal 分支

这说明：

1. 即使在 prompt 上强制“只输出坐标”
2. 当前 Qwen 在这个状态下仍然可能回到动作输出模式
3. 也就是说，单纯依赖 prompt 约束，还不足以把拒停后的策略拉回 `pixel goal -> planner`

当前可以确定的结论是：

- `STOP` verifier 本身已经能稳定做 `YES/NO`
- 但“拒停后的恢复子策略”不能只靠 prompt 修
- 需要更强的结构性限制，或显式的模式切换逻辑

### 9.11 复用上一条有效 pixel goal 的结构性恢复

在上一步基础上，又做了一个更强的结构性恢复实验：

- 当 `STOP` 被 verifier 拒绝后
- 下一轮如果“恢复专用 prompt”仍然没有给出坐标
- 就不再允许它直接落回动作分支
- 而是直接复用 `last_valid_pixel_goal`
- 强行重新进入：
  - `pixel goal -> planner`

实现位置：

- `block_action_branch_once`
- `recover_non_digit -> reuse_last_valid_pixel_goal`
- 文件：
  - `/mnt/data/0923_Interndata/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py`

在 `ep128` 上，这次第一次拒停后的关键日志变成：

- `step_id: 46 output text: STOP`
- `stop_verify step=46 answer='NO scores=[-19.5, -18.625]' accept=False`
- `step_id: 46 pixel_recover_retry output text: ←←←←`
- `step_id: 46 recover_non_digit -> reuse_last_valid_pixel_goal [290, 315]`
- `predicted goal [290, 315]`

这说明新的结构性变化已经成立：

1. 拒停后即使恢复 prompt 失败
2. 系统也不会直接坠回纯动作模式
3. 而是会复用上一条有效 pixel goal，重新进入 planner

随后在同一条 `ep128` 里又出现了第二次候选 `STOP`：

- `step_id: 59 output text: STOP`
- `stop_verify step=59 answer='NO scores=[-19.625, -18.75]' accept=False`
- 再次触发：
  - `recover_non_digit -> reuse_last_valid_pixel_goal [290, 315]`
  - `predicted goal [290, 315]`

也就是说，相比前几版：

- 这版至少已经把“拒停后退化成纯箭头死循环”压住了
- 系统能在拒停后反复回到 planner

当前还没有拿到完整 episode 的最终成功结果，但这是目前为止最接近“形成闭环纠偏”的一版。

## 10. `stop phrase` 对齐方向

前面的探索已经说明：

- 单独问 `YES/NO` 的 prompt verifier 不可靠
- 泛化的 `should_stop` 二分类也过于粗糙

更合理的方向是把问题改写成：

- 当前图像
- 对齐 instruction 里的最终停靠描述

也就是只学习：

- “当前视角是否满足最后那句 `stop / wait / halt` 描述”

### 10.1 数据构造

新增了一个 stop phrase 抽取工具：

- `/mnt/data/0923_Interndata/InternNav/scripts/data_collect/stop_alignment_utils.py`

其中：

- `extract_stop_phrase(instruction)`
  - 从 instruction 中抽出最终停靠短语
- `build_stop_alignment_prompt(stop_phrase)`
  - 生成训练时的图像-文本对齐 prompt

基于真实 `video20` eval 结果，新增了 manifest 构造脚本：

- `/mnt/data/0923_Interndata/InternNav/scripts/data_collect/build_stop_phrase_manifest_from_eval.py`

它会从：

- `logs/habitat/video20/progress.json`
- `logs/habitat/video20/vis_debug/epoch_0/*.mp4`

抽取：

- 当前帧图像
- 原 instruction
- 自动抽出的 `stop_phrase`
- `label`

当前生成出的 manifest 为：

- `/mnt/data/0923_Interndata/tmp_stop_phrase/video20_stop_phrase_manifest.jsonl`

统计：

- `663` 条样本
- `40` 条正样本
- `623` 条负样本

从抽样结果看，`stop_phrase` 抽取已经基本对齐到最终停靠描述，例如：

- instruction:
  - `Walk into the living room ... Wait in the entrance to the other room.`
- stop_phrase:
  - `Wait in the entrance to the other room`

### 10.2 最小训练实现

新增训练脚本：

- `/mnt/data/0923_Interndata/InternNav/scripts/data_collect/train_stop_phrase_head.py`

这版不再依赖之前丢失的 `stop_head` 模块源码，而是：

1. 直接加载现有 `InternVLA-N1-DualVLN`
2. 输入：
   - 当前图像
   - `stop_phrase`
3. 取 Qwen/VLM 最后一层 hidden state 的最后 token 特征
4. 在训练脚本内部接一个小 MLP classifier：
   - `StopPhraseHead`
5. 只训这个小头，不改主模型

也就是说，这是一条独立于线上 evaluator 的最小可训路径。

### 10.3 smoke test

在小样本上已经跑通一轮：

- 训练命令：
  - `python scripts/data_collect/train_stop_phrase_head.py --manifest /mnt/data/0923_Interndata/tmp_stop_phrase/video20_stop_phrase_manifest.jsonl --model-path checkpoints/InternVLA-N1-DualVLN --output-dir /mnt/data/0923_Interndata/tmp_stop_phrase/run_smoke --epochs 1 --max-samples 64 --batch-size 4`

结果：

- `epoch 0`
- `train loss = 26.127`
- `train acc = 0.6316`
- `val loss = 27.772`
- `val acc = 0.0`

这个结果本身还不能说明方法有效，但说明两点：

1. 数据构造链是通的
2. 图像 + stop phrase 对齐训练链是通的

下一步如果继续推进，这条线的优先级高于 `YES/NO` verifier。
