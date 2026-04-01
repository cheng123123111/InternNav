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

### 6.4 当前已落地的最小 `should_stop` 方案

在确认 prompt 式 `STOP check` 不稳定之后，当前已经开始转向一个更小、更靠谱的方案：

- 不改主干 `Qwen / planner`
- 不重写原导航链
- 只额外增加一个轻量 `stop_head`
- 只在 `Qwen` 想输出任务级 `STOP` 时调用

当前方案的形式是：

```text
如果主模型没有输出 STOP
-> 原流程完全不变

如果主模型输出 STOP
-> 额外计算 should_stop 概率
-> 概率超过阈值才真的执行 STOP
-> 否则拒绝 STOP，改成 fallback 动作继续导航
```

这比之前的 prompt 式问答 checker 更合理，原因是：

1. 不再要求同一个导航生成头切换成 `YES/NO` 判别模式
2. 不再混用动作 token 分布和判别输出
3. 保留原网络主干与 planner，不做大改

---

### 6.5 当前代码里已经增加了哪些内容

#### 1. 模型里增加 `stop_head`

位置：

- `internnav/model/basemodel/internvla_n1/internvla_n1_arch.py`
- `internnav/model/basemodel/internvla_n1/internvla_n1.py`

当前实现：

- 在 `InternVLAN1MetaModel` 中挂了一个轻量 MLP：
  - `LayerNorm -> Linear -> GELU -> Linear -> 1`
- 新增了：
  - `extract_stop_features(...)`
  - `predict_should_stop(...)`

目前特征来源是：

- `generate_latents(...)` 生成的 `traj_latents`
- 再对 `n_query` 个 latent 做平均池化
- 输入 `stop_head`

也就是说，当前 stop 判别用的是：

- 同一套 `Qwen/System2` 视觉语言特征
- 但单独的判别头

#### 2. evaluator 里增加了默认关闭的 stop-head 开关

位置：

- `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`

新增参数：

- `enable_stop_head`
- `stop_head_ckpt`
- `stop_head_threshold`
- `stop_head_reject_action`

新增逻辑：

- 只有当主模型已经输出任务级 `STOP` 时
- 才会额外调用一次 `stop_head`
- 若未通过阈值，则拒绝 STOP

重要的是：

- 该逻辑默认关闭
- 不会影响原始评测流程
- 方便做 A/B 对照实验

#### 3. stop-head 使用单独的 stop prompt

当前没有复用原始导航 prompt，而是单独定义了：

- “判断当前视图是否已经到达最终停靠位置”

这样做的目的，是让：

- 训练时的 stop 特征抽取
- 推理时的 stop 判别

用同一类文本任务定义，避免和“输出 pixel goal / 动作”的 prompt 混在一起。

---

### 6.6 当前已增加的脚本

#### 1. 构建 stop 样本清单

文件：

- `scripts/data_collect/build_stop_manifest.py`

用途：

- 从 `N1-like` converted 数据中自动抽取 stop 训练样本
- 输出 manifest jsonl

当前规则：

- episode 最后 `k` 帧作为正样本
- 较早帧按 stride 抽负样本
- 中间靠近结束的一小段默认跳过，避免标签太脏

#### 2. 训练 stop 头

文件：

- `scripts/data_collect/train_stop_head.py`

用途：

- 冻结原始 `InternVLA-N1-DualVLN`
- 只训练 `stop_head`
- 输出：
  - `train_log.jsonl`
  - `stop_head.pt`

当前训练方式是：

- 读取 manifest
- 用 stop prompt + 当前帧图像构造输入
- 抽取 `traj_latents`
- 平均池化后喂给 `stop_head`
- 做 `BCEWithLogitsLoss`

---

### 6.7 已验证和暴露的问题

#### 已验证

1. `stop manifest` 能正常从我们现有 `short10` 数据集生成  
2. 正负样本比例合理  
3. 新增模型代码和 evaluator 代码能通过语法检查  

示例 manifest 结果：

- `206` 条样本
- `20` 条正样本
- `186` 条负样本

#### 暴露的问题

在最小 smoke test 里，最先暴露的问题不是结构问题，而是 dtype：

- 初版 `stop_head` 跟随大模型走了 `bf16`
- 导致极小规模训练时出现 `NaN loss`

当前已经做的修正：

- `stop_head` 训练时强制转到 `float32`
- `predict_should_stop(...)` 内部也按 `stop_head` 自身 dtype 对齐

这说明：

- 当前方案已经进入“可训练实现”阶段
- 主要剩下的是继续做正式训练与评测
- 不是还停留在概念阶段
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

## 9. 最小 `stop_head` 实验与 `fail4` 对照

这轮为了验证“单独建一个 `should_stop` 小头是否能立刻改善 `fail4`”，做了一个最小改动实验：

- 不改主干 `Qwen/System2`
- 不改 `planner/System1`
- 只在主模型想输出任务级 `STOP` 时，额外调用一个轻量 `stop_head`
- 只有 `stop_head` 判为允许，才真的执行 `STOP`

### 9.1 代码改动

新增或修改的关键位置：

- 模型侧 `stop_head`
  - [/mnt/data/0923_Interndata/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1_arch.py](/mnt/data/0923_Interndata/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1_arch.py)
  - [/mnt/data/0923_Interndata/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py](/mnt/data/0923_Interndata/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py)
- 训练脚本
  - [/mnt/data/0923_Interndata/InternNav/scripts/data_collect/build_stop_manifest.py](/mnt/data/0923_Interndata/InternNav/scripts/data_collect/build_stop_manifest.py)
  - [/mnt/data/0923_Interndata/InternNav/scripts/data_collect/train_stop_head.py](/mnt/data/0923_Interndata/InternNav/scripts/data_collect/train_stop_head.py)
- 评测接入
  - [/mnt/data/0923_Interndata/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py](/mnt/data/0923_Interndata/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py)
- `fail4` 对照配置
  - [/mnt/data/0923_Interndata/InternNav/scripts/eval/configs/habitat_dual_system_fail4_stophead_cfg.py](/mnt/data/0923_Interndata/InternNav/scripts/eval/configs/habitat_dual_system_fail4_stophead_cfg.py)

### 9.2 训练中暴露的问题与修正

一开始 `stop_head` 训练直接出现 `NaN`。

定位结果：

- `stop_features = model.extract_stop_features(...)` 是 finite
- `NaN` 出现在 `stop_head` 前向时
- 根因是新加 `stop_head` 的第一层 `LayerNorm` 在 `from_pretrained(...)` 的 meta 加载流程后参数变成了 `NaN`

定位脚本里看到的现象：

- `0.weight finite False min nan max nan`
- `0.bias finite False min nan max nan`

修正方式：

- 在 [/mnt/data/0923_Interndata/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py](/mnt/data/0923_Interndata/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py) 新增：
  - `reset_stop_head(self)`
- 在 [/mnt/data/0923_Interndata/InternNav/scripts/data_collect/train_stop_head.py](/mnt/data/0923_Interndata/InternNav/scripts/data_collect/train_stop_head.py) 中：
  - `model.to(device)` 后显式调用 `model.reset_stop_head()`

修正后确认：

- `stop_head` 参数全部 finite
- `logits` finite
- `BCEWithLogitsLoss` finite

### 9.3 训练结果

训练产物：

- manifest:
  - [/mnt/data/0923_Interndata/tmp_stop_head/stop_manifest.jsonl](/mnt/data/0923_Interndata/tmp_stop_head/stop_manifest.jsonl)
- 训练输出目录：
  - [/mnt/data/0923_Interndata/tmp_stop_head/run_fail4_v2](/mnt/data/0923_Interndata/tmp_stop_head/run_fail4_v2)
- 训练日志：
  - [/mnt/data/0923_Interndata/tmp_stop_head/run_fail4_v2/train_log.jsonl](/mnt/data/0923_Interndata/tmp_stop_head/run_fail4_v2/train_log.jsonl)

这版 1 epoch 的结果是：

- `train loss = 4.1906`
- `train acc = 0.8216`
- `val loss = 13.3104`
- `val acc = 0.8095`

这里只能说明：

- 数值链路已经打通
- `stop_head` 已经能训练

但还不能说明：

- 它已经学到了可用于线上 gating 的可靠 stop 判别

### 9.4 `fail4` 对照实验结果

基线结果文件：

- [/mnt/data/0923_Interndata/InternNav/logs/habitat/fail4/progress.json](/mnt/data/0923_Interndata/InternNav/logs/habitat/fail4/progress.json)

stop-head 实验日志：

- [/mnt/data/0923_Interndata/InternNav/logs/habitat/fail4_stophead_stdout.log](/mnt/data/0923_Interndata/InternNav/logs/habitat/fail4_stophead_stdout.log)

这轮 `fail4_stophead` 没有跑完整体 4 条，因为在第一条基线成功样本上就已经出现了明显的坏信号，于是提前停止。

最关键的对照对象是：

- 基线第一条：
  - `scene_episode = 8194nk5LbLH_0220`
  - 基线在 [/mnt/data/0923_Interndata/InternNav/logs/habitat/fail4/progress.json](/mnt/data/0923_Interndata/InternNav/logs/habitat/fail4/progress.json) 中是成功的：
  - `success = 1.0`
  - `spl = 0.6624`
  - `ne = 2.0024`
  - `steps = 127`

stop-head 版在对应日志里表现为：

- 当 Qwen 开始连续输出任务级 `STOP` 时，`stop_head` 给出的概率始终是：
  - `prob = 0.0000`
- 所以每一次都被拒绝，并回退成：
  - `FORWARD`

日志中能直接看到：

- `step_id: 154 output text: STOP`
- `stop_head prob=0.0000 threshold=0.5000 accept=False`
- `actions [FORWARD]`

随后又连续出现：

- `step_id: 161 output text: STOP`
- `step_id: 162 output text: STOP`
- `step_id: 163 output text: STOP`
- ...
- 一直到 `step_id: 184 output text: STOP`

每次都被同样拒绝。

这说明这版最小 `stop_head` 实验的直接效果是：

- **明显过拒（over-reject）**
- 连基线本来已经能成功的终止点，也被全部压掉了

### 9.5 当前结论

这轮 `fail4` 对照可以得出一个很明确的结论：

- “单独做 `should_stop` 小头”这个方向是合理的
- 但**直接用当前这版小数据 + 极轻训练出来的 `stop_head`，马上上线 gating，并不能立竿见影改善**
- 当前最直接的问题是：
  - `stop_head` 严重偏向 `not stop`
  - 对真正该停的位置也给出接近 `0` 的概率

所以这轮实验的价值主要是：

- 证明了工程接入路径是通的
- 证明了单独 `stop_head` 可以最小侵入地插进现有 evaluator
- 同时也证明了：
  - **数据与标签设计远比“先接个头”更关键**

### 9.6 后续更合理的改进方向

如果继续做 `STOP` 方向，更合理的是：

1. 明确构造更强的 stop 标签
- 正样本：真正终止前若干帧
- 近负样本：离终点很近但还不该停
- 误停样本：失败日志里错误 `STOP` 前的帧

2. 减少类别不平衡
- 当前 manifest 负样本远多于正样本
- 会天然导致 `stop_head` 偏向“全拒”

3. 不要直接把实验头上线到全部 `fail4`
- 先在单条 episode 上离线打分
- 看 stop 概率曲线是否能在真正终点前后形成分离

4. 如果继续线上 gating
- 先只在明确 `os=1` 风险样本上试
- 或者改为“高置信拒停”而不是“默认拒停”

### 9.7 进一步尝试：放宽正样本窗口 + 调低阈值

在看到“全拒”之后，又做了一轮更宽松的实验：

- 重新构造 manifest
  - `positive-last-k = 6`
  - `negative-stride = 6`
  - `near-negative-gap = 8`
  - 数据量从 `206` 变成 `176`
  - 正样本从 `20` 变成 `60`
- 重新训练 `stop_head`
  - 输出目录：
    - [/mnt/data/0923_Interndata/tmp_stop_head/run_k6](/mnt/data/0923_Interndata/tmp_stop_head/run_k6)
  - 训练日志：
    - [/mnt/data/0923_Interndata/tmp_stop_head/run_k6/train_log.jsonl](/mnt/data/0923_Interndata/tmp_stop_head/run_k6/train_log.jsonl)
- 重新评测时把阈值从 `0.5` 降到 `0.01`

这一轮的验证集分布显示：

- `stop_head` 的概率不再贴近 0
- 典型输出落在 `0.002 ~ 0.03`
- 阈值扫描里，`0.01` 是比 `0.5` 更合理的区间

但是把它接回 `fail4` 后，仍然没有得到“立竿见影”的改善，反而出现新的问题：

1. `8194nk5LbLH_0220`
   - 基线是成功的
   - 但这轮 `stop_head` 版在 `step_id: 103` 放行了一个偏早的 `STOP`
   - 最终结果变成失败：
     - `success = 0.0`
     - `steps = 104`
     - `ne = 19.5933`

2. `EU6Fwq7SyZv_0128`
   - 仍然失败
   - 当前这版没有表现出比基线更可靠的终止行为

这说明：

- `stop_head` 的问题已经从“全拒”转成“会早停”
- 这仍然不是一个可直接上线的 gating 信号

### 9.8 当前对 STOP 的更准确判断

到这里，已经可以更明确地说：

- `STOP` 不能只靠阈值和少量 N1-like 末端帧监督解决
- 单独 `stop_head` 的方向本身是对的
- 但它需要更贴近目标域的 stop 标签
- 还需要和更强的几何/进度约束一起做，而不是单独靠一个二分类头

目前最有可能的下一步是：

1. 从真正的 VLN-CE 成功轨迹里抽 stop 样本
2. 把“接近目标但还不能停”和“已经到位可以停”分开标得更细
3. 再用这个更近分布的数据重训 stop head

如果不做这一步，继续调 threshold 只能在“全拒”和“早停”之间来回摆动。

### 9.9 从 VLN-CE 评测视频抽 target-domain stop 数据

为了避免继续用 N1-like proxy 去训 stop head，我补了一个更贴近真实分布的 manifest 构造器：

- `scripts/data_collect/build_stop_manifest_from_eval.py`

它直接读取 VLN-CE eval 输出里的：

- `progress.json`
- `vis_debug/epoch_0/*.mp4`

然后把 episode 的视频逐帧导出成图片，再按轨迹末端切 stop 标签。

当前标签策略是：

- `success == 1`
  - 最后 `positive_last_k` 帧标成正样本
- `success == 0 && oracle_success == 1 && ne <= near_goal_threshold`
  - 也把最后 `positive_last_k` 帧当作“应停”正样本
- 其他帧
  - 按 `negative_stride` 采样成负样本

这样可以把真实 VLN-CE 轨迹里“已经到位但没停”和“成功停对”的终止语义都纳进来，避免继续只看自采短任务的 proxy label。

接下来要做的是：

1. 用 `video20` 抽一版 target-domain stop manifest
2. 用这版 manifest 重训 `stop_head`
3. 再拿 `fail4` 做 A/B 对照

如果这版仍然不稳，就说明 stop 不能只靠一个末端二分类头，必须把进度/轨迹/目标点一起建模。

### 9.10 目标域平衡集 stop head 训练结果

在 `video20` target-domain manifest 上，我又做了一版更平衡的小集：

- `40` 正样本
- `40` 负样本
- 总计 `80` 条

训练配置：

- `epochs=3`
- `batch_size=4`
- `feature_batch_size=1`
- `lr=5e-4`

训练结果：

- epoch 0:
  - `train loss=2.1929`
  - `train acc=0.50`
  - `val loss=0.8433`
  - `val acc=0.75`
- epoch 1:
  - `train loss=0.3971`
  - `train acc=0.8194`
  - `val loss=0.5416`
  - `val acc=0.75`
- epoch 2:
  - `train loss=0.2585`
  - `train acc=0.8889`
  - `val loss=0.5967`
  - `val acc=0.75`

验证集概率分布：

- 负样本大多落在 `0.03` 以下
- 成功停对的正样本有一个在 `0.6257`
- 另一个 oracle-success 但终止模糊的正样本约 `0.1040`

因此这版我暂时把 stop gate 阈值定在 `0.1`，再接 `fail4` 做一次直接 A/B。

当前状态：

- `fail4_stophead_target` 正在跑
- 还没有最终 `progress.json`

这轮比前两轮更有希望的地方是：

- 不再全拒 `STOP`
- 也不再用特别激进的 `0.01` 阈值
- 标签来自真实 VLN-CE eval 视频，而不是 N1-like proxy
