# InternVLA-N1 推理方法对比文档

生成时间: 2026-03-10

---

## 📋 对比总览

| 维度 | 我的简化实现 | 官方完整实现 |
|------|------------|-------------|
| **推理方式** | 直接调用 `model.generate()` | 使用 `InternVLAN1AsyncAgent.step()` |
| **输出内容** | 仅 System 2 文本符号 (←→↑↓) | **双系统输出** (轨迹+动作) |
| **频率控制** | 手动采样（每10帧） | `plan_step_gap=4`（内置，每4帧） |
| **深度信息** | 未使用 | 必须提供（RGB-D） |
| **相机位姿** | 未使用 | 必须提供（4x4矩阵） |
| **历史帧** | 未使用 | 使用 8 帧历史（`num_history=8`） |
| **代码复杂度** | 简单（~100行） | 中等（需要完整agent） |
| **推理速度** | 快（1-2秒/次） | 慢（1-7秒/次，随历史累积） |

---

## 🔍 详细对比

### 1. 简化实现

**文件**: `generate_video_with_real_instruction.py`

**核心代码**:
```python
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from transformers import AutoTokenizer, AutoProcessor

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)
model = InternVLAN1ForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto"
)

# 推理
image = Image.open(image_path).convert('RGB')
prompt = f"Your task: {instruction}. Where should you go next?"
conversation = [{'role': 'user', 'content': [
    {'type': 'text', 'text': prompt},
    {'type': 'image', 'image': image}
]}]

text = processor.apply_chat_template(conversation, tokenize=False)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
generated_text = processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

# 输出示例: "←←←←" 或 "→→→→"
```

**优点**:
- ✅ 简单直观，易于理解
- ✅ 推理速度快
- ✅ 不需要深度图和位姿

**缺点**:
- ❌ 只有高层动作符号，没有底层轨迹规划
- ❌ 无法用于实际机器人控制
- ❌ 不符合论文描述的双系统架构

---

### 2. 官方完整实现

**文件**: `scripts/notebooks/inference_only_demo.ipynb`

**核心代码**:
```python
from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

# 配置参数
class Args:
    device = "cuda:0"
    model_path = "checkpoints/InternVLA-N1-DualVLN"
    resize_w = 384
    resize_h = 384
    num_history = 8              # 使用8帧历史
    plan_step_gap = 4            # 每4帧推理一次
    camera_intrinsic = np.array([  # D435i 相机内参
        [386.5, 0.0, 328.9, 0.0],
        [0.0, 386.5, 244.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

# 加载模型
agent = InternVLAN1AsyncAgent(args)
agent.reset()

# 推理
dual_sys_output = agent.step(
    rgb,           # numpy array (H, W, 3)
    depth,         # numpy array (H, W)
    camera_pose,   # numpy array (4, 4)
    instruction,   # str
    intrinsic=args.camera_intrinsic
)

# 双系统输出
if dual_sys_output.output_action is not None:
    # System 2 (VLN): 高层语言导航
    print(f"Action: {dual_sys_output.output_action}")
    # 输出: ['↑'] 或 ['←'] 或 ['→'] 或 ['↓'] 或 ['STOP']
else:
    # System 1 (VN): 底层视觉导航
    print(f"Trajectory: {dual_sys_output.output_trajectory}")
    # 输出: numpy array, shape (N, 2), 例如 [[0.5, 1.2], [0.8, 1.5], ...]

    print(f"Pixel goal: {dual_sys_output.output_pixel}")
    # 输出: [u, v], 图像坐标，例如 [240, 135]
```

**优点**:
- ✅ 完整的双系统输出（System 1 + System 2）
- ✅ 符合论文架构
- ✅ 可用于实际机器人控制
- ✅ 内置频率控制（`plan_step_gap`）
- ✅ 利用历史信息（8帧）

**缺点**:
- ❌ 需要提供深度图（实际场景需要RGB-D相机）
- ❌ 需要提供相机位姿（需要SLAM或其他定位系统）
- ❌ 推理速度较慢（历史帧累积导致）
- ❌ 代码复杂度较高

---

## 🎯 双系统架构详解

InternVLA-N1 是**双系统架构**，模拟人类导航的两种思维模式：

### System 1: 视觉导航 (VN) - "本能反应"
- **输入**: RGB-D 图像 + 相机内参
- **输出**:
  - `output_trajectory`: 2D 轨迹航点 (N×2 numpy array)
  - `output_pixel`: 像素级目标位置 (u, v)
- **模型**: NavDP (Navigation Diffusion Policy)
- **特点**: 快速、底层、基于视觉的路径规划

### System 2: 视觉-语言导航 (VLN) - "逻辑思考"
- **输入**: RGB 图像 + 自然语言指令
- **输出**:
  - `output_action`: 高层动作序列 ['↑'], ['←'], ['→'], ['↓'], ['STOP']
- **模型**: InternVLA LLM
- **特点**: 慢速、高层、基于语言理解的决策

### 协同工作机制

```python
if dual_sys_output.output_action is not None:
    # 使用 System 2 输出（高层决策）
    action = dual_sys_output.output_action  # 例如 ['↑']
else:
    # 使用 System 1 输出（底层轨迹）
    trajectory = dual_sys_output.output_trajectory
    pixel_goal = dual_sys_output.output_pixel
    # 将轨迹转换为控制命令（v, w）
```

**决策频率**: 每 `plan_step_gap` 帧重新规划一次

---

## 📊 输出格式对比

### 简化实现输出

```python
# 推理结果（每10帧）
results = {
    0: "←←←←",
    10: "←←←←",
    20: "↓",
    30: "↓",
    40: "→→→→",
    181: "→→→→"
}
```

### 官方实现输出

```python
# 推理结果（每4帧）
results = {
    0: {
        'action': ['←←←←'],     # System 2 输出
        'trajectory': None,
        'pixel': None
    },
    4: {
        'action': None,
        'trajectory': [          # System 1 输出
            [0.52, 1.18],
            [0.76, 1.42],
            [0.98, 1.65],
            ...
        ],
        'pixel': [245, 138]
    },
    8: {
        'action': ['↓'],
        'trajectory': None,
        'pixel': None
    },
    ...
}
```

---

## ⚙️ 关键参数说明

### plan_step_gap (推理频率)

```python
plan_step_gap = 4  # 官方推荐值
```

**含义**: 每 N 帧执行一次推理（重新规划）

**影响**:
- 值越小 → 推理越频繁 → 决策更精细，但计算量大
- 值越大 → 推理越稀疏 → 计算量小，但反应迟钝

**官方推荐**: `plan_step_gap=4`
- 在模拟环境中平衡了精度和效率
- 保证 sim-to-real 迁移质量

### num_history (历史帧数)

```python
num_history = 8  # 使用最近 8 帧
```

**含义**: 模型输入包含过去 N 帧的图像信息

**影响**:
- 帮助模型理解运动趋势
- 提升时序决策能力
- **副作用**: 推理速度随历史帧累积而变慢

### camera_intrinsic (相机内参)

```python
# D435i 默认内参（640x480分辨率）
camera_intrinsic = np.array([
    [386.5, 0.0, 328.9, 0.0],   # fx, 0, cx, 0
    [0.0, 386.5, 244.0, 0.0],   # 0, fy, cy, 0
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
```

**含义**: 相机的焦距和光心位置参数

**用途**: System 1 需要内参来计算像素坐标到3D空间的映射

---

## 🚀 实际使用建议

### 场景1: 快速原型开发/演示

**推荐**: 简化实现

```bash
python generate_video_with_real_instruction.py
```

**适用于**:
- 快速验证模型效果
- 生成演示视频
- 不需要实际控制机器人

### 场景2: 实际机器人部署

**推荐**: 官方完整实现

```python
from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

# 配置并加载
agent = InternVLAN1AsyncAgent(args)
agent.reset()

# 实时推理循环
while not done:
    # 获取传感器数据
    rgb, depth, pose = get_sensor_data()

    # 推理
    output = agent.step(rgb, depth, pose, instruction, intrinsic=K)

    # 执行控制
    if output.output_action:
        execute_action(output.output_action)
    else:
        execute_trajectory(output.output_trajectory)
```

**适用于**:
- 真实机器人控制
- 需要轨迹规划的场景
- Sim-to-real 迁移

### 场景3: 仿真环境评估

**推荐**: 官方评估框架

```bash
python tests/function_test/test_evaluator.py
```

**适用于**:
- VLN-CE benchmark 评估
- 大规模场景测试
- 与其他方法对比

---

## 📈 性能对比

基于 office_3/trajectory_68 (182帧) 的测试结果：

| 指标 | 简化实现 | 官方实现 |
|------|---------|---------|
| **总推理次数** | 19次 (每10帧) | 47次 (每4帧) |
| **单次推理时间** | 1-2秒 | 1-7秒 (历史累积) |
| **总推理时间** | ~30秒 | ~3-5分钟 |
| **视频生成时间** | ~5秒 | ~10秒 |
| **总耗时** | ~35秒 | ~5分钟 |
| **输出信息量** | 低（仅动作） | 高（动作+轨迹+像素目标） |
| **GPU显存占用** | ~8GB | ~10GB（历史帧） |

---

## 🔧 常见问题

### Q1: 为什么官方方法需要深度图？

**A**: System 1 (NavDP) 是基于 RGB-D 的视觉导航模型，需要深度信息来：
- 计算障碍物距离
- 生成可行驶的轨迹航点
- 避免碰撞

如果没有真实深度图，可以用常量填充（测试用）：
```python
depth = 10.0 * np.ones((H, W), dtype=np.float32)
```

### Q2: 为什么推理速度越来越慢？

**A**: 因为 `num_history=8`，模型需要处理历史帧：
- 第1次推理: 1帧 → 快
- 第10次推理: 8帧 → 慢（历史累积）

**解决方案**:
- 减少 `num_history`（但可能影响精度）
- 使用更快的 GPU
- 使用 flash_attention_2（需要安装）

### Q3: 如何判断应该使用哪个 System 的输出？

**A**: 由模型内部决策：
```python
if output.output_action is not None:
    # 模型认为当前需要高层决策（System 2）
    # 例如：遇到路口、需要选择方向
    use_action(output.output_action)
else:
    # 模型认为当前需要底层轨迹跟踪（System 1）
    # 例如：沿直线前进、精细避障
    use_trajectory(output.output_trajectory)
```

### Q4: 能否只用 System 2（文本动作）？

**A**: 可以，这就是简化实现的做法。但会损失：
- 精细的轨迹规划能力
- 像素级目标定位
- 底层避障能力

---

## 📚 参考资料

1. **官方 Notebook**: `scripts/notebooks/inference_only_demo.ipynb`
2. **Agent 实现**: `internnav/agent/internvla_n1_agent_realworld.py`
3. **实际部署示例**: `scripts/realworld/http_internvla_client.py`
4. **评估框架**: `tests/function_test/test_evaluator.py`

---

## 🎬 生成的视频对比

### 简化实现视频

| 文件 | 特点 | 决策变化 |
|------|------|---------|
| `inference_output_video.mp4` | 通用指令 | ❌ 单一 (全程 ←←←←) |
| `inference_real_instruction_video.mp4` | 真实指令 | ✅ 动态 (←→↓ 变化) |

### 官方实现视频

| 文件 | 特点 | 输出内容 |
|------|------|---------|
| `official_inference_office_3_trajectory_68.mp4` | 双系统 | ✅ System 1 轨迹 + System 2 动作 |

---

生成时间: 2026-03-10
作者: Claude Code
