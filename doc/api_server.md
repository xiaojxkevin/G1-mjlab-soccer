# api_server.py 使用说明

`scripts/api_server.py` 是 Phase 2 交叉评测使用的策略服务器。`scripts/compete.py` 会在每个仿真步把同一个 raw state 字典发给双方服务器，服务器需要从 raw state 中构造自己 checkpoint 对应的 observation，并返回形状为 `(1, 29)` 的 action。

## 相对原始脚本的修改

1. 修正策略加载环境

原始脚本直接把 `ManagerBasedRlEnv` 传给 runner。现在先构造 `ManagerBasedRlEnv`，再用 `RslRlVecEnvWrapper(env_base, clip_actions=100.0)` 包装后加载策略。这与 eval 脚本一致，可以保证 runner 看到正确的 `num_actions`、observation space 和 action clipping 接口。

2. 对齐 goalkeeper observation

Goalkeeper checkpoint 使用 10 帧 history，单帧 observation 为 96 维：

```text
ball_pos_local, base_ang_vel, projected_gravity, joint_pos, joint_vel, actions
```

当前脚本使用 `_REF_DEFAULT_DOF_POS` 计算 goalkeeper 的相对关节角，并使用与 eval/training 一致的缩放：`base_ang_vel * 0.25`、`joint_vel * 0.05`。服务器内部维护 10 帧 history，最终向 policy 输入 960 维 actor observation。

同时修正了 goalkeeper history 的拼接顺序。`compete.py` 每一步只能给 API server 一帧 96 维 observation，原脚本直接按帧拼接为 frame-major：

```text
frame0_all_terms, frame1_all_terms, ..., frame9_all_terms
```

但 mjlab 的 observation manager 给 `GoalkeeperActorCritic` 的 960 维输入是 term-major：

```text
ball_pos_local_frame0..9, base_ang_vel_frame0..9, projected_gravity_frame0..9,
joint_pos_frame0..9, joint_vel_frame0..9, actions_frame0..9
```

`GoalkeeperActorCritic` 内部会基于这个 term-major 布局再转换为按帧的 history embedding。如果 API server 传入 frame-major，网络看到的各个 observation term 会错位，表现为 Phase 2 中 goalkeeper 姿态扭曲、乱动或倒地，而同一个 checkpoint 在原始 eval 脚本中正常。现在脚本通过 `stack_goalkeeper_history_term_major()` 显式重建 mjlab 的 term-major 布局。

3. 使用 `api_server_shooter.py` 的 shooter observation

Shooter 分支使用 `liberary233/CS2810-soccer-project` 中 `scripts/api_server_shooter.py` 的 Stage-II 逻辑。它会从服务器内部 `Eval-Shooter` env 的 motion command 中重建训练时使用的 reference terms，并和 `compete.py` 发来的 raw state 组合成 160 维 observation：

```text
command, projected_gravity, motion_ref_ang_vel, base_ang_vel,
joint_pos_rel, joint_vel, last_action, ball_pos_local, goal_pos_local
```

同时，shooter action 会经过 `adapt_action()` 加上 motion 初始关节姿态到 env default pose 的 offset correction。脚本支持 `--shooter-motion-index` 固定 motion，也支持 `--aim-mode center/open/adaptive` 选择射门目标；`adaptive` 会根据 goalkeeper 的 root pose 粗略估计被封堵区域并选择空档。

4. 修正 `/reset`

`/reset` 现在会调用 policy 自身的 `reset()`，同时执行内部 env 的 `reset()`，并清空服务器维护的 history buffer。这样每个 episode 不会继承上一局的 history 或 recurrent state。

5. 可选 goalkeeper static-ball guard

脚本保留了一个可选 guard：当 `GK_STATIC_GUARD=1` 时，如果球速低于阈值，goalkeeper API 会暂时输出 hold action；球速超过阈值后再释放给 checkpoint policy。该功能默认关闭，因为实验发现固定 hold action 不能在 compete 环境中稳定站立，不能作为最终解决方案。

相关环境变量：

```bash
GK_STATIC_GUARD=0          # 默认值，关闭 guard
GK_STATIC_GUARD=1          # 开启 guard
GK_STATIC_GUARD_SPEED=0.5  # 释放阈值，单位 m/s
GK_STATIC_GUARD_HISTORY=5  # 最近多少帧内超过阈值即释放
GK_STATIC_GUARD_ACTION=stand  # hold action 类型，目前支持 stand 或 zero
```

## 启动方式

在服务器上进入仓库并激活环境：

```bash
cd ~/G1-mjlab-soccer-latest
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unitree_rl_mjlab
```

启动 shooter API：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python scripts/api_server.py \
  --checkpoint checkpoints/phase2_external/shooter_model_6499.pt \
  --port 8000 \
  --task shooter \
  --device cuda:0 \
  --aim-mode adaptive
```

启动 goalkeeper API：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python scripts/api_server.py \
  --checkpoint checkpoints/phase2_external/goalkeeper_latest.pt \
  --port 8001 \
  --task goalkeeper \
  --device cuda:0
```

默认不建议开启 static-ball guard。若只想做实验，可以这样启动 goalkeeper：

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl GK_STATIC_GUARD=1 python scripts/api_server.py \
  --checkpoint checkpoints/phase2_external/goalkeeper_latest.pt \
  --port 8001 \
  --task goalkeeper \
  --device cuda:0
```

## 配合 compete.py 测试

无头测试 10 个 episode：

```bash
python scripts/compete.py \
  --shooter-api http://127.0.0.1:8000 \
  --goalkeeper-api http://127.0.0.1:8001 \
  --headless \
  --num-trials 10 \
  --device cuda:0
```

使用 Viser 可视化：

```bash
python scripts/compete.py \
  --shooter-api http://127.0.0.1:8000 \
  --goalkeeper-api http://127.0.0.1:8001 \
  --viewer viser \
  --device cuda:0
```

如果 Viser 在服务器上监听 `8080`，本地可以建立 SSH 隧道：

```bash
ssh -L 8080:127.0.0.1:8080 group
```

然后在本地浏览器打开：

```text
http://127.0.0.1:8080
```

## 接口格式

`POST /reset`：重置 policy state、内部 env 和 history buffer。

```bash
curl -X POST http://127.0.0.1:8001/reset
```

`POST /act`：输入 `compete.py` 提供的 raw state，返回 29 维 action。

返回格式：

```json
{
  "action": [[0.0, 0.0, "... total 29 values ..."]]
}
```

## 注意事项

`api_server.py` 只负责从 raw state 构造 observation 并调用 checkpoint policy。它不能解决策略本身没学会的行为，例如 goalkeeper 在 Phase 2 中等待静止球时的站立平衡问题。这个问题需要通过新的 Phase 2 训练配置解决，例如加入 delayed launch、静止等待阶段稳定奖励，以及水平/低平球轨迹分布。
