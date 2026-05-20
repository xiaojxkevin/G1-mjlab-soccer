# Unitree RL Mjlab — Soccer

A humanoid robot soccer RL project built on [mjlab](https://github.com/mujocolab/mjlab.git),
using MuJoCo as the physics backend. Currently supports Unitree G1 Shooter and Goalkeeper tasks.


## Setup

Please refer to [setup_en.md](doc/setup_en.md) for installation and configuration.


## List Environments

List all registered task environments:

```bash
python scripts/list_envs.py
```

Currently registered soccer environments:
- `Unitree-G1-Naive-Shooter` — G1 at penalty spot facing goal, ball placed ahead
- `Unitree-G1-Naive-Goalkeeper` — G1 at goal line facing the ball


## Visualize the Scene

Use `play.py` with `--agent=zero` to inspect the scene layout (no policy loaded, robot holds default pose):

```bash
# View Shooter scene
python scripts/play.py Unitree-G1-Naive-Shooter --agent=zero

# View Goalkeeper scene
python scripts/play.py Unitree-G1-Naive-Goalkeeper --agent=zero
```

> `--agent=zero` outputs zero actions, so the robot stays in its default standing pose — useful for checking the relative positions of ball, goal, and robot.


## Acknowledgements

- [mjlab](https://github.com/mujocolab/mjlab.git) — training and execution framework
- [Humanoid-Goalkeeper](https://github.com/InternRobotics/Humanoid-Goalkeeper) — goalkeeper design reference
- [HumanoidSoccer](https://github.com/TeleHuman/HumanoidSoccer) — shooter design reference
