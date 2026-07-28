"""
CHECK 1 (make-or-break): does AMO still balance the gripper-humanoid?
Run AMO standing in place for ~300 control steps on g1_amo_gripper.mjb and confirm the
robot stays UPRIGHT (pelvis height ~0.74, no fall / tip).
"""
import numpy as np, mujoco, torch
from collections import deque
from unified_env import UnifiedHumanoidEnv
from play_amo import quat_to_euler

device = "cuda" if torch.cuda.is_available() else "cpu"
policy = torch.jit.load("amo_jit.pt", map_location=device)
env = UnifiedHumanoidEnv(policy_jit=policy, robot_type="g1", device=device, headless=True)

# spawn from keyframe home (pelvis at [-0.45,-1.45,0.793], -90deg yaw facing -Y)
mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
env.data.qvel[:] = 0.0
mujoco.mj_forward(env.model, env.data)

pid = env.pelvis_id
start_quat = env.data.qpos[env.model.jnt_qposadr[
    mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, 'pelvis')]+3:
    env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, 'pelvis')]+7].copy()
target_yaw = float(quat_to_euler(start_quat)[2])

env._extract_state()
env.last_action = np.zeros(env.num_dofs, dtype=np.float32)
env._in_place_stand = True
env.gait_cycle = np.array([0.25, 0.25])
env.proprio_history = deque(maxlen=env.history_len)
env.extra_history = deque(maxlen=env.extra_history_len)
for _ in range(env.history_len):
    env.proprio_history.append(np.zeros(env.n_proprio, dtype=np.float32))
for _ in range(env.extra_history_len):
    env.extra_history.append(np.zeros(env.n_proprio, dtype=np.float32))

def amo_step(grip_cmd=0.0):
    env.viewer.commands[:] = 0.0
    env.viewer.commands[1] = target_yaw
    env._extract_state()
    obs = env._compute_observation()
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    with torch.no_grad():
        eh = torch.tensor(np.array(env.extra_history).flatten().copy(), dtype=torch.float).view(1,-1).to(device)
        leg = policy(obs_t, eh).cpu().numpy().squeeze()
    leg = np.clip(leg, -40.0, 40.0)
    scaled = leg * env.action_scale
    env.last_action = np.concatenate([leg.copy(),
        (env.dof_pos[15:]-env.default_dof_pos[15:])/env.action_scale])
    pd_target = env.default_dof_pos.copy()
    pd_target[:15] = scaled + env.default_dof_pos[:15]
    env.gait_cycle = np.remainder(env.gait_cycle + env.control_dt*env.gait_freq, 1.0)
    if env._in_place_stand and np.any(np.abs(env.gait_cycle-0.25)<0.05):
        env.gait_cycle = np.array([0.25,0.25])
    for _ in range(env.sim_decimation):
        torque = (pd_target - env.dof_pos)*env.stiffness - env.dof_vel*env.damping
        env.apply_ctrl(torque, grip_cmd)
        mujoco.mj_step(env.model, env.data)
        env._extract_state()

# settle
for _ in range(10):
    amo_step()

z0 = float(env.data.xpos[pid][2])
heights, rolls, pitches = [], [], []
for step in range(300):
    amo_step()
    z = float(env.data.xpos[pid][2])
    rpy = quat_to_euler(env.quat)
    heights.append(z); rolls.append(float(rpy[0])); pitches.append(float(rpy[1]))

heights = np.array(heights)
print(f"CHECK 1: standing 300 steps on g1_amo_gripper.mjb")
print(f"  pelvis z: start {z0:.3f}  min {heights.min():.3f}  max {heights.max():.3f}  end {heights[-1]:.3f}")
print(f"  |roll| max {np.max(np.abs(rolls)):.3f} rad  |pitch| max {np.max(np.abs(pitches)):.3f} rad")
fell = heights[-1] < 0.55 or heights.min() < 0.45
print(f"  VERDICT: {'FELL / TIPPED' if fell else 'STAYS UPRIGHT'}")

# render a still
r = mujoco.Renderer(env.model, 480, 640)
cam = mujoco.MjvCamera()
cam.lookat[:] = env.data.xpos[pid]
cam.distance = 2.2; cam.azimuth = 130; cam.elevation = -12
r.update_scene(env.data, camera=cam)
import imageio
imageio.imwrite("_unified_stand.png", r.render())
print("  saved _unified_stand.png")
