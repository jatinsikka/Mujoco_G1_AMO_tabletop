"""
CHECK 2 (make-or-break): can the unified gripper-humanoid PICK without falling?

AMO balances the legs (torque, standing in place). We drive the RIGHT ARM (7 DOF: shoulder
pitch/roll/yaw, elbow, wrist roll/pitch/yaw) via DLS IK on the gripper pad-midpoint (like
ik_pick) toward a box on a pedestal -> reach down, close gripper, lift. We measure pelvis
recoil (height + tilt) during the reach-down, and whether the box is HELD.

The arm is driven through the AMO PD-TORQUE scheme (NOT position actuators): IK gives an arm
qpos TARGET; we PD-track it with torque on the right-arm + wrist DOFs each sub-step, exactly
as AMO drives its own arm targets. Legs come from AMO.
"""
import numpy as np, mujoco, torch, imageio
from collections import deque
from unified_env import UnifiedHumanoidEnv, AMO_JOINTS, WRIST_JOINTS
from play_amo import quat_to_euler

device = "cuda" if torch.cuda.is_available() else "cpu"
policy = torch.jit.load("amo_jit.pt", map_location=device)
env = UnifiedHumanoidEnv(policy_jit=policy, robot_type="g1", device=device, headless=True)
env.model_path = "g1_amo_gripper_pick.mjb"
# rebuild sim on the pick model
env.model = mujoco.MjModel.from_binary_path("g1_amo_gripper_pick.mjb")
env.model.opt.timestep = env.sim_dt
env.data = mujoco.MjData(env.model)
# recompute addresses on the pick model
env.amo_qadr = np.array([env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in AMO_JOINTS])
env.amo_dadr = np.array([env.model.jnt_dofadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in AMO_JOINTS])
env.wrist_qadr = np.array([env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in WRIST_JOINTS])
env.wrist_dadr = np.array([env.model.jnt_dofadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in WRIST_JOINTS])
env.wrist_act = np.array([mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in WRIST_JOINTS])
env.grip_act = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rg_fingers_actuator")
env.pelvis_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
env.lpad = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "rg_left_pad")
env.rpad = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "rg_right_pad")

m, d = env.model, env.data

# right-arm chain: 4 AMO arm joints (shoulder p/r/y, elbow) + 3 wrist joints = 7 DOF
RARM_JOINTS = ["right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint",
               "right_elbow_joint"] + WRIST_JOINTS
rarm_qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in RARM_JOINTS])
rarm_dadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in RARM_JOINTS])
rarm_range = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in RARM_JOINTS])
# actuator ids for the 7 arm dofs (4 AMO arm motors + 3 wrist motors)
rarm_act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in RARM_JOINTS])
# PD gains / torque limits for the arm-track (AMO arm gains + wrist gains)
rarm_kp = np.array([80,80,40,60, 40,40,40], dtype=float)
rarm_kv = np.array([2,2,1,1, 2,2,2], dtype=float)
rarm_tlim = np.array([25,25,25,25, 40,40,40], dtype=float)

box_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pick_box")

def grasp_point():
    return 0.5*(d.xpos[env.lpad] + d.xpos[env.rpad])

def grasp_jac():
    jl = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jl, None, env.lpad)
    mujoco.mj_jacBody(m, d, jr, None, env.rpad)
    return 0.5*(jl+jr)[:, rarm_dadr]

def ik_target_qpos(cart_target, damping=0.03):
    """DLS IK: current arm qpos + dq toward cart_target, clamped to joint ranges."""
    J = grasp_jac()
    err = np.clip((cart_target - grasp_point())*5.0, -0.10, 0.10)
    JJt = J @ J.T + damping**2 * np.eye(3)
    dq = np.clip(J.T @ np.linalg.solve(JJt, err), -0.15, 0.15)
    q = d.qpos[rarm_qadr] + dq
    return np.clip(q, rarm_range[:,0], rarm_range[:,1])

# ---------------- AMO + arm-track control ----------------
mujoco.mj_resetDataKeyframe(m, d, 0); d.qvel[:] = 0.0; mujoco.mj_forward(m, d)
pjoint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'pelvis')
pq0 = m.jnt_qposadr[pjoint]
target_yaw = float(quat_to_euler(d.qpos[pq0+3:pq0+7])[2])

env._extract_state()
env.last_action = np.zeros(env.num_dofs, dtype=np.float32)
env._in_place_stand = True
env.gait_cycle = np.array([0.25,0.25])
env.proprio_history = deque(maxlen=env.history_len)
env.extra_history = deque(maxlen=env.extra_history_len)
for _ in range(env.history_len): env.proprio_history.append(np.zeros(env.n_proprio, dtype=np.float32))
for _ in range(env.extra_history_len): env.extra_history.append(np.zeros(env.n_proprio, dtype=np.float32))

renderer = mujoco.Renderer(m, 480, 640)
frames = []
def grab():
    cam = mujoco.MjvCamera()
    # frame the gripper + box from the robot's right side (outside the arm), slightly above
    cam.lookat[:] = np.array([-0.14, -1.31, 0.52])
    cam.distance = 0.8; cam.azimuth = -35; cam.elevation = -25
    renderer.update_scene(d, camera=cam)
    frames.append(renderer.render())

log = {"z":[], "roll":[], "pitch":[]}
def record_state():
    z = float(d.xpos[env.pelvis_id][2]); rpy = quat_to_euler(env.quat)
    log["z"].append(z); log["roll"].append(float(rpy[0])); log["pitch"].append(float(rpy[1]))

def step(arm_qtarget, grip_cmd, rec=True):
    """One control step: AMO legs (torque) + right-arm PD-track to arm_qtarget + gripper."""
    env.viewer.commands[:] = 0.0
    env.viewer.commands[1] = target_yaw
    env._extract_state()
    obs = env._compute_observation()
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    with torch.no_grad():
        eh = torch.tensor(np.array(env.extra_history).flatten().copy(), dtype=torch.float).view(1,-1).to(device)
        leg = policy(obs_t, eh).cpu().numpy().squeeze()
    leg = np.clip(leg, -40.0, 40.0)
    scaled = leg*env.action_scale
    env.last_action = np.concatenate([leg.copy(),
        (env.dof_pos[15:]-env.default_dof_pos[15:])/env.action_scale])
    pd_target = env.default_dof_pos.copy()
    pd_target[:15] = scaled + env.default_dof_pos[:15]
    env.gait_cycle = np.remainder(env.gait_cycle + env.control_dt*env.gait_freq, 1.0)
    if env._in_place_stand and np.any(np.abs(env.gait_cycle-0.25)<0.05):
        env.gait_cycle = np.array([0.25,0.25])
    for _ in range(env.sim_decimation):
        # AMO torque for the 23 DOFs (leg+waist+arm). We OVERRIDE the right-arm 4 AMO
        # actuators with our arm-track torque below, so AMO's arm target is ignored there.
        amo_torque = (pd_target - env.dof_pos)*env.stiffness - env.dof_vel*env.damping
        amo_torque = np.clip(amo_torque, -env.torque_limits, env.torque_limits)
        d.ctrl[:23] = amo_torque
        # right-arm PD-track (4 AMO arm actuators + 3 wrist actuators) toward IK qtarget
        aq = d.qpos[rarm_qadr]; av = d.qvel[rarm_dadr]
        atrq = (arm_qtarget - aq)*rarm_kp - av*rarm_kv
        atrq = np.clip(atrq, -rarm_tlim, rarm_tlim)
        d.ctrl[rarm_act] = atrq
        d.ctrl[env.grip_act] = grip_cmd
        mujoco.mj_step(m, d)
        env._extract_state()
    if rec:
        record_state()

# settle standing (hold arm at current pose, gripper open)
for _ in range(15):
    step(d.qpos[rarm_qadr].copy(), 0.0, rec=False)
    grab()

z_settled = float(d.xpos[env.pelvis_id][2])
box0 = d.xpos[box_bid].copy()
print(f"settled pelvis z={z_settled:.3f}, box at {box0}")

# PHASE 1: reach to 6cm above box (gripper OPEN) -- pre-position ABOVE, don't ram the box
above = box0 + np.array([0,0,0.06])
for _ in range(200):
    qt = ik_target_qpos(above)
    step(qt, 0.0)
    grab()
    if np.linalg.norm(above - grasp_point()) < 0.015: break
reach_recoil_z = min(log["z"])
print(f"after approach-above: gp={grasp_point().round(3)}, dist={np.linalg.norm(above-grasp_point()):.3f}")

# PHASE 2: descend straight DOWN -- track the box XY tightly (re-center every step so the pads
# straddle it, not shove it), lower Z to box center. Gripper OPEN.
for i in range(240):
    b = d.xpos[box_bid].copy()
    tgt = np.array([b[0], b[1], b[2]])
    qt = ik_target_qpos(tgt)
    step(qt, 0.0)
    grab()
    xy = np.linalg.norm((d.xpos[box_bid]-grasp_point())[:2])
    dz = abs(grasp_point()[2] - d.xpos[box_bid][2])
    if i % 40 == 0:
        print(f"  descend {i}: gp={grasp_point().round(3)} boxz={float(d.xpos[box_bid][2]):.3f} xy={xy:.3f} dz={dz:.3f}")
    if xy < 0.012 and dz < 0.012: break
print(f"after descend: gp={grasp_point().round(3)}, dist_to_box={np.linalg.norm(d.xpos[box_bid]-grasp_point()):.3f}")

# PHASE 3: fine re-center at box center (kill XY offset), then CLOSE
for _ in range(25):
    step(ik_target_qpos(d.xpos[box_bid].copy()), 0.0); grab()
for _ in range(80):
    step(ik_target_qpos(d.xpos[box_bid].copy()), 255.0); grab()

# PHASE 4: lift straight up (CLOSED)
z_box_before_lift = float(d.xpos[box_bid][2])
for _ in range(160):
    tgt = grasp_point() + np.array([0,0,0.004])
    step(ik_target_qpos(tgt), 255.0); grab()
    if (float(d.xpos[box_bid][2]) - z_box_before_lift) >= 0.10: break

# PHASE 5: hold
for _ in range(40):
    step(ik_target_qpos(grasp_point()), 255.0); grab()

# ---- metrics ----
z = np.array(log["z"]); roll = np.array(log["roll"]); pitch = np.array(log["pitch"])
box_final = d.xpos[box_bid].copy()
box_lift = float(box_final[2] - box0[2])
gp = grasp_point()
box_to_gp = float(np.linalg.norm(d.xpos[box_bid] - gp))
pad_gap = float(np.linalg.norm(d.xpos[env.lpad] - d.xpos[env.rpad]))
held = (box_to_gp < 0.06) and (box_lift > 0.05)
upright = z.min() > 0.55 and np.max(np.abs(roll)) < 0.4 and np.max(np.abs(pitch)) < 0.4

print("\n==== CHECK 2 RESULTS ====")
print(f"pelvis z: settled {z_settled:.3f}  min {z.min():.3f}  max {z.max():.3f}  end {z[-1]:.3f}")
print(f"pelvis |roll| max {np.max(np.abs(roll)):.3f} rad ({np.degrees(np.max(np.abs(roll))):.1f}deg)")
print(f"pelvis |pitch| max {np.max(np.abs(pitch)):.3f} rad ({np.degrees(np.max(np.abs(pitch))):.1f}deg)")
print(f"recoil (settled - min z): {(z_settled - z.min())*100:.1f} cm")
print(f"box lift: {box_lift*100:.1f} cm   box-to-graspPoint: {box_to_gp*100:.1f} cm   pad gap: {pad_gap*100:.1f} cm")
print(f"STAYED UPRIGHT: {upright}    BOX HELD: {held}")

imageio.mimsave("_unified_pick.mp4", frames, fps=30)
print(f"saved _unified_pick.mp4 ({len(frames)} frames)")
