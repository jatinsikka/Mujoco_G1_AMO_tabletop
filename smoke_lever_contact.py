"""Smoke-test reset-in-contact for the UNIFIED lever env.
Checks: (1) reset seats the CLOSED gripper ON the lever grip (small grip->gripper dist, angle 0),
(2) a scripted forward push turns the lever (hinge angle rises meaningfully, >0.2 rad),
(3) AMO stays upright (pelvis z ~0.74)."""
import numpy as np, mujoco
from lever_press_env import LeverPressEnv

env = LeverPressEnv(unified=True, reset_in_contact=True, headless=True)  # lever x=0.6 -> right arm/gripper
obs, _ = env.reset(seed=0)
e = env.env

# ---- (1) contact at reset ----
hand = env._right_hand_pos()
grip = env._get_handle_pos()
pel = e.data.xpos[e.pelvis_id]
print("=== RESET-IN-CONTACT (lever) ===")
print(f"  gripper pad-mid  {hand.round(3)}")
print(f"  lever grip       {grip.round(3)}")
print(f"  grip->gripper dist {np.linalg.norm(hand-grip):.4f} m   (contact if < ~0.05)")
print(f"  pelvis z {pel[2]:.3f}   lever angle {env._get_lever_angle():.4f} rad")

# ---- (2) scripted forward push: each step, IK-servo the gripper toward a point just -Y (into
# the panel) of the CURRENT knob position — i.e. TRACK the knob and drive it through its +angle
# arc (a constant push slides off the spherical knob after a few degrees; RL would track it too).
def lever_angle():
    return float(e.data.qpos[env.lever_qadr])

def push_action():
    grip_now = e.data.geom_xpos[env.grip_geom_id].copy()
    into = grip_now + np.array([0.0, -0.06, 0.0])        # 6cm into the panel from the current knob
    a4 = e.right_arm_ik_step(into)                        # 4-DOF shoulder/elbow IK step
    base = e.default_dof_pos[19:23] + env.arm_reach_bias[4:]
    act = np.clip((a4 - base) / env.action_scale, -1.0, 1.0)
    full = np.zeros(8, dtype=np.float32); full[4:8] = act
    return full

zs, angs, dists = [], [], []
for i in range(200):
    obs, r, term, trunc, info = env.step(push_action())
    pel = e.data.xpos[e.pelvis_id]
    hand = env._right_hand_pos(); grip = env._get_handle_pos()
    zs.append(float(pel[2])); angs.append(lever_angle()); dists.append(float(np.linalg.norm(hand-grip)))
    if i % 40 == 0:
        print(f"  push {i}: pelvis_z={pel[2]:.3f} lever_angle={angs[-1]:.4f} grip->gripper={dists[-1]:.4f} turned={info.get('lever_turned')}")
    if term:
        print(f"  FELL at step {i}"); break

zs = np.array(zs); angs = np.array(angs)
print("\n=== SCRIPTED PUSH RESULT ===")
print(f"  max lever angle {angs.max():.4f} rad ({np.degrees(angs.max()):.1f} deg)  -> TURNED >0.2rad: {angs.max()>0.2}")
print(f"  pelvis z min {zs.min():.3f} max {zs.max():.3f} end {zs[-1]:.3f}  -> UPRIGHT: {zs.min()>0.55}")
print(f"  grip->gripper min {min(dists):.4f} end {dists[-1]:.4f}")
