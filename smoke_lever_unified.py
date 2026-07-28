"""Smoke-test: LeverPressEnv on the UNIFIED gripper-humanoid.
Runs ~300 steps (zero RL action = reach_bias + AMO), prints pelvis height + gripper->lever
distance, and renders a frame to inspect the arm pose (contortion check)."""
import numpy as np, mujoco, imageio
from lever_press_env import LeverPressEnv

env = LeverPressEnv(unified=True, headless=True)   # lever at x=0.6 -> right arm/gripper
obs, _ = env.reset(seed=0)
print("obs dim", obs.shape)

zs, dists = [], []
for i in range(300):
    obs, r, term, trunc, info = env.step(np.zeros(8, dtype=np.float32))
    pel = env.env.data.xpos[env.env.pelvis_id]
    hand = env._right_hand_pos()
    grip = env._get_handle_pos()
    zs.append(float(pel[2])); dists.append(float(np.linalg.norm(hand - grip)))
    if i % 60 == 0:
        print(f"  step {i}: pelvis_z={pel[2]:.3f} gripper={hand.round(3)} lever_grip={grip.round(3)} "
              f"hand->lever={dists[-1]:.3f} angle={env._get_lever_angle():.3f}")
    if term:
        print(f"  FELL at step {i}"); break

zs = np.array(zs)
print(f"\nLever smoke: pelvis z min {zs.min():.3f} max {zs.max():.3f} end {zs[-1]:.3f}")
print(f"  hand->lever min {min(dists):.3f} end {dists[-1]:.3f}")
print(f"  UPRIGHT: {zs.min() > 0.55}   (final lever angle {env._get_lever_angle():.3f})")

# render arm-pose frame: view from the robot's RIGHT-FRONT (outside the panel) so the
# right arm + gripper + lever are all visible and torso-crossing is obvious if present.
r = mujoco.Renderer(env.env.model, 480, 640)
pel = env.env.data.xpos[env.env.pelvis_id]
cam = mujoco.MjvCamera()
cam.lookat[:] = np.array([0.5, -1.6, 0.85])
cam.distance = 1.5; cam.azimuth = 70; cam.elevation = -12
r.update_scene(env.env.data, camera=cam)
imageio.imwrite("_unified_lever_arm.png", r.render())
print("saved _unified_lever_arm.png")
