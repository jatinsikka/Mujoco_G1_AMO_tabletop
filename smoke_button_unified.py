"""Smoke-test: ButtonPressEnv on the UNIFIED gripper-humanoid.
Runs ~300 steps (zero RL action = reach_bias + AMO), prints pelvis height + gripper->button
distance, confirms the robot stays upright and the CLOSED gripper reaches toward the button."""
import numpy as np
from env_wrapper_button import ButtonPressEnv

env = ButtonPressEnv(button_name="button_blue", unified=True, headless=True)  # x=0.45 -> right arm/gripper
obs, _ = env.reset(seed=0)
print("obs dim", obs.shape, "action dim", env.action_space.shape)

zs, dists = [], []
for i in range(300):
    obs, r, term, trunc, info = env.step(np.zeros(8, dtype=np.float32))
    pel = env.env.data.xpos[env.env.pelvis_id]
    hand = env._right_hand_pos()
    bpos = env.env.data.xpos[env.button_body_id]
    zs.append(float(pel[2])); dists.append(float(np.linalg.norm(hand - bpos)))
    if i % 60 == 0:
        print(f"  step {i}: pelvis_z={pel[2]:.3f} gripper={hand.round(3)} button={bpos.round(3)} "
              f"hand->button={dists[-1]:.3f} btn_disp={env._get_button_displacement():.4f}")
    if term:
        print(f"  FELL at step {i}"); break

zs = np.array(zs)
print(f"\nButton smoke: pelvis z min {zs.min():.3f} max {zs.max():.3f} end {zs[-1]:.3f}")
print(f"  hand->button min {min(dists):.3f} end {dists[-1]:.3f}")
print(f"  UPRIGHT: {zs.min() > 0.55}   (final btn displacement {env._get_button_displacement():.4f})")
