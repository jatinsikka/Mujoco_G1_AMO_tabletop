"""Smoke-test reset-in-contact for the UNIFIED button env (blue button).
Checks: (1) reset places the CLOSED gripper ON the button cap (small button->gripper gap),
(2) a scripted push depresses the button (displacement rises toward the 2cm threshold),
(3) AMO stays upright (pelvis z ~0.74)."""
import numpy as np
from env_wrapper_button import ButtonPressEnv

env = ButtonPressEnv(button_name="button_blue", unified=True, reset_in_contact=True, headless=True)
obs, _ = env.reset(seed=0)

# ---- (1) contact at reset ----
hand = env._right_hand_pos()
cap_gid = env.env.model.geom(f"{env.button_body_name}_top").id
cap = env.env.data.geom_xpos[cap_gid]
pel = env.env.data.xpos[env.env.pelvis_id]
print("=== RESET-IN-CONTACT ===")
print(f"  gripper pad-mid   {hand.round(3)}")
print(f"  button cap (top)  {cap.round(3)}")
print(f"  button->gripper dist {np.linalg.norm(hand-cap):.4f} m   (contact if < ~0.03)")
print(f"  pelvis z {pel[2]:.3f}   button_disp {env._get_button_displacement():.4f}")

# ---- (2) scripted push forward: hold a CONSTANT action in the -Y (into-cap) joint direction
# derived once from the pad-midpoint Jacobian at the seated pose. A steady push (not a
# re-solved IK that oscillates) — the cleanest test that a forward push depresses + HOLDS the
# button from the contact start.
import mujoco
e = env.env
jl = np.zeros((3, e.model.nv)); jr = np.zeros((3, e.model.nv))
mujoco.mj_jacBody(e.model, e.data, jl, None, e.lpad); mujoco.mj_jacBody(e.model, e.data, jr, None, e.rpad)
J = 0.5 * (jl + jr)[:, e.rarm_dadr]
dq_push = np.linalg.pinv(J) @ np.array([0.0, -0.05, 0.0])   # 5cm -Y (into cap) in joint space
push = np.zeros(8, dtype=np.float32)
push[4:8] = np.clip(dq_push / env.action_scale, -1.0, 1.0)  # hold this constant into-cap push

zs, disps, dists = [], [], []
for i in range(200):
    obs, r, term, trunc, info = env.step(push)
    pel = env.env.data.xpos[env.env.pelvis_id]
    hand = env._right_hand_pos(); cap = env.env.data.geom_xpos[cap_gid]
    zs.append(float(pel[2])); disps.append(env._get_button_displacement()); dists.append(float(np.linalg.norm(hand-cap)))
    if i % 40 == 0:
        print(f"  push {i}: pelvis_z={pel[2]:.3f} btn_disp={disps[-1]:.4f} hand->cap={dists[-1]:.4f} pressed={info.get('button_pressed')}")
    if term:
        print(f"  FELL at step {i}"); break

zs = np.array(zs); disps = np.array(disps)
print("\n=== SCRIPTED PUSH RESULT ===")
print(f"  max button_displacement {disps.max():.4f} m  (threshold 0.02)  -> PRESSED THRESHOLD: {disps.max()>0.02}")
print(f"  pelvis z min {zs.min():.3f} max {zs.max():.3f} end {zs[-1]:.3f}  -> UPRIGHT: {zs.min()>0.55}")
print(f"  hand->cap min {min(dists):.4f} end {dists[-1]:.4f}")
