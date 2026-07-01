# bdx-new URDF Migration Notes

Compared files:

- Old: `source/BDX/assets/BDX/BDX.urdf`
- New: `source/BDX/assets/bdx-new/bdx-new.urdf`

## Summary

| Item | Old BDX | bdx-new |
| --- | ---: | ---: |
| Robot name | `URDF` | `bdx-new` |
| Links | 77 | 11 |
| Joints | 76 | 10 |
| Movable joints | 10 | 10 |
| Fixed joints | 66 | 0 |
| Total link mass | 17.749 kg | 14.047357 kg |
| Mesh references | `./meshes/*.stl` with `scale="0.001 0.001 0.001"` | `./meshes/*.STL` without scale |
| Zero-pose base-to-lowest-geometry height | 0.329722 m | 0.208431 m |

The new URDF is a simplified 10-DOF leg model. It removes the old model's fixed visual/detail tree for the head,
covers, holders, pads, IMU meshes, and other shell parts. The 10 movable joint names are preserved, so the existing
joint-name based action and reward regexes still match.

Note: both raw URDF files list movable joints as left-leg joints followed by right-leg joints, but Isaac Lab resolves
the runtime articulation and `joint_pos` action order as interleaved:
`Left_Hip_Yaw`, `Right_Hip_Yaw`, `Left_Hip_Roll`, `Right_Hip_Roll`, `Left_Hip_Pitch`, `Right_Hip_Pitch`,
`Left_Knee`, `Right_Knee`, `Left_Ankle`, `Right_Ankle`. This matches the shared RSL-RL symmetry helper, so
`bdx-new` does not need a separate mirror-loss implementation for joint ordering.

The mirror-loss transform also depends on joint-coordinate signs, not only ordering. FK checks against the mirrored
left/right leg chains show that all five joint coordinates need a sign flip after swapping left and right:
Hip_Yaw, Hip_Roll, Hip_Pitch, Knee, and Ankle.

## Preserved Movable Joint Names

- `Left_Hip_Yaw`
- `Left_Hip_Roll`
- `Left_Hip_Pitch`
- `Left_Knee`
- `Left_Ankle`
- `Right_Hip_Yaw`
- `Right_Hip_Roll`
- `Right_Hip_Pitch`
- `Right_Knee`
- `Right_Ankle`

## Link Naming Changes

The only common link name is `base_link`. Foot bodies changed from the old uppercase names to ankle pitch link names:

- Old feet: `Left_Foot`, `Right_Foot`
- New feet: `left_ankle_pitch_link`, `right_ankle_pitch`

Task configs for bdx-new therefore use body regex `.*ankle_pitch.*` where the old task used `.*_Foot`.

## Joint Tree Changes

The old URDF inserts fixed motor/support/holder links between the base and moving links. The new URDF attaches the
10-DOF kinematic chain directly:

- `base_link -> left_hip_yaw_link -> left_hip_roll_link -> left_hip_pitch_link -> left_knee_pitch_link -> left_ankle_pitch_link`
- `base_link -> right_hip_yaw_link -> right_hip_roll_link -> right_hip_pitch -> right_knee_pitch -> right_ankle_pitch`

## Joint Axis and Limit Differences

The hip yaw and hip pitch axes in the new URDF are slightly tilted instead of aligned exactly to a principal axis:

- `Left_Hip_Yaw`: old `0 0 1`, new `0 0.0279575622762108 0.999609110958665`
- `Right_Hip_Yaw`: old `0 0 1`, new `0 -0.0279575622762108 0.999609110958665`
- `Left_Hip_Pitch`: old `0 -1 0`, new `-0.000259155361818971 -0.99967537729627 0.0254769084870745`
- `Right_Hip_Pitch`: old `0 1 0`, new `-0.00025915536181831 0.99967537729627 0.0254769084870742`
- `Right_Knee`: old `0 -1 0`, new `0.0254782265387592 -0.99967537729627 0`

Position limits are nearly unchanged. Effort/velocity limits differ:

| Joint group | Old effort | New effort | Old velocity | New velocity |
| --- | ---: | ---: | ---: | ---: |
| Hip yaw | 60 | 35 | 20 | 20 |
| Hip roll | 60 | 60 | 20 | 10 |
| Hip pitch | 60 | 60 | 20 | 10 |
| Knee | 60 | 60 | 20 | 10 |
| Ankle | 17 | 14 | 43 | 30 |

`BDX_NEW_CFG` uses the new URDF limits for actuator simulation effort and velocity limits.

## Mass and Height Differences

The new base is much heavier (`base_link` is 4.733140 kg vs 1.0 kg in the old URDF), while many old shell/detail links
are removed. The zero-pose lowest geometry is also closer to the base frame:

- Old zero-pose lowest geometry: `z = -0.329722` relative to `base_link`
- New zero-pose lowest geometry: `z = -0.208431` relative to `base_link`

The bdx-new task uses `init_state.pos.z = 0.21` to spawn the zero pose just above contact. The base-height reward
uses `base_height_deviation.target_height = 0.18717`, preserving the old task's roughly 2.1 cm lower-than-zero-pose
stance target (`0.329722 - 0.30846`) on the new root frame (`0.208431 - 0.021262`).

## Mesh Path Changes

The imported URDF originally used ROS package paths such as `package://bdx/meshes/base_link.STL`.
For this Isaac Lab extension, those paths were rewritten to relative paths such as `./meshes/base_link.STL`.
