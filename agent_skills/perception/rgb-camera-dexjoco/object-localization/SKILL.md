---
name: object-localization
description: Locate DexJoCo objects from RGB images using camera intrinsics and world camera poses, without depth. Use to triangulate object positions from two views or from one view plus a known height before reaching, grasping or pressing.
---

# Object localization: RGB + camera calibration (no depth)

Every observation carries RGB images plus `camera_intrinsics` (3x3 matrix per camera)
and `camera_poses` (camera pose in the world frame, quaternion `[w, x, y, z]`, OpenGL
convention: the camera looks along its -Z axis with +Y up; image row 0 is at the top).
There is **no depth**. A pixel therefore gives a world-frame **ray**, not a point. Do
not seek simulator files or hidden camera or object state.

The state's flange poses are the arm's tool plate, not fingertip, palm, camera or
object poses; the fingertips are about 0.18 m beyond the flange at the start posture
and move with the finger joints.

## From pixel to ray

For pixel (u, v) with intrinsics K and pose (R from the quaternion, t):
`d_cam = [(u - cx)/fx, -(v - cy)/fy, -1]`, `d_world = R @ d_cam` (normalize),
origin `t`. Use `dexjoco_rotation_math` for quaternion conversions if needed.

## From rays to points

- **Two views:** take the same object feature in two cameras (e.g. the overview and a
  wrist camera) and intersect the two rays (closest point of two lines). A large miss
  distance means the pixels were not the same feature.
- **One view plus a plane:** objects resting on the table lie at the table height plus
  half their thickness; intersect the ray with that horizontal plane. Estimate the
  table height once, e.g. by touching it gently with a fingertip and reading the flange
  height, or by triangulating a table feature from two views.
- Verify a computed point by projecting the hand's measured flange position into the
  image (inverse of the above) and checking it lands on the hand.

## Align before contact

- The wrist camera moves with the hand; after moving it, recompute rays from the new
  pose. Confirm alignment in a second view before contact.
{% if not formal_episode %}
Record in memory what you aligned against, the views and motions used, and the
outcome, not coordinates.
{% endif %}
