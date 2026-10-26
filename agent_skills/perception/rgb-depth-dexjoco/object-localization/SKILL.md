---
name: object-localization
description: Locate DexJoCo objects from RGB images, depth images and measured flange poses, without camera intrinsics or camera poses. Use to judge where objects are relative to the hand and how far they are from each camera before reaching, grasping or pressing.
---

# Object localization: RGB + depth (no calibration)

Every observation carries RGB images and a depth image per camera (`<camera>_depth_m`
stats plus a depth PNG attachment): linear depth in metres along each camera's optical
axis. There are **no** camera intrinsics or camera poses, so a pixel and its depth do
not directly give a world point. Do not seek simulator files or hidden camera or object
state.

The state's flange poses are the arm's tool plate, not fingertip, palm, camera or
object poses; the fingertips are about 0.18 m beyond the flange at the start posture
and move with the finger joints.

## Use depth for what it gives

- Depth separates objects from the table and from each other: a depth step at an
  object's outline marks its silhouette even where colours are similar.
- Relative depth along one camera is reliable: whether the hand is in front of or
  behind the target, and by how many centimetres, along that camera's viewing axis.
- The wrist camera's depth near the hand is the most useful: when the hand is above or
  beside the target, its depth tells the remaining approach distance along that view.

## Calibrate a camera yourself

- Move the empty hand by small, known flange displacements at a safe height and note
  where a visible hand feature moves in the image and how its depth changes. A few such
  moves fit the camera's image-to-world mapping near that region (scale per pixel at that
  depth, and the direction of each image axis in the world).
- Combine that fitted mapping with the target's pixel and depth to estimate where the
  target is relative to the hand. Recheck after large motions or a different camera.

## Align before contact

- Close in with small flange moves, comparing hand and target pixels and depths in fresh
  observations. Confirm alignment in a second view before contact: one image's pixel
  overlap does not prove alignment (parallax).
{% if not formal_episode %}
Record in memory what you aligned against, the views and motions used, and the
outcome, not coordinates.
{% endif %}
