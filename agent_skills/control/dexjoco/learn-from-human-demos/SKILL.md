---
name: learn-from-human-demos
description: Use the recordings of people doing a similar task with their bare hands in runtime/experts/ - video and keyframes, plus tracked 3-D hand motion when present - to learn the order of phases, approach directions, hand postures and timing, then transfer them to the Allegro hand. Not a replay script.
---

# Learn from human demonstrations

`runtime/experts/` holds {{ expert_demonstrations }} recordings of a **person** doing a
task similar to this one with bare hands, filmed from a head-mounted camera. They are
not robot episodes: the scene, the objects, the camera and the body differ from yours,
and the person may do more or less than your task asks (for example take the object
out of its case as well as put it back). Find the part that matches your task.

Each recording directory has:

- `video.mp4`: the head-camera video (30 fps).
- `frames/`: keyframes (`index.json` gives each frame's time in seconds and a short
  label of what the hands are doing when known).
- `episode.json`: the source dataset, the person's instruction for the clip, which
  hands act, the time span that matches this task, and the coordinate conventions.
{% if expert_human_motion %}
- `keyposes.json`: **start here.** The recording summarized into its key moments per
  acting hand (start, pre-shape, approach, pre-contact, contact, grasp, release, end):
  the wrist pose and its offset from the pose at contact, and the hand posture
  (Allegro retarget, in the step hand order) with its mean flexion. Wrist positions are in
  the clip's frame, not your world: use the offsets and directions, not the positions.
  It has no object poses.
- `hand_trajectory.npz` (10 Hz, hand slot 0 = left, 1 = right):
  - `t` (N,): seconds from the clip start.
  - `present` (N, 2): the hand is tracked.
  - `wrist_pos` (N, 2, 3) metres and `wrist_rotvec` (N, 2, 3) radians, in the frame
    `episode.json` names (`frame`); for `gravity_head0` the axes are x right, y forward,
    z up, anchored at the head at the first frame. Only relative motion is meaningful.
  - `keypoints` (N, 2, 21, 3): the 21 hand joints (wrist, then thumb, index, middle,
    ring, little: 4 joints each, base to tip) in the same frame.
  - `finger_flexion` (N, 2, 5): total curl per finger (thumb..little), radians;
    0 = straight.
  - `aperture` (N, 2): thumb-tip to index-tip distance, metres.
  - `contact` (N, 2): estimated probability the hand touches something, 0..1
    (`force` (N, 2) when present: estimated total contact force, arbitrary units).
- `phases.json`: per hand, time segments labelled `idle`, `reach`, `grasp`,
  `manipulate`, `release`, `retract`, derived from contact and wrist speed.
- `allegro_retarget.npz` (when present): `q` (N, 2, 16) Allegro joint angles that
  reproduce the human finger shape per frame (fingertip-position retargeting), already
  in the `dexjoco_step` hand order of each hand; `valid` (N, 2). A hand-shape hint
  only: it ignores the object, the arm and contact forces. It is approximate: fingertips
  land about 2-3 cm from the human shape, and thumb-to-finger opening follows the human's
  only loosely, so treat it as open versus closed, not as a precise grasp.
{% endif %}

## How to use them

- Watch the matching span first. Write down the phases in order, which hand does
  what, where the hand approaches the object from (above, side, front), and how the
  fingers are arranged at contact.
{% if expert_human_motion %}
- Use `phases.json` and `contact` to find contact times; compare `aperture` and
  `finger_flexion` just before and after contact to learn how much the hand opens
  before the approach and how far it closes.
- Read approach directions and distances as **relative** wrist motion around contact
  (for example the last 10 cm before the grasp, the lift after it), then re-express
  them relative to the object you localize in your own scene. Human absolute positions
  do not map to the robot's world frame; the robot flange is not a human wrist.
- Seed hand postures from `allegro_retarget.npz` at the contact frames, then check
  them in your own episode: the Allegro hand is larger and has four fingers.
{% endif %}
- Human timing is a guide to ordering, not speed: move the robot as your own
  observations allow.
- Confirm every borrowed rule in your own episodes and keep only what helps.

Read the files with your own code{% if expert_human_motion %} (numpy){% endif %}; they are read-only.
