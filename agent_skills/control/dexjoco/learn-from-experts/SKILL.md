---
name: learn-from-experts
description: Use the recorded human demonstrations of this task in runtime/experts/ - teleoperated episodes on other layouts, with their {% if expert_video_only %}camera videos and keyframes{% else %}official action rows, states and camera videos{% endif %} - to learn approach poses, hand postures and timing. Not a replay script.
---

# Learn from expert episodes

`runtime/experts/` holds {{ expert_demonstrations }} human teleoperation episodes of
this task, recorded by the benchmark's authors on other layouts with a data glove and
trackers, in the same simulator and with the same controller you use. Each episode
directory has:

{% if expert_video_only %}
- `videos/*.mp4`: the recording's camera videos.
- `frames/`: keyframes of the base camera (`index.json` gives each frame's step).
- `episode.json`: the step count and how the episode was verified.

Only the camera views are given: no action rows or state vectors. Estimate approach
poses, hand postures and timing from what the videos show (the control rate is
{{ robot.hz }} steps per second).

{% else %}
- `keyposes.json`: **start here.** The recording summarized into its key moments
  (`start`, `approach`, `pre_contact`, `contact`, then `grasp`, `lift`, `release` when they
  happen, and `end`), per arm. Each moment gives the step and time, the flange pose
  (`wrist_position_m`, `wrist_rotvec_rad`: the {% if action_targets %}`dexjoco_move_ee`{% else %}action-row{% endif %} frame), the hand joints
  (`hand_joints_rad`, the {% if action_targets %}`dexjoco_hand`{% else %}action-row{% endif %} order) with a 0..1 `hand_closure`, the offset
  from the contact pose (`wrist_from_contact_m`), and for each task object its pose in that
  recording and the flange position relative to it (`wrist_minus_object_world_m`, and
  `wrist_minus_object_in_object_yaw_frame_m` in the object's own heading). The top level
  gives the approach direction and distance, the speed near contact, and `contact_cue`
  (what marked contact: the hand closing, an object starting to move).
- `trajectory.npz`: `action_rotvec` (rows in exactly the {% if action_targets %}official action layout: flange position and rotation vector, then hand joint targets, as `dexjoco_status` `action_layout` lists{% else %}`dexjoco_step` layout{% endif %}),
  `state` (the official state vector; values after its first
{% if robot.arms == 'dual' %}46{% else %}23{% endif %} entries describe the
  recording's initial layout and are not available to you during an episode) and
  `timestamp` (seconds, {{ robot.hz }} rows per second).
- `replay.zarr/`: the same data in the original recording format.
- `videos/*.mp4`: the recording's camera videos.
- `episode.json`: the step count and how the episode was verified.

{% endif %}Use them to learn what works: where the flange goes relative to an object before
contact, the hand posture and how far past contact the fingers close, how fast the
hand moves near contact, and the order of phases. {% if expert_video_only %}Measure these from the
videos and keyframes{% else %}Measure these from the actions and
states, look at the videos{% endif %}, and turn them into rules that adapt to what you see.

{% if not expert_video_only %}A quick way to use them: compare the object-relative offsets of the same moment across the
three recordings; what stays nearly the same is the rule (for example the pre-contact offset from
the object), what varies is up to you. Localize the object in your own scene, add the offset, and
use the moment's hand posture.

{% endif %}{% if action_targets %}Turn what you measure into `dexjoco_move_ee` waypoints (flange poses relative to the objects) and
`dexjoco_hand` postures. {% endif %}The layouts differ from yours: replaying an episode's rows will not reproduce its
outcome. Read the files with your own code{% if not expert_video_only %} (numpy){% endif %}; they are read-only.
