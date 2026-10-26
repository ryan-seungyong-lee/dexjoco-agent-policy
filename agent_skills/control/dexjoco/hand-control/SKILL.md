---
name: hand-control
description: Shape, close and open the 16-DoF Allegro hand{% if robot.arms == 'dual' %}s{% endif %} {% if action_targets %}with dexjoco_hand{% else %}with official action rows{% endif %} - joint roles, grasp postures, squeezing past contact, holding and releasing - and verify grasps from images. Use before any grasp, press or release.
---

# Hand control: Allegro

Each hand has four fingers of four joints (`ff` index, `mf` middle, `rf` ring, `th`
thumb). For the index, middle and ring fingers, joint 0 spreads the finger sideways
(-0.47..0.47 rad) and joints 1-3 flex it from the knuckle to the tip (about
-0.2..1.6 rad; positive curls toward the palm). The thumb's joint 0
(0.263..1.396) turns it from beside the palm toward opposing the fingers; joints 1-3
(about -0.1..1.7) rotate and flex it toward the fingertips. Before relying on a
posture, move one joint at a time on the empty hand and watch the wrist image to see
what it does. See
[robot-interface](../robot-interface/SKILL.md) for {% if action_targets %}the joint order{% else %}where these sit in a row{% endif %}{% if robot.arms == 'dual' %}, and remember the left hand's mirrored order{% endif %}.

## Postures

- **Open:** flexion 0, spread 0, thumb joint 0 at 0.263. Approach with the hand open
  and the thumb out of the way.
- **Power grasp:** wrap the three fingers (joints 1-3 together) and bring the thumb
  across (joint 0 up, joints 2-3 flexed) so the object sits against the palm.
- **Precision grasp:** oppose the thumb tip and one or two fingertips (index, or
  index and middle) with the other fingers curled away; good for thin or small parts
  and for pressing with one fingertip.
- **Pointing/pressing:** extend one finger (flexion near 0) and curl the others;
  move the flange so that fingertip makes the contact.

## Closing and squeezing

Joints are weak position servos: a finger stopped by an object pushes with a force
that grows with the gap between its target and its measured angle. To hold an
object, command flexion clearly beyond the contact angle (for example 0.2-0.4 rad
{% if action_targets %}
past where the fingers stopped, as `blocked_joints` and the tracking errors show).
Commanded targets persist, so later moves keep squeezing; sending the measured joints
as new targets releases the squeeze.

- Close over several steps (`dexjoco_hand` `steps` 10-30), then hold a few steps
  before lifting.
{% else %}
past where the fingers stopped), and keep that target on every following row while
the object is held. Copying the measured hand joints into later rows releases the
squeeze.

- Close over several steps (10-30 rows), then hold a few steps before lifting.
{% endif %}
- Lift slowly at first and look: an object that slides in the fingers needs a
  different posture or more squeeze, not a faster lift.
- Keep the flange target moving smoothly while a grasped object is carried; large
  flange jumps whip the hand.

## Releasing and contact

- Release by opening the fingers over a few {% if action_targets %}steps{% else %}rows{% endif %} with the flange held still, then
  retreat upward.
- Verify every grasp and release from images (wrist and overview), not from the
  commanded joints: the measured joints show where the fingers stopped, not whether
  the object is held.
- Joints that do not reach their target while the hand is empty usually mean the
  fingers are touching something; observe before pushing further.
