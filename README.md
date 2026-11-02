# pinto_model_zoo_ros2

ROS 2 Humble wrapper for [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo),
built for RoboCup@Home. It ships a GPU-ready Docker image with a person perception node
and a catalog of models useful for @Home tasks.

## Quick start

```bash
docker compose build
ROS_DOMAIN_ID=0 IMAGE_TOPIC=/camera/color/image_raw docker compose up
```

The image contains the weights, so it runs without internet access.

## `wholebody_detector` node

Runs [DEIMv2-Wholebody49](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/488_DEIMv2-Wholebody49):
body, head, face, hands, feet, 8 head directions, age, gender and body keypoints.

| Topic | Type | |
|---|---|---|
| `image` (sub) | `sensor_msgs/Image` | Remap to your camera |
| `detections` (pub) | `vision_msgs/Detection2DArray` | Pixel boxes; `class_id` holds the class name |
| `detections_image` (pub) | `sensor_msgs/Image` | Debug drawing, only when subscribed |

| Parameter | Default | |
|---|---|---|
| `model_variant` | `s` | `n`, `s` or `x` |
| `model_directory` | `/models` | |
| `object_score_threshold` | `0.50` | Body, head, face, hand, foot |
| `attribute_score_threshold` | `0.75` | Age, gender, head direction |
| `keypoint_score_threshold` | `0.50` | Joints |

Attribute classes share their box with the parent (`male` with `body`, for example);
match them by box. Overlapping `body` boxes for one person can occur because, like the upstream
demo, the node suppresses duplicates for keypoints only. Humans only: use another detector for
household objects.

## Model catalog

[`models/catalog.tsv`](models/catalog.tsv) lists models by task: re-identification, face
recognition and attributes, gaze, head pose, gestures (waving, pointing), sitting, clothing
parsing, pose, class-agnostic objects, depth, OCR and audio.

```bash
python3 scripts/fetch_catalog.py ~/pinto_models                     # default tier
python3 scripts/fetch_catalog.py ~/pinto_models --include-optional  # adds about 200 GB
```

Archives with a `sha256` in the catalog are verified; others are recorded on first download.
Only `wholebody_detector` has a ROS node so far. The rest are downloaded as published by
PINTO; read each model's upstream README for its input and output format.

## Licenses

**The default detector is for non-commercial use.** PINTO publishes DEIMv2-Wholebody49 as
Apache-2.0, but it derives from DEIMv2, whose license permits non-commercial purposes only, and
variants `s` and `x` also use DINOv3. See [NOTICE](NOTICE).

This repository's code is Apache-2.0. Weights are not stored here and keep their upstream
licenses, listed per model in the catalog. Entries marked `UNSPECIFIED` have no license file
upstream, so check before redistributing them.
