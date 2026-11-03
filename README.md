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

## `person_tracker` node

Tracks people with Wholebody49 + OSNet-AIN ReID and locks onto one target. The first
candidate that passes the filters is locked; after it is lost, the node re-acquires it by
ReID similarity until `lost_timeout_sec` expires. Depth must be `16UC1` (mm), aligned to color.

| Topic / service | Type | |
|---|---|---|
| `image` (sub) | `sensor_msgs/Image` | Color, `bgr8`-convertible |
| `depth` (sub) | `sensor_msgs/Image` | `16UC1`, aligned to color |
| `camera_info` (sub) | `sensor_msgs/CameraInfo` | Without it X and Y are 0 |
| `~/tracked_persons` (pub) | `vision_msgs/Detection2DArray` | `id` is the track id; position in the camera optical frame [m] |
| `~/target` (pub) | `vision_msgs/Detection2D` | Locked target, published only while it is visible |
| `~/tracking_image` (pub) | `sensor_msgs/Image` | Debug drawing, only when subscribed |
| `~/release_target` (srv) | `std_srvs/Trigger` | Drops the lock |

| Parameter | Default | |
|---|---|---|
| `model_directory` | `/models` | |
| `model_variant` | `s` | Detector variant |
| `reid_model_file` | `osnet_ain_x1_0_p_unified_aug_n.onnx` | |
| `object_score_threshold` | `0.50` | |
| `min_bbox_height_px` / `min_bbox_area_px` | `100` / `2000` | Candidate filter |
| `depth_min_mm` / `depth_max_mm` | `100` / `8000` | Candidate filter |
| `iou_match_threshold` | `0.3` | Frame-to-frame matching |
| `max_lost_frames` | `30` | |
| `reid_similarity_threshold` | `0.5` | |
| `lost_timeout_sec` | `7.0` | |
| `target_embedding_ema_alpha` | `0.1` | |
| `max_rate_hz` | `15.0` | Frames arriving faster are skipped |
| `sync_slop_sec` | `0.05` | Image/depth sync |

## `face_recognizer` node

Detects faces with Wholebody49 and scores each crop (LVFace-T) by cosine similarity against
a gallery directory. Faces are cropped from the detector box with no alignment. The gallery is
re-read when it changes (`Gallery: [...]` is logged).

| Topic | Type | |
|---|---|---|
| `image` (sub) | `sensor_msgs/Image` | |
| `~/faces` (pub) | `vision_msgs/Detection2DArray` | One `Detection2D` per face; each result is `class_id` = gallery name, `score` = cosine |
| `~/faces_image` (pub) | `sensor_msgs/Image` | Debug drawing |

Nothing is computed while no one subscribes to either output.

| Parameter | Default | |
|---|---|---|
| `model_directory` | `/models` | |
| `model_variant` | `n` | |
| `face_model_file` | `LVFace-T_Glint360K.onnx` | |
| `face_score_threshold` | `0.50` | |
| `gallery_directory` | `/faces` | See `face_gallery` in `face_recognition.py` for the layout |
| `rate_hz` | `5.0` | Must be positive |

## Running the tracker and face recognizer with compose

Prepare the gallery directory first (compose mounts it read-only at `/faces`):

```bash
mkdir -p ~/pylot_robocup/face_images     # or set FACE_GALLERY_DIR
docker compose build
docker compose up person_tracker face_recognizer
```

Both services use `profiles`, so they start only when named. Defaults are the D455 topics;
override with `COLOR_TOPIC`, `DEPTH_TOPIC`, `CAMERA_INFO_TOPIC`, `ROS_DOMAIN_ID`.

| Variable | Default |
|---|---|
| `COLOR_TOPIC` | `/d455/camera/color/image_raw` |
| `DEPTH_TOPIC` | `/d455/camera/aligned_depth_to_color/image_raw` |
| `CAMERA_INFO_TOPIC` | `/d455/camera/color/camera_info` |
| `FACE_GALLERY_DIR` | `${HOME}/pylot_robocup/face_images` |

Stop with `docker compose down`.

## Evaluating on a rosbag

```bash
ros2 run pinto_model_zoo_ros2 evaluate_bag <bag_directory> --output <dir> \
  [--start SEC] [--end SEC] [--image-topic T] [--depth-topic T] \
  [--model-variant s] [--model-directory /models]
```

`--start` and `--end` are seconds from the start of the bag. The output directory gets
`frames.csv` (per frame), `summary.json` (also printed) and `tracking.mp4`.
Measured on bag seconds 290 to 320 (RTX 3080 Laptop): 449 frames, detect 32.9 ms mean,
ReID 2.1 ms mean, total 35.1 ms mean (p95 38.0 ms). The 7 ReID re-acquisitions all picked
the same person.

`target_id_changes` counts how often the target's track id changed. It is not a count of
switches to a different person; check `tracking.mp4` for that.

## Checking on the robot

Do not use `ros2 bag play` against the live robot network. If a bag must be played live, a
human does it with `--topics` limited to the three camera topics and a separate
`ROS_DOMAIN_ID`.

## Known limits

- Thresholds (`reid_similarity_threshold`, `lost_timeout_sec`, ...) are provisional values
  without a basis and need tuning on the live robot.
- The existing bags contain few people, so accuracy has not been evaluated.
- Track ids are reassigned often.
- Faces are not aligned before encoding.

## Use from pylot_robocup2025

`use_pinto` only switches which following node pylot_robocup2025 runs. It does not start the
containers: start them in a separate terminal and run `docker compose down` when finished.

## Model catalog

[`models/catalog.tsv`](models/catalog.tsv) lists models by task: re-identification, face
recognition and attributes, gaze, head pose, gestures (waving, pointing), sitting, clothing
parsing, pose, class-agnostic objects, depth, OCR and audio.

```bash
python3 scripts/fetch_catalog.py ~/pinto_models                     # default tier
python3 scripts/fetch_catalog.py ~/pinto_models --include-optional  # adds about 200 GB
```

Archives with a `sha256` in the catalog are verified; others are recorded on first download.
Only `wholebody_detector`, `person_tracker` and `face_recognizer` have ROS nodes so far. The rest are downloaded as published by
PINTO; read each model's upstream README for its input and output format.

## Licenses

**The default detector is for non-commercial use.** PINTO publishes DEIMv2-Wholebody49 as
Apache-2.0, but it derives from DEIMv2, whose license permits non-commercial purposes only, and
variants `s` and `x` also use DINOv3. See [NOTICE](NOTICE).

Bundled in the built image (see [NOTICE](NOTICE)): the OSNet-AIN ReID model is Apache-2.0.
LVFace's code is MIT, but its upstream README says the models are for non-commercial research
only and contradicts itself, so **treat the LVFace weights as non-commercial research use**.
The Glint360K training data terms are unconfirmed.

This repository's code is Apache-2.0. Weights are not stored here and keep their upstream
licenses, listed per model in the catalog. Entries marked `UNSPECIFIED` have no license file
upstream, so check before redistributing them.
