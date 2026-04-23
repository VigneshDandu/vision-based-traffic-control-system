# Flow Redirect

Flow Redirect is a small Python project that analyzes traffic congestion and decides which road should get signal priority.

This repository is intentionally kept minimal and centered around these files:

- `main.py`
- `congestion_calculator.py`
- `requirements.txt`
- `README.md`
- `.gitignore`

## Project Overview

`main.py` handles the traffic monitoring flow. It:

- reads four road video feeds
- uses YOLOv8 to detect vehicles
- uses OpenCV and NumPy for frame processing
- estimates lane count
- detects whether traffic is stopped
- collects vehicle counts during each waiting cycle
- sends the processed road data to the congestion scoring logic

`congestion_calculator.py` contains the decision logic. It:

- converts vehicle counts into a weighted score
- divides traffic load by lane count
- applies a minimum baseline score
- selects the highest-priority road

## Congestion Logic

The current vehicle weights are:

- Car = `1`
- Bike = `0.5`
- Bus = `3`
- Truck = `3`

The congestion score is based on:

```text
weighted vehicle score / number of lanes
```

Each road gets a minimum baseline score of `5` so signal timing does not become too small.

## Dependencies

Install the exact packages required by the two project files:

```powershell
pip install -r requirements.txt
```

The project uses:

- `ultralytics`
- `opencv-python`
- `numpy`

`time` is also imported in `main.py`, but it is part of Python's standard library and does not need to be installed separately.

## Runtime Inputs

The repository keeps only the core source and config files, but `main.py` still expects local runtime assets such as:

- a YOLO model file such as `yolov8n.pt`
- road video files:
  - `roadA.mp4`
  - `roadB.mp4`
  - `roadC.mp4`
  - `roadD.mp4`

If these files are missing, `main.py` will not run correctly.

## Run

```powershell
python main.py
```

## Execution Flow

1. `main.py` reads frames from each road feed.
2. It checks lane structure and traffic movement.
3. It counts vehicles when traffic is in a waiting state.
4. It prepares road-wise counts and lane data.
5. `congestion_calculator.py` calculates congestion scores.
6. The road with the highest score gets priority for the green signal.

## Notes

- This repository is meant to keep only the core code and setup files under version control.
- Virtual environments, model files, videos, and other large local assets are intentionally ignored in Git.
- The traffic scoring logic is intentionally simple and can be extended later if needed.
