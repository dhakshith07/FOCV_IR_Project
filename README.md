# Context-Aware Workspace Organizer

> **Intelligent Robotics Project** — Optimizing Object Placement Based on User Workflow

---

## Overview

A virtual intelligent robotic system that organizes books and study materials on a student's desk according to their current workflow and usage behavior. The system uses computer vision to observe the workspace, analyzes usage patterns, and deploys a virtual robot arm to reorganize objects for maximum efficiency — all running **100% locally on a Windows laptop**.

---

## Mandatory Models

| Model | Role |
|---|---|
| **YOLO11** | Real-time desk object detection |
| **ByteTrack** | Multi-object tracking with persistent IDs |
| **TrOCR** | Book title / text recognition |
| **SegFormer** | Semantic desk zone segmentation |

## Mandatory Algorithms

| Algorithm | Role |
|---|---|
| **Genetic Algorithm** | Optimal object placement evolution |
| **DQN** | Adaptive reorganization decision making |
| **RRT*** | Collision-free robot path planning |
| **Jacobian IK** | 6-DOF arm end-effector control |

---

## System Architecture

```
YOLO11 → ByteTrack → TrOCR + SegFormer
       → Workflow Analyzer → DQN → Genetic Algorithm
       → RRT* → Jacobian IK → Virtual Robot (PyBullet)
```

---

## Project Structure

```
IR-project/
├── app.py                     # Streamlit dashboard entry point
├── config.yaml                # Global configuration
├── requirements.txt
├── README.md
│
├── config/                    # Config loaders
├── data/                      # Database, sessions, logs
├── models/                    # Model weights
│
├── cv/                        # Computer Vision pipeline
│   ├── yolo/                  # YOLO11 detector
│   ├── tracking/              # ByteTrack
│   ├── trocr/                 # TrOCR OCR reader
│   └── segformer/             # SegFormer segmenter
│
├── workflow/                  # Usage analysis & priority engine
├── optimization/              # Genetic Algorithm + DQN
│
├── robotics/                  # Motion planning
│   ├── rrt/                   # RRT* path planner
│   ├── ik/                    # Jacobian IK solver
│   └── dqn/                   # DQN agent
│
├── simulation/                # 2D/3D virtual desk
├── database/                  # SQLite ORM layer
├── visualization/             # Charts & plots
├── utils/                     # Shared utilities
├── tests/                     # Unit & integration tests
└── docs/                      # Documentation
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Dashboard

```bash
streamlit run app.py
```

### 3. Open Browser

Navigate to: `http://localhost:8501`

---

## Development Phases

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Project structure + Dashboard | ✅ Complete |
| **Phase 2** | YOLO11 + ByteTrack integration | 🔲 Pending |
| **Phase 3** | TrOCR + SegFormer integration | 🔲 Pending |
| **Phase 4** | Workflow Analysis engine | 🔲 Pending |
| **Phase 5** | Genetic Algorithm + DQN | 🔲 Pending |
| **Phase 6** | RRT* + Jacobian IK | 🔲 Pending |
| **Phase 7** | PyBullet simulation | 🔲 Pending |
| **Phase 8** | Full integration | 🔲 Pending |

---

## Requirements

- Python 3.11
- Windows 10/11
- Webcam (optional — mock mode available)
- 8GB RAM minimum
- No internet required after model download

---

## License

For academic/educational use only.
