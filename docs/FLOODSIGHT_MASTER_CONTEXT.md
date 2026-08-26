# FloodSight — Complete Project Context

I am building **FloodSight** for a hackathon.

Treat the following project scope, architecture, ML strategy, demo plan, and product direction as the current agreed design unless I explicitly change something.

---

# 1. Core Idea

## FloodSight — Drone-Based Disaster Mapping & Decision-Intelligence System

FloodSight is an AI-powered platform that analyses **live or recorded drone footage during floods and other disasters** and converts it into actionable emergency-response information.

It should NOT be presented as merely:

- flood detection
- object detection
- flood prediction
- image segmentation
- a drone video viewer

The real value proposition is:

> **FloodSight converts raw disaster drone footage into a live priority-aware rescue and resource-allocation map.**

The platform should identify:

- flooded areas
- flooded roads
- clear roads
- blocked roads
- damaged buildings
- destroyed buildings
- people
- vehicles
- isolated regions
- potentially inaccessible rescue locations

Then it should automatically determine:

- rescue zones
- priority ranking
- why each zone is high priority
- road accessibility
- potential rescue routes
- incident summaries

The main user is an emergency-response command centre / disaster-management team.

---

# 2. Hackathon Demo Goal

The final demo should feel like a **real emergency command-centre product**, not an ML notebook.

Planned demo:

A flood/drone video plays.

FloodSight analyses it live.

Input can be:

1. Video file
2. Webcam feed
3. Eventually a real drone stream

For the hackathon, I may point my webcam at another screen playing flood drone footage to demonstrate a "live camera feed".

However, FloodSight should also support directly loading a video file because that avoids:

- monitor moiré
- reflections
- camera distortion
- quality reduction

Both modes should use the SAME inference pipeline.

---

# 3. Desired Demo Experience

The judge should see:

```text
Drone / Flood Video
        ↓
FloodSight AI
        ↓
Flood segmentation
Road analysis
Building analysis
People detection
Vehicle detection
        ↓
Zone generation
        ↓
Priority calculation
        ↓
Rescue Priority List
        ↓
Map / Routes / Incident Intelligence

```

Example dashboard information:

```text
FLOODSIGHT                      ● LIVE

Incident Status:
🔴 SEVERE

Flooded Area:        43%
People Detected:      12
Blocked Roads:         3
Damaged Buildings:     8
Vehicles:              6

RESCUE PRIORITIES

🔴 #1 Zone A       94/100
6 people
severe flooding
road access blocked

🟠 #2 Zone C       76/100
3 people
damaged structures
alternate route available

🟡 #3 Zone B       52/100
localized flooding
road partially accessible

```

---

# 4. Machine-Learning Strategy

Do NOT build one huge monolithic model.

Use a hybrid system.

## Model A — Disaster Semantic Segmentation

Use:

- FloodNet
- RescueNet

Fine-tune a pretrained semantic-segmentation architecture such as:

**SegFormer**, preferably starting around SegFormer-B2 unless experiments suggest another variant.

Purpose:

Identify pixels belonging to things such as:

- water
- flooded road
- clear road
- blocked road
- buildings
- building damage
- vehicles where available
- debris eventually

---

# 5. Model B — Person / Vehicle Detection

Use:

**VisDrone**

Fine-tune a pretrained modern YOLO detector.

Purpose:

Detect aerial-view:

- person
- car
- van
- truck
- bus
- bicycle / motorcycle if useful

The important FloodSight outputs initially are:

- people
- vehicles

VisDrone is being used because ordinary human-detection datasets are mostly ground-level whereas VisDrone contains aerial/drone perspectives.

---

# 6. Why Separate Models

FloodNet and RescueNet are primarily useful for **semantic segmentation**.

VisDrone is primarily useful for **bounding-box object detection**.

Therefore DO NOT blindly merge all three datasets into a single annotation format/model.

Preferred architecture:

```text
                    VIDEO FRAME
                         │
             ┌───────────┴───────────┐
             ▼                       ▼

       Segmentation Model       Detection Model
      FloodNet + RescueNet         VisDrone
             │                       │
             ▼                       ▼
 Flood/water/roads/buildings     People/vehicles

             └───────────┬───────────┘
                         ▼

                Scene Understanding
                         ▼
                  Zone Generation
                         ▼
                 Priority Engine
                         ▼
              FloodSight Dashboard

```

---

# 7. Public Datasets

## FloodNet

Official project/repository:

[https://github.com/BinaLab/FloodNet-Supervised\_v1.0](https://github.com/BinaLab/FloodNet-Supervised_v1.0)

Approximately **2,343 high-resolution UAV images** captured after Hurricane Harvey.

Useful semantic classes include things such as:

- flooded building
- non-flooded building
- flooded road
- non-flooded road
- water
- vehicle
- tree
- other/background

FloodNet is highly relevant because it is actual post-flood UAV imagery.

---

## RescueNet

Official repository:

[https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation](https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation)

Approximately **4,494 high-resolution UAV disaster images**.

Extremely useful classes include:

- Road-Clear
- Road-Blocked
- different building-damage levels
- water
- vehicles
- disaster infrastructure classes

It complements FloodNet because FloodNet has flooded/non-flooded roads, while RescueNet explicitly has clear/blocked roads and damage severity.

Important:

Check RescueNet licensing carefully before any eventual commercial deployment. For the hackathon/research demo, it can be used according to the dataset's license, but licensing must not be ignored later.

---

## VisDrone

Official repository:

[https://github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)

Use mainly:

**VisDrone-DET**

Do NOT initially download/train on all VisDrone video sequences unless needed.

VisDrone contains aerial annotations for:

- pedestrians
- people
- cars
- vans
- trucks
- buses
- bicycles
- motorcycles
- etc.

Use it for aerial person/vehicle detection.

---

# 8. Dataset Folder Concept

Something like:

```text
FloodSight-Datasets/
│
├── FloodNet/
│   ├── Train/
│   ├── Validation/
│   └── Test/
│
├── RescueNet/
│   ├── train/
│   ├── val/
│   └── test/
│
└── VisDrone/
    ├── VisDrone2019-DET-train/
    └── VisDrone2019-DET-val/

```

---

# 9. Unified FloodSight Segmentation Taxonomy

FloodNet and RescueNet labels need to be mapped carefully into a consistent FloodSight taxonomy.

Tentative classes:

```text
0  Background / Other
1  Water / Flood
2  Road Clear
3  Road Flooded
4  Road Blocked
5  Building Normal
6  Building Flooded
7  Building Minor Damage
8  Building Major Damage
9  Building Destroyed
10 Vehicle
11 Tree / Vegetation
12 Debris / Landslide

```

DO NOT incorrectly assume:

```text
Flooded Road == Blocked Road

```

They are conceptually different.

A road may be:

- flooded but traversable
- flooded and blocked
- physically blocked by debris
- completely clear

Dataset harmonisation is one of the most important ML engineering problems in this project.

If a source dataset does not contain information to distinguish a class, do not fabricate labels.

---

# 10. Training Philosophy

Use **transfer learning**, not training from random initialization.

Segmentation:

```text
Pretrained SegFormer
        ↓
Fine-tune
        ↓
FloodNet + RescueNet

```

Detection:

```text
Pretrained YOLO
        ↓
Fine-tune
        ↓
VisDrone

```

Potential later enhancement:

Fine-tune VisDrone detection model using disaster-specific human annotations if we collect or find suitable data.

---

# 11. Landslides and Other Disasters

The long-term project includes:

- floods
- landslides
- potentially earthquakes / cyclones / other disasters

BUT for the hackathon, FLOODS should work extremely well first.

Do not jeopardise the flood MVP trying to solve every disaster.

Landslides can later add classes such as:

- landslide
- debris
- exposed soil
- rockfall
- road collapse
- blocked road
- damaged bridge

---

# 12. Real-Time Detection

After training, both models should support continuous inference.

Pipeline:

```text
camera/video.read()
        ↓
current frame
        ↓
SegFormer
        +
YOLO
        ↓
combine predictions
        ↓
zone / priority analysis
        ↓
draw visual overlays
        ↓
send results to frontend

```

The weights are loaded ONCE.

Training happens offline.

Inference runs repeatedly on video frames.

Exact FPS is not a strict requirement as long as the demo visually feels real-time.

Something in the range of approximately **10+ FPS** can already look convincing.

Optimisation options later:

- lower resolution
- frame skipping
- half precision
- TensorRT / ONNX
- smaller YOLO variant
- smaller SegFormer
- asynchronous model execution

Do not sacrifice accuracy immediately just to chase 30 FPS.

---

# 13. Zone Generation

FloodSight should automatically divide the disaster into **rescue zones**.

For the hackathon MVP, use image-space zoning.

Internally divide the frame into a **4×4 grid**:

```text
A1 A2 A3 A4
B1 B2 B3 B4
C1 C2 C3 C4
D1 D2 D3 D4

```

DO NOT necessarily display this raw grid to users.

For every grid cell calculate information such as:

- percentage flooded
- number of people
- number of vehicles
- damaged-building pixels / objects
- blocked-road pixels
- clear-road pixels
- accessibility
- isolation

Example:

```text
B2

People: 4
Flood coverage: 72%
Road blocked: yes
Building damage: severe

```

Neighbouring dangerous cells should be merged.

Example:

```text
B2 + B3
   ↓
Zone 1

```

The UI can display a smooth polygon/bounding region around the merged zone.

Users see:

```text
ZONE 1
🔴 CRITICAL
Priority: 93/100

```

rather than raw B2/B3 grid names.

---

# 14. Temporal Smoothing

Do not let statistics jump wildly every frame.

Keep a rolling observation window, approximately:

**1–2 seconds**

Example:

```text
Frame 1 → 6 people
Frame 2 → 7
Frame 3 → 6
Frame 4 → 7
Frame 5 → 7

FloodSight output → ~7 people

```

Likewise, zones should persist across adjacent frames.

Track zones based on:

- region overlap / IoU
- proximity
- detection persistence

Zone IDs should remain stable:

```text
Zone 1
Zone 2
Zone 3

```

instead of continuously appearing/disappearing.

---

# 15. Production-Level Geographic Zoning

The hackathon uses image-space zones.

The real-world architecture should use drone telemetry:

- GPS
- altitude
- heading
- camera orientation
- timestamp

Then detections can be projected into geographical coordinates.

Instead of:

```text
x = 742
y = 391

```

FloodSight eventually knows:

```text
latitude
longitude

```

Geographic areas can then be divided into persistent spatial cells/tiles.

This means Zone 04 remains Zone 04 even if the drone flies away and returns.

This should be mentioned to judges if they ask how moving drones are handled.

---

# 16. Priority Engine

Do NOT initially train an opaque neural network to directly predict:

```text
Priority = 93

```

For the hackathon, use an **explainable deterministic priority engine**.

Inputs may include:

- people/human risk
- isolation
- road accessibility
- building damage
- flood severity
- vehicle presence
- vulnerability
- confidence
- potentially distance to rescue resources later

Illustrative weighting:

```text
Human risk             ~35%
Isolation/access       ~25%
Structural damage      ~20%
Flood severity         ~15%
Other factors           ~5%

```

Exact weights may be tuned.

Normalize score to:

```text
0–100

```

Suggested categories:

```text
80–100  🔴 CRITICAL
60–79   🟠 HIGH
40–59   🟡 MODERATE
0–39    🟢 LOW

```

The score MUST be explainable.

Example:

```text
ZONE A3

Priority Score: 94/100
🔴 CRITICAL

Why?

+35  8 people detected
+25  No accessible road
+18  Severe structural damage
+16  High flood coverage

```

This explainability is important because emergency responders should understand WHY the system recommends a location.

---

# 17. Rescue Priority List

This is one of FloodSight's most important product features.

Example:

```text
RESCUE PRIORITIES

🔴 #1 ZONE A3              94/100

8 people detected
Severe flooding
All nearby roads blocked
3 severely damaged buildings

Reason:
People appear isolated with
no accessible ground route.

[VIEW ZONE]
[FIND ROUTE]


🔴 #2 ZONE C1              87/100

4 people
Heavy structural damage
High flood exposure


🟠 #3 ZONE B7              71/100

3 people
Main road blocked
Alternate road available


🟡 #4 ZONE D2              46/100

Localized flooding
Road partially accessible

```

The ranked priority list should automatically reorder as conditions change.

---

# 18. Tactical Disaster Map

FloodSight should contain a map/tactical-view section.

Layers might include:

```text
☑ Flooded Areas
☑ Blocked Roads
☑ People
☑ Damaged Buildings
☑ Rescue Zones
☑ Accessible Roads

```

Suggested visual semantics:

```text
Blue       flood
Red        critical zone
Orange     high priority
Yellow     moderate
Green      safe / accessible
Dark/black blocked road

```

For the hackathon, the map can initially be a **relative tactical map** derived from the video rather than real GIS coordinates.

Production architecture should later support actual GIS using drone GPS.

---

# 19. Rescue Routing

FloodSight should eventually recommend accessible rescue routes.

AI detects:

```text
Road A → Clear
Road B → Flooded
Road C → Blocked
Road D → Clear

```

A graph/pathfinding engine then determines a safe route.

Use a traditional algorithm such as:

- Dijkstra
- A\*

No need to train AI for shortest-path routing.

Example:

```text
🚑 RECOMMENDED ROUTE

Base
 ↓
Road A
 ↓
Road D
 ↓
Zone A3

Distance: 1.8 km

✓ avoids flooded section
✓ avoids blocked bridge
✓ roads appear accessible

```

This is a hybrid AI + algorithmic decision system.

---

# 20. Incident Event Feed

FloodSight should continuously create events.

Example:

```text
10:42:17 🔴 New critical rescue zone identified — A3
10:42:09 🚧 Road R04 classified as blocked
10:41:52 👤 Additional people detected in Zone A3
10:41:31 🌊 Flood coverage increased
10:40:58 🟠 Building B12 classified as severe damage

```

This makes the product feel like continuous disaster monitoring rather than a static computer-vision demo.

---

# 21. Confidence Levels

Predictions should include model confidence.

Example:

```text
Road Blocked
Confidence: 94%

```

Low-confidence results should not be presented as absolute truth.

Example:

```text
⚠ POSSIBLE BLOCKAGE
Confidence: 61%

Human verification recommended.

```

This is important for responsible disaster-response AI.

---

# 22. Automatic Incident Report

FloodSight should support generating a concise situation report.

Example:

```text
FLOODSIGHT INCIDENT REPORT

Incident severity:
SEVERE

Analysed area:
2.4 km²

Estimated flooded area:
38%

People detected:
17

Critical rescue zones:
2

Blocked roads:
4

Severely damaged structures:
6

Highest priority:
Zone A3 — 94/100

Reason:
8 individuals detected in an isolated
flooded region with primary access routes blocked.

Recommended response:
Prioritise Zone A3.

```

For the hackathon, measurements such as actual km² should only be shown as real measurements if geographical calibration is available.

Otherwise label them appropriately as estimated/demo values.

---

# 23. Optional FloodSight Assistant

Potential feature:

```text
Ask FloodSight

> Which area requires immediate attention?

Zone A3 currently has the highest rescue priority.

8 people have been detected there and nearby
ground access is blocked.

> Can an ambulance reach the zone?

Road R04 appears blocked.

Alternative Route R07 → R11 currently appears accessible.

```

This does NOT need a complicated autonomous agent.

The backend already contains structured data such as:

```json
{
  "people": 8,
  "flood_severity": 0.91,
  "roads_blocked": 2,
  "priority": 94
}

```

An LLM can receive this structured context and explain it conversationally.

This feature is optional for the hackathon and should not delay the core product.

---

# 24. Frontend Requirements

The frontend needs to be EXCELLENT.

Do not make it look like a generic Bootstrap dashboard or a Python OpenCV demo.

Desired style:

**modern emergency command-centre interface**

Potential stack:

```text
React
TypeScript
Tailwind CSS

```

Additional frontend libraries can be chosen as needed.

Main dashboard layout concept:

```text
┌──────────────────────────────────────────────────────┐
│ FLOODSIGHT        INCIDENT FS-001           ● LIVE │
├───────────────────────────────┬──────────────────────┤
│                               │ INCIDENT OVERVIEW    │
│                               │                      │
│       LIVE DRONE FEED         │ 🔴 SEVERE            │
│                               │ People          12   │
│ AI segmentation + detections  │ Flooded Area    43%  │
│                               │ Blocked Roads     3   │
│                               │ Damaged Bldgs     8   │
├───────────────────────────────┴──────────────────────┤
│ RESCUE PRIORITIES       │ TACTICAL MAP              │
│                        │                           │
│ 🔴 #1 Zone A    94     │ map / zones / routes      │
│ 🟠 #2 Zone C    76     │                           │
│ 🟡 #3 Zone B    52     │                           │
├──────────────────────────────────────────────────────┤
│ LIVE EVENTS                                          │
│ 10:43:12 Zone A escalated to critical               │
│ 10:43:05 New road blockage detected                 │
└──────────────────────────────────────────────────────┘

```

Important UI features:

- dark command-centre theme
- attractive typography
- clear severity colours
- subtle animations
- live status indicator
- statistics cards
- segmentation overlay
- detection bounding boxes
- clickable zones
- zone details drawer
- rescue-priority ranking
- tactical map
- live event timeline
- charts only when useful
- video controls
- confidence indicators
- incident report
- graceful loading/error states

The UI should feel like a polished SaaS/emergency-management platform.

---

# 25. Preferred Backend

Recommended:

**FastAPI + Python**

Reason:

The ML inference stack is naturally Python-based.

Backend responsibilities:

```text
video ingestion
webcam frames
model inference
SegFormer
YOLO
zone generation
temporal tracking
priority calculation
event generation
routing
incident summaries
WebSocket/API communication

```

Frontend receives structured results rather than doing ML itself.

Potential communication:

- REST for normal requests
- WebSocket for live results

---

# 26. Suggested Project Architecture

Conceptually:

```text
floodsight/
│
├── frontend/
│
│   React + TypeScript
│
├── backend/
│   ├── api/
│   ├── inference/
│   ├── zones/
│   ├── priority/
│   ├── routing/
│   ├── events/
│   └── reports/
│
├── ml/
│   ├── segmentation/
│   │   ├── datasets/
│   │   ├── training/
│   │   └── evaluation/
│   │
│   └── detection/
│       ├── datasets/
│       ├── training/
│       └── evaluation/
│
├── models/
│   ├── segmentation/
│   └── detection/
│
├── demo/
│   └── videos/
│
└── docs/

```

Exact architecture can be improved if there is a better engineering approach.

---

# 27. Hardware Available

I have access to a server containing:

**2 × NVIDIA H100 GPUs**

I may not always be able to fully utilize both GPUs simultaneously.

That is fine.

Even ONE H100 is more than enough for this hackathon training workload.

Do NOT waste excessive time trying to perfect distributed/multi-GPU utilisation unless it materially helps.

Ideal parallel setup if both GPUs are available:

```text
H100 #1
↓
SegFormer training
FloodNet + RescueNet

H100 #2
↓
YOLO training
VisDrone

```

Otherwise train jobs sequentially on one GPU.

---

# 28. Training vs Demo Machine

Training can occur on the H100 server.

After training:

```text
H100 server
    ↓
best segmentation checkpoint
best detection checkpoint
    ↓
copy/export
    ↓
hackathon demo machine

```

Prefer LOCAL inference on the demo machine if performance is acceptable.

Reason:

Do not depend completely on unreliable hackathon internet/network access.

If local inference isn't fast enough:

```text
Demo laptop
   ↓
H100 inference API
   ↓
results

```

But maintain some local/demo fallback so the presentation doesn't collapse if connectivity fails.

---

# 29. Codex Usage

I have access to Codex and want it to build as much of FloodSight as possible.

Codex should be used for:

- repository setup
- frontend
- backend
- dataset download scripts
- dataset converters
- label harmonisation code
- SegFormer training
- YOLO training
- evaluation scripts
- video ingestion
- webcam integration
- inference APIs
- WebSockets
- zone engine
- temporal smoothing
- priority engine
- tactical-map UI
- routing algorithms
- incident timeline
- reports
- tests
- Docker/startup scripts
- error handling
- demo hardening

Codex can build almost all software, but it cannot magically guarantee model quality.

Human inspection/testing is still required for:

- labels
- segmentation quality
- false positives
- detection thresholds
- model choice
- demo video selection

---

# 30. Codex Development Strategy

DO NOT give Codex one enormous prompt like:

> "Build FloodSight."

Develop in controlled phases.

Suggested order:

### Phase 0 — Repository / architecture

Create:

- frontend
- backend
- ML directories
- configs
- README
- environment setup

---

### Phase 1 — Product UI using simulated data

Build the COMPLETE beautiful frontend FIRST using mocked disaster information.

Implement:

- live video panel
- stats
- rescue priorities
- zones
- tactical map
- event feed
- zone drawer
- incident report UI

This guarantees we have a demoable product even before ML is complete.

---

### Phase 2 — Video input

Implement:

- file upload
- prerecorded video
- webcam
- frame extraction
- live result stream

Initially results can still be mocked.

---

### Phase 3 — Dataset pipeline

Download/prepare:

- FloodNet
- RescueNet
- VisDrone

Create:

- validation scripts
- taxonomy mapping
- training/validation/test handling
- dataset inspection utilities

---

### Phase 4 — Segmentation training

Train/fine-tune:

**SegFormer**

using:

```text
FloodNet + RescueNet

```

Evaluate masks visually and quantitatively.

---

### Phase 5 — Detection training

Train/fine-tune:

**YOLO**

using:

```text
VisDrone

```

Focus especially on:

- people
- cars
- vans
- trucks
- buses

Test specifically on actual flood drone videos.

---

### Phase 6 — Real-time inference

Connect real models to video pipeline.

Output structured data per frame.

---

### Phase 7 — Zone Engine

Implement:

- 4×4 internal risk grid
- risk aggregation
- adjacent-cell merging
- polygon/region output
- temporal tracking
- stable zone IDs

---

### Phase 8 — Priority Engine

Compute:

```text
Priority 0–100

```

using explainable risk components.

Generate ranked list.

---

### Phase 9 — Tactical Intelligence

Add:

- road graph
- simple route recommendations
- map visualization
- incident events
- reports

---

### Phase 10 — Demo Hardening

Test multiple flood videos.

Tune:

- thresholds
- colours
- FPS
- resolution
- UI animations
- frame sampling
- model confidence thresholds

Prepare fallback behaviour.

---

# 31. Development Time Expectation

Aggressively using Codex + existing pretrained models + H100 compute:

A strong hackathon MVP may be achievable in roughly:

**12–20 focused engineering hours**

A more polished version benefits greatly from:

**24–36 hours**

Approximate breakdown previously discussed:

```text
Frontend + mock data:
3–5 hours

Backend/video integration:
3–5 hours

Dataset preparation + first model training:
2–6 hours

Integration/testing/demo selection:
3–5 hours

```

These are planning estimates, not guarantees.

The key is to parallelise work.

---

# 32. Parallel Work Strategy

If possible:

```text
TRACK A — PRODUCT
Codex
↓
Frontend + dashboard


TRACK B — SEGMENTATION
H100 #1
↓
FloodNet + RescueNet
↓
SegFormer


TRACK C — DETECTION
H100 #2
↓
VisDrone
↓
YOLO


TRACK D — INTEGRATION
Codex
↓
video
models
zones
priority engine
frontend

```

A/B/C can happen simultaneously.

---

# 33. Demo Videos

We discussed using real aerial flood footage.

A particularly promising type is:

**Hurricane Harvey drone footage**

because FloodNet itself contains post-Hurricane-Harvey UAV imagery, making the visual domain relatively similar.

Previously identified examples included:

- CBC News — Tropical Storm Harvey drone footage of flood damage
- TIME — Hurricane Harvey destruction / before-after drone footage

However:

Do NOT choose the demo video simply because it is dramatic.

Before the hackathon, test the trained models against around **10+ candidate videos**.

Pick the video where FloodSight gives the cleanest and most compelling predictions.

Keep held-out dataset results separate from demo-video selection so demo selection is not presented as benchmark accuracy.

---

# 34. Two Demo Video Types

Ideally have TWO prepared clips.

## Clip A — Area Assessment

Higher altitude.

Best for showing:

- flood extent
- roads
- buildings
- flood masks
- affected zones

---

## Clip B — Rescue Assessment

Lower altitude.

Should visibly contain:

- people
- vehicles
- flooded roads
- houses/buildings

Best for showing:

```text
👤 Person detections
        ↓
Rescue zone
        ↓
Priority score
        ↓
CRITICAL

```

Tiny people in high-altitude footage may be only a few pixels large and could be difficult for the detector.

---

# 35. Desired Hero Demo

The hackathon sequence should approximately be:

```text
1. Open FloodSight.

2. Dashboard appears.

3. Select LIVE CAMERA or VIDEO.

4. Flood footage starts.

5. Flood segmentation appears.

6. Roads are classified.

7. People/vehicles get detected.

8. Disaster zones appear.

9. Dashboard updates:

Flooded Area        42%
People               6
Blocked Roads         2
Damaged Buildings     5

10. Rescue Priority list appears:

🔴 #1 Zone 2     92
🟠 #2 Zone 4     76
🟡 #3 Zone 1     54

11. Click Zone 2.

Zone details:

Priority: CRITICAL

Why?
• 5 people detected
• 81% flooding
• primary road inaccessible
• damaged structures nearby

12. FloodSight recommends this as the first response area.

13. Tactical map highlights access/blocked roads.

14. Optional recommended rescue route appears.

15. Event timeline updates while footage runs.

16. Generate Incident Report.

```

The emotional/technical "wow moment" should be:

> **FloodSight doesn't merely detect flooding — it automatically tells responders where they should go first and why.**

---

# 36. What MUST Actually Work

Highest priority features:

1. Video input
2. Webcam input
3. Real flood segmentation
4. Person/vehicle detection
5. Road state recognition
6. Rescue-zone generation
7. Ranked priority list
8. Explainable priority score
9. Live dashboard
10. Incident/event feed

Very desirable:

11. Clickable zone details
12. Tactical map
13. Incident report

Can be partially mocked/derived in the hackathon if necessary:

14. True GIS coordinates
15. Real-world route distances
16. Fully production-grade rescue routing
17. LLM assistant
18. Multi-disaster support

Do NOT fake core ML results while claiming they are real.

Derived/demo-only values should be labelled accordingly.

---

# 37. Product Positioning

Do NOT pitch the project as:

> "We trained YOLO and SegFormer on drone data."

Pitch it as:

> **FloodSight is a real-time geospatial disaster decision-intelligence platform that transforms heterogeneous drone observations into priority-aware rescue and resource-allocation decisions.**

Simpler judge-friendly wording:

> **FloodSight turns hours of drone footage into an immediately actionable disaster map — showing responders where people are at risk, which roads are inaccessible, and where rescue resources should go first.**

The models are components.

The PRODUCT is the decision system.

---

# 38. Key Engineering Principle

Always build **backwards from the hackathon demo**.

Do not spend the entire hackathon training models and end with:

```text
python detect.py

```

Instead:

```text
Beautiful product
        ↓
mock data
        ↓
working dashboard
        ↓
real models connected incrementally

```

If one ML component is imperfect, FloodSight should still remain a coherent product.

---

# 39. Reliability Strategy

The hackathon demo should be bulletproof.

Prepare:

- video-file mode
- webcam mode
- known-good demo clips
- pretrained checkpoints saved locally
- model-loading error handling
- confidence thresholds
- graceful UI fallback
- optional mock/demo data only as explicitly labelled backup

Avoid depending completely on:

- internet
- remote APIs
- live drone hardware
- cloud LLMs
- H100 server connectivity

for the main demonstration.

---

# 40. Model Evaluation

Do not judge quality just from loss curves.

For segmentation inspect:

- IoU / mIoU
- Dice/F1
- per-class IoU
- visual masks
- road-boundary quality
- confusion between water and flooded road
- building-damage confusion

For detection inspect:

- precision
- recall
- mAP
- tiny-person performance
- false positives on rooftops/debris
- performance on unseen disaster video

The most important practical test:

Run the full pipeline on actual candidate hackathon footage.

---

# 41. Possible Detection Enhancement

People may be very small in drone footage.

Potential improvements:

- higher inference resolution
- image tiling/slicing
- SAHI-style sliced inference
- temporal tracking
- lower but controlled detection threshold
- choose a larger YOLO variant if compute allows

Use these only if necessary after baseline evaluation.

---

# 42. Important Responsible-AI Principle

FloodSight should be a **decision-support system**, NOT an autonomous authority.

Do not say:

> AI decides who lives or dies.

Say:

> FloodSight prioritizes zones and explains the evidence so emergency personnel can make faster informed decisions.

Low-confidence observations should be flagged for human verification.

---

# 43. Current Overall Architecture

Final current concept:

```text
                         DRONE / VIDEO
                               │
                               ▼
                       VIDEO INGESTION
                               │
                   ┌───────────┴───────────┐
                   ▼                       ▼
          DISASTER SEGMENTATION      OBJECT DETECTION
             SegFormer                   YOLO
                   │                       │
       Flood / roads / buildings     People / vehicles
                   │                       │
                   └──────────┬────────────┘
                              ▼
                      SCENE UNDERSTANDING
                              │
              ┌───────────────┼─────────────────┐
              ▼               ▼                 ▼
          ZONE ENGINE     ACCESS ANALYSIS    DAMAGE MAP
              │               │
              ▼               ▼
         PRIORITY ENGINE   ROAD GRAPH
              │               │
              │               ▼
              │            ROUTING
              │               │
              └───────┬───────┘
                      ▼
              FLOODSIGHT BACKEND
                      │
                      ▼
           COMMAND-CENTRE FRONTEND
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
 Rescue Priorities  Tactical Map   Event Feed
       │              │              │
       └──────────────┼──────────────┘
                      ▼
              INCIDENT REPORT

```

---

# 44. Immediate Recommended Next Steps

Continue from here rather than redesigning the project.

The next concrete sequence should be:

1. Freeze FloodSight repository architecture.
2. Build polished frontend with simulated data.
3. Download FloodNet.
4. Download RescueNet.
5. Download VisDrone-DET.
6. Inspect exact annotation formats/classes.
7. Design exact FloodSight unified taxonomy.
8. Write reliable conversion scripts.
9. Create segmentation training pipeline.
10. Create YOLO training pipeline.
11. Train models using H100 server.
12. Evaluate.
13. Connect checkpoints to FastAPI inference.
14. Connect live video.
15. Implement zone generation.
16. Implement temporal smoothing.
17. Implement explainable priority engine.
18. Connect results to UI via WebSocket.
19. Test several flood videos.
20. Choose the strongest demo footage.
21. Add route/tactical-map functionality.
22. Harden final demo.

---

# 45. Instruction to the New Assistant

From this point onward:

- Treat the architecture above as the current agreed plan.
- Do not repeatedly ask me to redefine FloodSight.
- Do not simplify it back into only a flood-recognition model.
- Help me actually BUILD the project.
- Prefer practical implementation over theoretical discussion.
- Give exact commands, code architecture, Codex prompts, training configurations, and troubleshooting steps where appropriate.
- Keep the hackathon time constraint in mind.
- Prioritize a working impressive demo.
- Avoid unnecessary complexity that does not improve the final presentation.
- When making ML decisions, consider that I have access to up to **2 NVIDIA H100 GPUs**.
- When making software decisions, assume I can use **Codex extensively**.
- Build the product incrementally and verify each stage before moving on.
- Preserve the distinction between real ML outputs, derived analytics, and mocked/demo-only functionality.
- The key differentiator is **priority-aware rescue decision intelligence**, not merely computer vision.

The next conversation should continue directly from this context.
