# CMPE 249 HW: 2D Object Detection Evaluation

Track A (experimental) evaluation of pretrained 2D object detectors across datasets, measuring accuracy, speed, and generalization under domain shift.

## Language

**Detector family**:
A group of detectors sharing a core design (e.g. two-stage anchor-based, one-stage anchor-free, query-based transformer, open-vocabulary). Different sizes of one model are the same family.
_Avoid_: Architecture type, model type

**Model configuration**:
One specific detector checkpoint evaluated as a unit (e.g. YOLO11x COCO-pretrained). The assignment requires at least 3, from at least 2 detector families.
_Avoid_: Model variant, run

**Unified taxonomy**:
The shared class list that every dataset's labels and every detector's outputs are folded into so they can be scored against each other.
_Avoid_: Label space, class mapping

**Domain shift**:
The change in data distribution between the dataset a detector was trained on (COCO) and the dataset it is evaluated on (e.g. driving scenes).
_Avoid_: Distribution drift, dataset gap

**Reproduction check**:
Evaluating a closed-set model on full COCO val2017 in all 80 classes and comparing against its published mAP, to validate the evaluation harness before trusting its cross-dataset numbers.
_Avoid_: Sanity run, baseline test

**DontCare region**:
A KITTI-annotated area containing objects too distant or unclear to label; in this project it is treated as background, not as an ignore region.
_Avoid_: Ignore box, crowd region

**Cyclist**:
In the unified taxonomy, a bicycle with or without its rider. Datasets disagree on whether the box covers the rider (KITTI) or only the bike (COCO), which is a definition mismatch rather than a detector error.
_Avoid_: Bicycle, bike rider
