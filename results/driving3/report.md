# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | coco | kitti |
|---|---|---|
| yolo:yolo11s.pt | 0.476 | 0.225 |
| yolo:yolo11x.pt | 0.558 | 0.240 |
| hf_detr:PekingU/rtdetr_r50vd | 0.517 | 0.245 |
| yolo:yolov8s-worldv2.pt | 0.344 | 0.210 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.493 | 0.238 |
| gdino:IDEA-Research/grounding-dino-tiny | 0.423 | 0.219 |

## Summary matrix — AP50

| model \ dataset | coco | kitti |
|---|---|---|
| yolo:yolo11s.pt | 0.693 | 0.429 |
| yolo:yolo11x.pt | 0.773 | 0.446 |
| hf_detr:PekingU/rtdetr_r50vd | 0.754 | 0.450 |
| yolo:yolov8s-worldv2.pt | 0.504 | 0.399 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.755 | 0.442 |
| gdino:IDEA-Research/grounding-dino-tiny | 0.587 | 0.415 |

## Generalization gap (COCO mAP − driving-dataset mAP)

Positive = drop when moving from the COCO training domain to a driving domain (higher = worse generalization).

| model | kitti |
|---|---|
| yolo:yolo11s.pt | +0.251 |
| yolo:yolo11x.pt | +0.318 |
| hf_detr:PekingU/rtdetr_r50vd | +0.272 |
| yolo:yolov8s-worldv2.pt | +0.134 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | +0.255 |
| gdino:IDEA-Research/grounding-dino-tiny | +0.204 |

## Full COCO tables (per run)

### yolo:yolo11s.pt  ×  coco  (5000 imgs, 46.7s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.476 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.693 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.510 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.201 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.488 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.704 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.229 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.538 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.620 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.382 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.671 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.795 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.516 |
| person | 0.562 |
| cyclist | 0.350 |

### yolo:yolo11s.pt  ×  kitti  (7481 imgs, 114.4s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.225 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.429 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.215 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.105 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.213 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.315 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.105 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.299 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.341 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.247 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.346 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.412 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.445 |
| person | 0.229 |
| cyclist | 0.001 |

### yolo:yolo11x.pt  ×  coco  (5000 imgs, 58.5s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.558 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.773 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.606 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.295 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.583 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.755 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.255 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.612 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.681 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.468 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.733 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.826 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.595 |
| person | 0.627 |
| cyclist | 0.451 |

### yolo:yolo11x.pt  ×  kitti  (7481 imgs, 121.5s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.240 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.446 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.235 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.135 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.228 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.314 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.109 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.314 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.341 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.284 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.344 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.385 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.476 |
| person | 0.243 |
| cyclist | 0.001 |

### hf_detr:PekingU/rtdetr_r50vd  ×  coco  (5000 imgs, 105.2s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.517 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.754 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.542 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.266 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.527 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.726 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.237 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.576 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.671 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.487 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.688 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.833 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.563 |
| person | 0.593 |
| cyclist | 0.395 |

### hf_detr:PekingU/rtdetr_r50vd  ×  kitti  (7481 imgs, 223.1s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.245 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.450 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.238 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.132 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.234 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.327 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.110 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.320 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.362 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.326 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.354 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.404 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.492 |
| person | 0.241 |
| cyclist | 0.001 |

### yolo:yolov8s-worldv2.pt  ×  coco  (5000 imgs, 49.5s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.344 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.504 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.363 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.138 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.335 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.558 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.194 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.463 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.527 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.303 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.553 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.724 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.375 |
| person | 0.517 |
| cyclist | 0.139 |

### yolo:yolov8s-worldv2.pt  ×  kitti  (7481 imgs, 107.6s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.210 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.399 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.202 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.099 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.205 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.311 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.109 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.309 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.346 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.231 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.343 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.453 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.406 |
| person | 0.216 |
| cyclist | 0.008 |

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  coco  (5000 imgs, 155.8s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.493 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.755 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.524 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.288 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.500 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.662 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.232 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.556 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.620 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.456 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.644 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.754 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.534 |
| person | 0.566 |
| cyclist | 0.380 |

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  kitti  (7481 imgs, 289.0s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.238 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.442 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.235 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.121 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.232 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.320 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.112 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.324 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.364 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.337 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.359 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.400 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.457 |
| person | 0.256 |
| cyclist | 0.001 |

### gdino:IDEA-Research/grounding-dino-tiny  ×  coco  (5000 imgs, 718.0s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.423 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.587 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.468 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.235 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.462 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.586 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.227 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.579 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.705 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.523 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.729 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.854 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.473 |
| person | 0.536 |
| cyclist | 0.258 |

### gdino:IDEA-Research/grounding-dino-tiny  ×  kitti  (7481 imgs, 947.3s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.219 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.415 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.211 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.130 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.232 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.325 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.115 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.465 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.550 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.527 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.530 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.619 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.439 |
| person | 0.189 |
| cyclist | 0.029 |
