# ngdet evaluation report

Unified taxonomy: **coco80** = `['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | coco |
|---|---|
| yolo:yolo11s.pt | 0.450 |
| yolo:yolo11x.pt | 0.529 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.462 |
| hf_detr:PekingU/rtdetr_r50vd | 0.523 |

## Summary matrix — AP50

| model \ dataset | coco |
|---|---|
| yolo:yolo11s.pt | 0.613 |
| yolo:yolo11x.pt | 0.692 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.667 |
| hf_detr:PekingU/rtdetr_r50vd | 0.702 |

## Generalization gap (COCO mAP − driving-dataset mAP)

Positive = drop when moving from the COCO training domain to a driving domain (higher = worse generalization).

| model |  |
|---|
| yolo:yolo11s.pt |  |
| yolo:yolo11x.pt |  |
| torchvision:fasterrcnn_resnet50_fpn_v2 |  |
| hf_detr:PekingU/rtdetr_r50vd |  |

## Full COCO tables (per run)

### yolo:yolo11s.pt  ×  coco  (5000 imgs, 61.8s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.450 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.613 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.486 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.212 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.428 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.602 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.347 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.565 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.614 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.373 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.629 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.734 |

| Class | AP@[.5:.95] |
|---|---|
| person | 0.562 |
| bicycle | 0.350 |
| car | 0.440 |
| motorcycle | 0.493 |
| airplane | 0.746 |
| bus | 0.706 |
| train | 0.740 |
| truck | 0.354 |
| boat | 0.296 |
| traffic light | 0.267 |
| fire hydrant | 0.705 |
| stop sign | 0.651 |
| parking meter | 0.545 |
| bench | 0.277 |
| bird | 0.346 |
| cat | 0.721 |
| dog | 0.692 |
| horse | 0.621 |
| sheep | 0.559 |
| cow | 0.556 |
| elephant | 0.678 |
| bear | 0.744 |
| zebra | 0.711 |
| giraffe | 0.747 |
| backpack | 0.154 |
| umbrella | 0.447 |
| handbag | 0.155 |
| tie | 0.346 |
| suitcase | 0.447 |
| frisbee | 0.665 |
| skis | 0.282 |
| snowboard | 0.399 |
| sports ball | 0.453 |
| kite | 0.420 |
| baseball bat | 0.374 |
| baseball glove | 0.398 |
| skateboard | 0.589 |
| surfboard | 0.404 |
| tennis racket | 0.523 |
| bottle | 0.379 |
| wine glass | 0.368 |
| cup | 0.429 |
| fork | 0.390 |
| knife | 0.199 |
| spoon | 0.200 |
| bowl | 0.433 |
| banana | 0.269 |
| apple | 0.231 |
| sandwich | 0.412 |
| orange | 0.306 |
| broccoli | 0.233 |
| carrot | 0.237 |
| hot dog | 0.372 |
| pizza | 0.547 |
| donut | 0.461 |
| cake | 0.407 |
| chair | 0.333 |
| couch | 0.486 |
| potted plant | 0.298 |
| bed | 0.470 |
| dining table | 0.331 |
| toilet | 0.699 |
| tv | 0.609 |
| laptop | 0.649 |
| mouse | 0.604 |
| remote | 0.293 |
| keyboard | 0.534 |
| cell phone | 0.354 |
| microwave | 0.607 |
| oven | 0.436 |
| toaster | 0.488 |
| sink | 0.419 |
| refrigerator | 0.672 |
| book | 0.126 |
| clock | 0.522 |
| vase | 0.409 |
| scissors | 0.380 |
| teddy bear | 0.504 |
| hair drier | 0.030 |
| toothbrush | 0.295 |

### yolo:yolo11x.pt  ×  coco  (5000 imgs, 71.2s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.529 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.692 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.576 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.284 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.524 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.669 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.387 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.638 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.682 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.467 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.706 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.784 |

| Class | AP@[.5:.95] |
|---|---|
| person | 0.627 |
| bicycle | 0.451 |
| car | 0.533 |
| motorcycle | 0.575 |
| airplane | 0.798 |
| bus | 0.751 |
| train | 0.759 |
| truck | 0.426 |
| boat | 0.367 |
| traffic light | 0.336 |
| fire hydrant | 0.784 |
| stop sign | 0.722 |
| parking meter | 0.557 |
| bench | 0.364 |
| bird | 0.422 |
| cat | 0.772 |
| dog | 0.748 |
| horse | 0.714 |
| sheep | 0.629 |
| cow | 0.649 |
| elephant | 0.749 |
| bear | 0.814 |
| zebra | 0.774 |
| giraffe | 0.803 |
| backpack | 0.213 |
| umbrella | 0.534 |
| handbag | 0.259 |
| tie | 0.471 |
| suitcase | 0.526 |
| frisbee | 0.759 |
| skis | 0.385 |
| snowboard | 0.463 |
| sports ball | 0.508 |
| kite | 0.502 |
| baseball bat | 0.540 |
| baseball glove | 0.468 |
| skateboard | 0.671 |
| surfboard | 0.536 |
| tennis racket | 0.694 |
| bottle | 0.474 |
| wine glass | 0.477 |
| cup | 0.527 |
| fork | 0.549 |
| knife | 0.338 |
| spoon | 0.326 |
| bowl | 0.516 |
| banana | 0.304 |
| apple | 0.242 |
| sandwich | 0.462 |
| orange | 0.355 |
| broccoli | 0.283 |
| carrot | 0.277 |
| hot dog | 0.473 |
| pizza | 0.624 |
| donut | 0.560 |
| cake | 0.473 |
| chair | 0.432 |
| couch | 0.536 |
| potted plant | 0.362 |
| bed | 0.557 |
| dining table | 0.355 |
| toilet | 0.732 |
| tv | 0.659 |
| laptop | 0.729 |
| mouse | 0.684 |
| remote | 0.483 |
| keyboard | 0.594 |
| cell phone | 0.491 |
| microwave | 0.679 |
| oven | 0.458 |
| toaster | 0.479 |
| sink | 0.482 |
| refrigerator | 0.716 |
| book | 0.195 |
| clock | 0.574 |
| vase | 0.454 |
| scissors | 0.429 |
| teddy bear | 0.600 |
| hair drier | 0.227 |
| toothbrush | 0.462 |

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  coco  (5000 imgs, 192.2s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.462 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.667 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.505 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.244 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.451 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.590 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.368 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.592 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.626 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.436 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.635 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.726 |

| Class | AP@[.5:.95] |
|---|---|
| person | 0.566 |
| bicycle | 0.380 |
| car | 0.488 |
| motorcycle | 0.493 |
| airplane | 0.686 |
| bus | 0.707 |
| train | 0.688 |
| truck | 0.426 |
| boat | 0.314 |
| traffic light | 0.312 |
| fire hydrant | 0.701 |
| stop sign | 0.706 |
| parking meter | 0.476 |
| bench | 0.304 |
| bird | 0.373 |
| cat | 0.721 |
| dog | 0.668 |
| horse | 0.639 |
| sheep | 0.530 |
| cow | 0.589 |
| elephant | 0.646 |
| bear | 0.698 |
| zebra | 0.693 |
| giraffe | 0.700 |
| backpack | 0.203 |
| umbrella | 0.439 |
| handbag | 0.205 |
| tie | 0.391 |
| suitcase | 0.449 |
| frisbee | 0.690 |
| skis | 0.296 |
| snowboard | 0.451 |
| sports ball | 0.504 |
| kite | 0.422 |
| baseball bat | 0.382 |
| baseball glove | 0.436 |
| skateboard | 0.585 |
| surfboard | 0.438 |
| tennis racket | 0.580 |
| bottle | 0.426 |
| wine glass | 0.402 |
| cup | 0.478 |
| fork | 0.433 |
| knife | 0.272 |
| spoon | 0.254 |
| bowl | 0.463 |
| banana | 0.256 |
| apple | 0.237 |
| sandwich | 0.393 |
| orange | 0.345 |
| broccoli | 0.228 |
| carrot | 0.199 |
| hot dog | 0.362 |
| pizza | 0.541 |
| donut | 0.456 |
| cake | 0.369 |
| chair | 0.330 |
| couch | 0.457 |
| potted plant | 0.315 |
| bed | 0.422 |
| dining table | 0.300 |
| toilet | 0.641 |
| tv | 0.584 |
| laptop | 0.650 |
| mouse | 0.668 |
| remote | 0.403 |
| keyboard | 0.566 |
| cell phone | 0.428 |
| microwave | 0.614 |
| oven | 0.388 |
| toaster | 0.564 |
| sink | 0.432 |
| refrigerator | 0.606 |
| book | 0.152 |
| clock | 0.547 |
| vase | 0.412 |
| scissors | 0.365 |
| teddy bear | 0.517 |
| hair drier | 0.119 |
| toothbrush | 0.355 |

### hf_detr:PekingU/rtdetr_r50vd  ×  coco  (5000 imgs, 1061.6s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.523 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.702 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.569 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.272 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.505 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.677 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.389 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.652 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.718 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.503 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.720 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.836 |

| Class | AP@[.5:.95] |
|---|---|
| person | 0.594 |
| bicycle | 0.395 |
| car | 0.500 |
| motorcycle | 0.542 |
| airplane | 0.769 |
| bus | 0.759 |
| train | 0.762 |
| truck | 0.457 |
| boat | 0.354 |
| traffic light | 0.324 |
| fire hydrant | 0.756 |
| stop sign | 0.705 |
| parking meter | 0.531 |
| bench | 0.331 |
| bird | 0.436 |
| cat | 0.817 |
| dog | 0.756 |
| horse | 0.691 |
| sheep | 0.628 |
| cow | 0.643 |
| elephant | 0.714 |
| bear | 0.848 |
| zebra | 0.782 |
| giraffe | 0.765 |
| backpack | 0.213 |
| umbrella | 0.506 |
| handbag | 0.254 |
| tie | 0.433 |
| suitcase | 0.529 |
| frisbee | 0.766 |
| skis | 0.375 |
| snowboard | 0.480 |
| sports ball | 0.523 |
| kite | 0.523 |
| baseball bat | 0.499 |
| baseball glove | 0.433 |
| skateboard | 0.628 |
| surfboard | 0.528 |
| tennis racket | 0.600 |
| bottle | 0.454 |
| wine glass | 0.447 |
| cup | 0.510 |
| fork | 0.535 |
| knife | 0.327 |
| spoon | 0.333 |
| bowl | 0.503 |
| banana | 0.300 |
| apple | 0.275 |
| sandwich | 0.508 |
| orange | 0.379 |
| broccoli | 0.281 |
| carrot | 0.282 |
| hot dog | 0.521 |
| pizza | 0.619 |
| donut | 0.549 |
| cake | 0.439 |
| chair | 0.386 |
| couch | 0.549 |
| potted plant | 0.361 |
| bed | 0.564 |
| dining table | 0.390 |
| toilet | 0.716 |
| tv | 0.653 |
| laptop | 0.723 |
| mouse | 0.676 |
| remote | 0.462 |
| keyboard | 0.622 |
| cell phone | 0.471 |
| microwave | 0.698 |
| oven | 0.461 |
| toaster | 0.619 |
| sink | 0.472 |
| refrigerator | 0.707 |
| book | 0.182 |
| clock | 0.577 |
| vase | 0.444 |
| scissors | 0.435 |
| teddy bear | 0.615 |
| hair drier | 0.243 |
| toothbrush | 0.425 |
