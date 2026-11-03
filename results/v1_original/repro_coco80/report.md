# ngdet evaluation report

Unified taxonomy: **coco80** = `['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | coco |
|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.459 |
| yolo:yolo11s.pt | 0.450 |
| yolo:yolo11x.pt | 0.529 |
| hf_detr:PekingU/rtdetr_r50vd | 0.482 |

## Summary matrix — AP50

| model \ dataset | coco |
|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.663 |
| yolo:yolo11s.pt | 0.613 |
| yolo:yolo11x.pt | 0.692 |
| hf_detr:PekingU/rtdetr_r50vd | 0.648 |

## Generalization gap (COCO mAP − driving-dataset mAP)

Positive = drop when moving from the COCO training domain to a driving domain (higher = worse generalization).

| model |  |
|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 |  |
| yolo:yolo11s.pt |  |
| yolo:yolo11x.pt |  |
| hf_detr:PekingU/rtdetr_r50vd |  |

## Full COCO tables (per run)

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  coco  (5000 imgs, 94.4s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.459 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.663 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.503 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.242 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.448 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.587 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.365 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.577 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.606 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.409 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.608 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.706 |

| Class | AP@[.5:.95] |
|---|---|
| person | 0.565 |
| bicycle | 0.378 |
| car | 0.486 |
| motorcycle | 0.491 |
| airplane | 0.684 |
| bus | 0.705 |
| train | 0.687 |
| truck | 0.423 |
| boat | 0.313 |
| traffic light | 0.310 |
| fire hydrant | 0.696 |
| stop sign | 0.701 |
| parking meter | 0.470 |
| bench | 0.303 |
| bird | 0.371 |
| cat | 0.719 |
| dog | 0.664 |
| horse | 0.637 |
| sheep | 0.528 |
| cow | 0.585 |
| elephant | 0.643 |
| bear | 0.691 |
| zebra | 0.690 |
| giraffe | 0.695 |
| backpack | 0.201 |
| umbrella | 0.438 |
| handbag | 0.203 |
| tie | 0.390 |
| suitcase | 0.446 |
| frisbee | 0.690 |
| skis | 0.294 |
| snowboard | 0.450 |
| sports ball | 0.502 |
| kite | 0.419 |
| baseball bat | 0.376 |
| baseball glove | 0.433 |
| skateboard | 0.584 |
| surfboard | 0.434 |
| tennis racket | 0.577 |
| bottle | 0.423 |
| wine glass | 0.400 |
| cup | 0.476 |
| fork | 0.431 |
| knife | 0.271 |
| spoon | 0.252 |
| bowl | 0.461 |
| banana | 0.255 |
| apple | 0.236 |
| sandwich | 0.390 |
| orange | 0.343 |
| broccoli | 0.227 |
| carrot | 0.197 |
| hot dog | 0.357 |
| pizza | 0.538 |
| donut | 0.456 |
| cake | 0.366 |
| chair | 0.329 |
| couch | 0.455 |
| potted plant | 0.313 |
| bed | 0.419 |
| dining table | 0.299 |
| toilet | 0.637 |
| tv | 0.582 |
| laptop | 0.649 |
| mouse | 0.666 |
| remote | 0.399 |
| keyboard | 0.562 |
| cell phone | 0.428 |
| microwave | 0.613 |
| oven | 0.387 |
| toaster | 0.564 |
| sink | 0.430 |
| refrigerator | 0.605 |
| book | 0.151 |
| clock | 0.545 |
| vase | 0.410 |
| scissors | 0.363 |
| teddy bear | 0.516 |
| hair drier | 0.118 |
| toothbrush | 0.355 |

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

### hf_detr:PekingU/rtdetr_r50vd  ×  coco  (5000 imgs, 136.4s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.482 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.648 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.525 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.254 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.473 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.629 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.355 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.598 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.661 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.472 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.670 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.772 |

| Class | AP@[.5:.95] |
|---|---|
| person | 0.593 |
| bicycle | 0.395 |
| car | 0.500 |
| motorcycle | 0.000 |
| airplane | 0.000 |
| bus | 0.759 |
| train | 0.762 |
| truck | 0.457 |
| boat | 0.354 |
| traffic light | 0.325 |
| fire hydrant | 0.756 |
| stop sign | 0.705 |
| parking meter | 0.531 |
| bench | 0.331 |
| bird | 0.436 |
| cat | 0.817 |
| dog | 0.757 |
| horse | 0.691 |
| sheep | 0.628 |
| cow | 0.643 |
| elephant | 0.714 |
| bear | 0.848 |
| zebra | 0.783 |
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
| baseball bat | 0.500 |
| baseball glove | 0.433 |
| skateboard | 0.629 |
| surfboard | 0.528 |
| tennis racket | 0.600 |
| bottle | 0.454 |
| wine glass | 0.442 |
| cup | 0.510 |
| fork | 0.535 |
| knife | 0.327 |
| spoon | 0.333 |
| bowl | 0.503 |
| banana | 0.301 |
| apple | 0.275 |
| sandwich | 0.508 |
| orange | 0.380 |
| broccoli | 0.282 |
| carrot | 0.282 |
| hot dog | 0.520 |
| pizza | 0.619 |
| donut | 0.549 |
| cake | 0.439 |
| chair | 0.386 |
| couch | 0.000 |
| potted plant | 0.000 |
| bed | 0.564 |
| dining table | 0.000 |
| toilet | 0.716 |
| tv | 0.000 |
| laptop | 0.723 |
| mouse | 0.676 |
| remote | 0.461 |
| keyboard | 0.622 |
| cell phone | 0.471 |
| microwave | 0.699 |
| oven | 0.461 |
| toaster | 0.619 |
| sink | 0.473 |
| refrigerator | 0.707 |
| book | 0.182 |
| clock | 0.577 |
| vase | 0.445 |
| scissors | 0.435 |
| teddy bear | 0.615 |
| hair drier | 0.244 |
| toothbrush | 0.425 |
