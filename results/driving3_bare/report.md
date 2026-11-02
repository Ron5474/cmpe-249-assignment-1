# ngdet evaluation report

Unified taxonomy: **driving3_bare** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| gdino:IDEA-Research/grounding-dino-tiny | 0.140 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| gdino:IDEA-Research/grounding-dino-tiny | 0.284 |

## Full COCO tables (per run)

### gdino:IDEA-Research/grounding-dino-tiny  ×  kitti  (7481 imgs, 658.9s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.140 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.284 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.119 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.063 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.157 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.259 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.136 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.288 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.399 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.338 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.393 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.475 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.125 |
| person | 0.255 |
| cyclist | 0.039 |
