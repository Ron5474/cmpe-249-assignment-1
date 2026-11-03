# ngdet evaluation report

Unified taxonomy: **driving3_bare** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| gdino:IDEA-Research/grounding-dino-tiny | 0.154 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| gdino:IDEA-Research/grounding-dino-tiny | 0.322 |

## Full COCO tables (per run)

### gdino:IDEA-Research/grounding-dino-tiny  ×  kitti  (7481 imgs, 982.5s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.154 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.322 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.128 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.087 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.165 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.251 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.151 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.391 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.517 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.493 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.503 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.566 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.158 |
| person | 0.258 |
| cyclist | 0.046 |
