# Evaluation protocol deviates from the ngdet tutorial defaults

All mAP runs use `--score-thr 0.001` and `--max-images 0` (the tutorial uses 0.3 and 200), because a 0.3 cutoff truncates the precision-recall curve and makes our numbers incomparable to published COCO mAP; 0.3 is kept only for qualitative videos. Cross-dataset numbers are scored in the `driving3` taxonomy on every dataset, COCO included, so that degradation reflects the domain change rather than a change in which classes are counted.

## Consequences

- KITTI `DontCare` regions are left as background (stock ngdet behaviour), so detections of distant unlabeled objects count as false positives. This is documented as a limitation, not patched.
- KITTI numbers are COCO-style AP over size bins, not the official Easy/Moderate/Hard KITTI metric, and are not comparable to the KITTI leaderboard.
- The `cyclist` class is a known definition mismatch (COCO `bicycle` is the bike only; KITTI `Cyclist` is bike + rider) and is discussed rather than excluded.
