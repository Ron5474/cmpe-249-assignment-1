# Score Grounding DINO classes from prompt tokens, not decoded phrases

ngdet labels each Grounding DINO box by decoding a phrase from tokens above `text_threshold` (0.25) and matching it to a class name. That labelled every empty phrase "vehicle" (the empty string is a substring of every name), so boxes below `text_threshold` were kept with a wrong label. A first fix dropped empty and mixed phrases instead; that removed the mislabelling but turned `text_threshold` = 0.25 into an effective score floor, unlike every other model's 0.001, and discarded mixed phrases such as "person cyclist" that often cover real people. We instead score classes directly from the prompt tokens, following the official Grounding DINO COCO evaluator (`PostProcessCocoGrounding`): a class's probability is the mean of its tokens' probabilities, and the top 300 query–class pairs are kept, so one query can predict several classes and no text threshold applies.

## Consequences

- Our curated prompts have several terms per class ("car. truck. bus. van"), which the official evaluator does not cover. Our adaptation averages tokens within each term, then takes the maximum term within the class, before top-k, so synonyms cannot produce duplicate boxes from one query. This pooling rule is ours, not the official protocol.
- Grounding DINO's `predict()` is replaced in `scripts/detectors.py`; ngdet stays unmodified (ADR-0002). `scripts/compare_gdino_postprocess.py` records how the three post-processors differ on the same forward passes.
