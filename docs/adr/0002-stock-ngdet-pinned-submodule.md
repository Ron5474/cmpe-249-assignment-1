# Use stock ngdet as a pinned submodule, extended only from our wrapper

The course's `ngdet` harness is included as a git submodule pinned to one commit and never edited; anything we need beyond it (the `coco80` taxonomy for the reproduction check, peak GPU memory measurement) is added at runtime from our own wrapper scripts. This keeps our results attributable to a known harness version and makes the reproduction check meaningful, at the cost of accepting harness behaviour we would otherwise patch (see ADR-0001).
