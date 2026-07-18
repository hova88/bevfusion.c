#!/usr/bin/env python3
"""Match canonical detection JSON by class/center and enforce full-graph gates."""

import argparse
import json
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    reference = json.load(open(args.reference, encoding="utf-8"))["detections"]
    candidate = json.load(open(args.candidate, encoding="utf-8"))["detections"]
    if len(reference) != len(candidate):
        raise SystemExit(f"count mismatch: {len(reference)} != {len(candidate)}")
    unused = set(range(len(candidate)))
    pairs = []
    for expected in reference:
        choices = [index for index in unused
                   if candidate[index]["class_id"] == expected["class_id"]]
        if not choices:
            raise SystemExit(f"no class match for {expected['class_id']}")
        index = min(choices, key=lambda item: sum(
            (expected[axis] - candidate[item][axis]) ** 2
            for axis in ("x", "y", "z")))
        unused.remove(index)
        pairs.append((expected, candidate[index]))
    limits = {"score": 1.5e-4, "width": 1.5e-3, "length": 1.5e-3,
              "height": 1.5e-3, "velocity_x": 4e-4, "velocity_y": 4e-4}
    maximum = {}
    center = []
    yaw = []
    for expected, actual in pairs:
        center.append(math.sqrt(sum((expected[a] - actual[a]) ** 2
                                    for a in ("x", "y", "z"))))
        yaw.append(abs(math.atan2(math.sin(expected["yaw"] - actual["yaw"]),
                                 math.cos(expected["yaw"] - actual["yaw"]))))
        for field in limits:
            maximum[field] = max(maximum.get(field, 0),
                                 abs(expected[field] - actual[field]))
    ok = max(center, default=0) <= 3e-3 and max(yaw, default=0) <= 1.5e-2
    ok &= all(maximum[field] <= limit for field, limit in limits.items())
    print(f"matched={len(pairs)} center_max={max(center, default=0):.6g} "
          f"yaw_max={max(yaw, default=0):.6g} " + " ".join(
              f"{field}_max={maximum[field]:.6g}" for field in limits))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
