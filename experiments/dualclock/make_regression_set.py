from __future__ import annotations

import argparse
import json
from pathlib import Path


SMOKE_CLASSES = (
    (1, "texture"),
    (8, "geometry"),
    (22, "animal"),
    (55, "animal"),
    (107, "animal"),
    (207, "animal"),
    (281, "animal"),
    (340, "animal"),
    (417, "geometry"),
    (438, "geometry"),
    (444, "texture"),
    (466, "geometry"),
    (483, "texture"),
    (507, "people-associated"),
    (515, "people-associated"),
    (574, "geometry"),
    (621, "text-like"),
    (673, "text-like"),
    (707, "geometry"),
    (738, "texture"),
    (784, "geometry"),
    (805, "text-like"),
    (852, "people-associated"),
    (888, "texture"),
    (954, "text-like"),
)

SCENES = (
    "A red fox curled beneath a frost-covered pine tree at sunrise",
    "A crowded night market with five distinct food stalls and hanging lanterns",
    "A glass sphere balanced on a black-and-white checkerboard",
    "A close-up studio portrait of an elderly watchmaker holding a tiny gear",
    "A modern library interior with repeating wooden arches and spiral stairs",
    "A handwritten chalkboard menu reading 'PIXEL CAFE' with prices",
    "Three astronauts planting colorful flags in a field of purple flowers",
    "A macro photograph of iridescent beetle wings covered in dew",
    "A yellow bicycle leaning against a turquoise brick wall",
    "An aerial view of rice terraces forming precise geometric curves",
    "Two children building a detailed sandcastle beside a striped umbrella",
    "A ceramic teapot, two cups, and sliced oranges on patterned fabric",
    "A snowy owl flying between tall city buildings during a blizzard",
    "A storefront sign reading 'OPEN 24 HOURS' reflected in a rain puddle",
    "A chef plating seven colorful dumplings in a bright restaurant kitchen",
    "A wooden suspension bridge crossing a misty jungle waterfall",
    "A robotic hand threading a needle beside spools of colored thread",
    "An ornate cathedral facade with hundreds of small carved figures",
    "A transparent cube containing a miniature desert and a lone cactus",
    "Four musicians performing on a small subway platform",
    "A detailed map of an imaginary island labeled with mountains and rivers",
    "A black cat sitting among repeating rows of white porcelain masks",
    "A family picnic with six people, a kite, and a red thermos",
    "Bioluminescent jellyfish drifting through a dark underwater cave",
    "A vintage train arriving at a platform beneath a large clock showing 8:15",
)

STYLES = (
    "Photorealistic, natural light, fine surface detail.",
    "Editorial illustration, crisp shapes, restrained color palette.",
    "Cinematic wide-angle composition, volumetric lighting.",
    "35mm documentary photograph, realistic imperfections and texture.",
)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic Phase 0 C2I/T2I regression manifests.")
    parser.add_argument("--output-dir", default="experiments/dualclock/regression")
    parser.add_argument("--base-seed", type=int, default=2025000)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    c2i_1000 = (
        {"index": index, "seed": args.base_seed + index, "class_id": index}
        for index in range(1000)
    )
    write_jsonl(output / "c2i_1000.jsonl", c2i_1000)
    write_jsonl(
        output / "c2i_smoke.jsonl",
        (
            {"index": index, "seed": args.base_seed + 10_000 + index, "class_id": class_id, "category": category}
            for index, (class_id, category) in enumerate(SMOKE_CLASSES)
        ),
    )

    t2i_rows = []
    for scene_index, scene in enumerate(SCENES):
        for style_index, style in enumerate(STYLES):
            index = scene_index * len(STYLES) + style_index
            t2i_rows.append(
                {
                    "index": index,
                    "seed": args.base_seed + 20_000,
                    "prompt": f"{scene}. {style}",
                }
            )
    write_jsonl(output / "t2i_100.jsonl", t2i_rows)
    with open(output / "t2i_prompts_100.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(row["prompt"] for row in t2i_rows) + "\n")
    print(f"Wrote regression manifests under {output}")


if __name__ == "__main__":
    main()
