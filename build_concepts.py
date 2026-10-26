import argparse
import re
import open_clip
import torch

import torch.nn.functional as F
import pandas as pd

from countex.utils import get_device
from pathlib import Path


CLIP_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# Paper: only the difference vector is normalised. True := also normalise both text embeddings first.
NORMALISE_TEXT_EMBEDDINGS = False
SOURCE_PROMPT = "A photo of object"
TARGET_TEMPLATES = {
    "color": "A photo of {} object",
    "texture": "A photo of {} object",
    "scene": "A photo of object on {}",
    "material": "A photo of object made of {}",
    "part": "A photo of object containing {}",
    "object": "A photo of object along with {}",
    "size": "A photo of {} object",
    "food": "A photo of object eating {}",
    "abstract": "A photo of {} object",
}

CUB_SOURCE_PROMPT = "A photo of a bird"
CUB_TARGET_TEMPLATE = "A photo of a bird with {} {}"

AWA2_SIZE = ("small", "big")
AWA2_FOOD = ("fish", "meat", "plankton", "vegetation", "insects")


def clean_name(name):
    """
    Turn a raw concept name into plain text

    :param name: raw name, e.g. "mountain_snowy-s" or "curved_(up_or_down)"
    :return: cleaned name, e.g. "mountain snowy" or "curved (up or down)"
    """

    name = re.sub(r"-[a-z]$", "", name)
    return name.replace("_", " ").strip()


def load_broden(label_csv):
    """
    Read C_{broden} from broden1_224/label.csv.

    :param label_csv: path to label.csv (columns: number, name, category, frequency, coverage, syns)
    :return: list of (concept, category); the category is the one with the largest count, e.g. "object(2391);part(64)" -> "object"
    """

    table = pd.read_csv(label_csv)
    concepts = []
    for name, category in zip(table["name"], table["category"]):
        if name == "-":
            continue
        counts = {c.split("(")[0]: int(c.split("(")[1].rstrip(")")) for c in category.split(";")}
        concepts.append((clean_name(name), max(counts, key = counts.get)))

    return concepts


def load_awa2(root, broden_concepts):
    """
    Read C_AwA2 from predicates.txt and assign every attribute a category.

    :param root: AwA2 root directory
    :param broden_concepts: output of load_broden, used to look up categories
    :return: list of (concept, category)
    """

    broden_category = dict(broden_concepts)
    with open(Path(root) / "predicates.txt") as f:
        names = [clean_name(line.split()[1]) for line in f if line.strip()]

    concepts = []
    for name in names:
        if name in AWA2_SIZE:
            category = "size"
        elif name in AWA2_FOOD:
            category = "food"
        else:
            category = broden_category.get(name, "abstract")
        concepts.append((name, category))

    return concepts


def load_cub(root):
    """
    Read C_CUB from attributes/attributes.txt, e.g. "has_bill_shape::curved_(up_or_down)".

    :param root: CUB_200_2011 root directory
    :return: list of (attribute type, attribute value), e.g. ("bill shape", "curved (up or down)")
    """

    concepts = []
    with open(Path(root) / "attributes" / "attributes.txt") as f:
        for line in f:
            if not line.strip():
                continue
            attribute_type, value = line.split()[1].split("::")
            concepts.append((clean_name(attribute_type.removeprefix("has_")), clean_name(value)))

    return concepts


def main(args):
    device = get_device()

    if args.library == "cub":
        cub = load_cub(args.data_root)
        names = [f"{attribute_type}: {value}" for attribute_type, value in cub]
        categories = [attribute_type for attribute_type, _ in cub]

        source_prompts = [CUB_SOURCE_PROMPT] * len(cub)
        target_prompts = [CUB_TARGET_TEMPLATE.format(value, attribute_type) for attribute_type, value in cub]
    else:
        broden = load_broden(Path(args.broden_root) / "label.csv")
        concepts = broden if args.library == "broden" else load_awa2(args.data_root, broden)
        names = [name for name, _ in concepts]
        categories = [category for _, category in concepts]

        source_prompts = [SOURCE_PROMPT] * len(concepts)
        target_prompts = [TARGET_TEMPLATES[category].format(name) for name, category in concepts]


    # v_c = CLIP_text(t_trg) - CLIP_text(t_src), normalised to a unit vector
    model = open_clip.create_model(CLIP_NAME, pretrained = CLIP_PRETRAINED).to(device).eval()
    tokenizer = open_clip.get_tokenizer(CLIP_NAME)
    with torch.no_grad():
        source = model.encode_text(tokenizer(source_prompts).to(device), normalize = NORMALISE_TEXT_EMBEDDINGS)
        target = model.encode_text(tokenizer(target_prompts).to(device), normalize = NORMALISE_TEXT_EMBEDDINGS)
    directions = F.normalize(target - source, dim = -1).cpu()   # V_CLIP: N x 512

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    torch.save({
        "library": args.library,
        "clip": f"{CLIP_NAME}/{CLIP_PRETRAINED}",
        "normalise_text_embeddings": NORMALISE_TEXT_EMBEDDINGS,
        "concepts": names,
        "categories": categories,
        "source_prompts": source_prompts,
        "target_prompts": target_prompts,
        "directions": directions,
    }, out_dir / f"{args.library}.pt")

    with open(out_dir / f"{args.library}_prompts.txt", "w") as f:
        for name, category, target_prompt in zip(names, categories, target_prompts):
            f.write(f"{name}\t{category}\t{target_prompt}\n")

    print(f"{args.library}: {len(names)} concepts, V_CLIP {tuple(directions.shape)} -> {out_dir / f'{args.library}.pt'}")

    print(" === === === === === ===\n")
    print("Categories: ", pd.Series(categories).value_counts().to_dict())
    print(" === === === === === ===\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required = True, choices = ("awa2", "cub", "broden"))
    parser.add_argument("--data-root", default = None)
    parser.add_argument("--broden-root", default = None)
    parser.add_argument("--output-dir", default = "concepts")

    main(parser.parse_args())
