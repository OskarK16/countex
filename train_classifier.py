import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from countex.data import build_datasets
from countex.models import MODEL_NAMES, build_classifier
from countex.utils import get_device, set_seed


EPOCHS = 10
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 2026
AWA2_TEST_RATIO = 0.2
CLIP_NORMALIZE = True   # L2-normalise CLIP image embeddings before the linear head
NUM_CLASSES = {"awa2": 50, "cub": 200}
RESULTS_FILE = "results/accuracies.txt"
DISPLAY_NAMES = {"clip_linear": "CLIP+linear", "resnet18": "ResNet18", "resnet50": "ResNet50",
                 "awa2": "AwA2", "cub": "CUB"}

LINEAR_LR = 1e-3
FINETUNE_HEAD_LR = 1e-2
FINETUNE_BACKBONE_LR = 1e-3
FINETUNE_MOMENTUM = 0.9
FINETUNE_WEIGHT_DECAY = 1e-4


def training_mode(model_name, dataset):
    """Appendix A.1: ResNets are fine-tuned on CUB; everything else is a linear probe."""
    return "finetune" if (dataset == "cub" and model_name != "clip_linear") else "linear"


def build_optimizer(model, mode):
    if mode == "linear":
        return torch.optim.Adam(model.fc.parameters(), lr = LINEAR_LR)

    backbone = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    groups = [
        {"params": model.fc.parameters(), "lr": FINETUNE_HEAD_LR},
        {"params": backbone, "lr": FINETUNE_BACKBONE_LR},
    ]

    return torch.optim.SGD(groups, momentum = FINETUNE_MOMENTUM, weight_decay = FINETUNE_WEIGHT_DECAY)


def run_epoch(model, loader, device, criterion, mode, optimizer = None, desc = ""):
    training = optimizer is not None
    model.train(training and mode == "finetune")
    total_loss, correct, count = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc = desc, leave = False):
            images = images.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)

    return total_loss / count, correct / count


def main(args):
    set_seed(SEED)
    device = get_device()
    mode = training_mode(args.model, args.dataset)
    out_dir = Path(args.output_dir or f"checkpoints/{args.model}_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  model={args.model}  dataset={args.dataset}  mode={mode}")

    model, train_tf, eval_tf = build_classifier(
        args.model, NUM_CLASSES[args.dataset], clip_normalise=CLIP_NORMALIZE)
    if mode == "linear":
        model.requires_grad_(False)
        model.fc.requires_grad_(True)
    model.to(device)

    train_set, test_set, class_names = build_datasets(args.dataset, args.data_root, train_tf, eval_tf, test_ratio = AWA2_TEST_RATIO, seed = SEED)
    assert len(class_names) == NUM_CLASSES[args.dataset], f"found {len(class_names)} classes"
    print(f"classes={len(class_names)}  train={len(train_set)}  test={len(test_set)}")

    loader_kwargs = dict(batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory = device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle = True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle = False, **loader_kwargs)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, mode)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS)

    history = []
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, mode, optimizer, desc = f"epoch {epoch} train")
        test_loss, test_acc = run_epoch(model, test_loader, device, criterion, mode, desc = f"epoch {epoch} test")
        scheduler.step()
        history.append(dict(epoch = epoch, train_loss = train_loss, train_acc = train_acc, test_loss = test_loss, test_acc = test_acc))
        print(f"epoch {epoch:3d}  train loss {train_loss:.4f} acc {train_acc:.4f}  "
              f"test loss {test_loss:.4f} acc {test_acc:.4f}")

    checkpoint = {
        "model_name": args.model,
        "dataset": args.dataset,
        "mode": mode,
        "clip_normalize": CLIP_NORMALIZE,
        "class_names": class_names,
        "fc": model.fc.state_dict(),
        "model": model.state_dict() if mode == "finetune" else None,
        "test_acc": history[-1]["test_acc"],
    }

    torch.save(checkpoint, out_dir / "classifier.pt")

    split = {
        "train": [[str(p), y] for p, y in train_set.samples],
        "test": [[str(p), y] for p, y in test_set.samples],
    }

    with open(out_dir / "split.json", "w") as f:
        json.dump(split, f)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"saved to {out_dir}  (final test acc {history[-1]['test_acc']:.4f})")

    results_file = Path(RESULTS_FILE)
    results_file.parent.mkdir(parents = True, exist_ok = True)
    with open(results_file, "a") as f:
        f.write(f"{DISPLAY_NAMES[args.model]} {DISPLAY_NAMES[args.dataset]}: {100 * history[-1]['test_acc']:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("awa2", "cub"))
    parser.add_argument("--data-root", required=True, help="Path to Animals_with_Attributes2/ or CUB_200_2011/")
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--output-dir", default=None, help="Default: checkpoints/<model>_<dataset>")

    main(parser.parse_args())
