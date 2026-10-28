"""Train / fine-tune one experiment.  Example:
    python -m src.train --name E3_tl3_basic --model maskrcnn_r50 --trainable-layers 3 --aug basic
Outputs -> results/<name>/{best.pt, history.csv, metrics.json, curves.png}
Model selection uses the VALIDATION set; the TEST set is evaluated once, with the best checkpoint.
"""
import argparse
import csv
import time
from pathlib import Path

import torch

from src.data import collate_fn, download_dataset, get_datasets
from src.engine import evaluate, train_one_epoch
from src.models import MODEL_NAMES, build_model
from src.utils import count_params, file_size_mb, get_device, save_json, seed_everything


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="maskrcnn_r50", choices=MODEL_NAMES)
    p.add_argument("--no-pretrained", action="store_true", help="random init (ablation)")
    p.add_argument("--trainable-layers", type=int, default=3)
    p.add_argument("--aug", default="none", choices=["none", "basic", "strong"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--optim", default="sgd", choices=["sgd", "adamw"])
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--min-size", type=int, default=None, help="train/inference short-side resize")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args(argv)


def plot_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ep, [h["train_loss"] for h in history])
    ax[0].set(title="Train loss", xlabel="epoch")
    ax[1].plot(ep, [h["val_segm_map"] for h in history], label="segm mAP")
    ax[1].plot(ep, [h["val_bbox_map"] for h in history], label="bbox mAP")
    ax[1].set(title="Validation mAP@[.5:.95]", xlabel="epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    args = get_args(argv)
    seed_everything(args.seed)
    device = get_device()
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    root = download_dataset(args.data_root)
    train_ds, val_ds, test_ds = get_datasets(root, args.aug)
    print(f"[{args.name}] device={device} train/val/test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.workers, collate_fn=collate_fn)

    cfg = dict(model=args.model, num_classes=2, min_size=args.min_size, max_size=None)
    model = build_model(args.model, 2, not args.no_pretrained, args.trainable_layers, args.min_size).to(device)
    total, trainable = count_params(model)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "sgd":
        opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda" and not args.no_amp))

    history, best, best_epoch = [], -1.0, -1
    t_start = time.time()
    for epoch in range(args.epochs):
        t0 = time.time()
        tr = train_one_epoch(model, opt, loader, device, scaler, epoch)
        sched.step()
        val = evaluate(model, val_ds, device)
        row = {"epoch": epoch, "train_loss": tr["loss"], "val_segm_map": val["segm_map"],
               "val_bbox_map": val["bbox_map"], "val_segm_map50": val["segm_map_50"],
               "lr": opt.param_groups[0]["lr"], "epoch_time_s": time.time() - t0}
        history.append(row)
        print(f"  ep {epoch:02d} loss {tr['loss']:.4f} | val segm mAP {val['segm_map']:.4f} "
              f"bbox mAP {val['bbox_map']:.4f} | {row['epoch_time_s']:.0f}s")
        if val["segm_map"] > best:
            best, best_epoch = val["segm_map"], epoch
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch}, out_dir / "best.pt")
    train_time = time.time() - t_start

    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    plot_curves(history, out_dir / "curves.png")

    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    val = evaluate(model, val_ds, device)
    test = evaluate(model, test_ds, device)
    save_json({"name": args.name, "args": vars(args), "best_epoch": best_epoch, "val": val, "test": test,
               "train_time_s": train_time, "params_total": total, "params_trainable": trainable,
               "checkpoint_mb": file_size_mb(out_dir / "best.pt"), "device": str(device)},
              str(out_dir / "metrics.json"))
    print(f"[{args.name}] best epoch {best_epoch} | VAL segm mAP {val['segm_map']:.4f} | "
          f"TEST segm mAP {test['segm_map']:.4f} bbox mAP {test['bbox_map']:.4f}")


if __name__ == "__main__":
    main()
