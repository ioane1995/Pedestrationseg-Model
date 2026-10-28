#!/usr/bin/env bash
# Ablation grid. Everything shares the same fixed splits, epochs and seed -> differences are due to the changed factor only.
# Skips experiments whose metrics.json already exists, so an interrupted Colab session can simply be re-run.
set -euo pipefail
EP=${EPOCHS:-15}
run() { name=$1; shift; [ -f "results/$name/metrics.json" ] && { echo "skip $name"; return; }; python -m src.train --name "$name" --epochs "$EP" "$@"; }

run E0_scratch_r50        --model maskrcnn_r50 --no-pretrained --aug basic          # value of transfer learning
run E1_frozen_backbone    --model maskrcnn_r50 --trainable-layers 0 --aug none      # heads only
run E2_tl3_noaug          --model maskrcnn_r50 --trainable-layers 3 --aug none      # standard fine-tune baseline
run E3_tl3_basic          --model maskrcnn_r50 --trainable-layers 3 --aug basic     # + flip/colour jitter
run E4_tl3_strong         --model maskrcnn_r50 --trainable-layers 3 --aug strong    # + heavy augmentation
run E5_tl5_basic          --model maskrcnn_r50 --trainable-layers 5 --aug basic     # unfreeze whole backbone
run E6_lr0.0025_basic     --model maskrcnn_r50 --trainable-layers 3 --aug basic --lr 0.0025   # LR sensitivity
run E7_adamw_basic        --model maskrcnn_r50 --trainable-layers 3 --aug basic --optim adamw --lr 0.0001 --weight-decay 0.01
run E8_mobilenet_basic    --model maskrcnn_mobilenet --trainable-layers 6 --aug basic          # light architecture
