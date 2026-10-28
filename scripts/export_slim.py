"""Save an fp16 copy of a checkpoint (~half size; GitHub rejects files > 100 MB). Predictor casts back to fp32."""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default="models/best_fp16.pt")
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
ck["state_dict"] = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()}
torch.save(ck, a.out)
print("saved", a.out)
