"""Fine-grained: alpha * fresh_ep2 + (1-alpha) * soup05_ep1"""
import os, sys, csv, torch, torch.nn.functional as F, numpy as np, open_clip
sys.path.insert(0, ".")
from lpcvc2026.modules.data import BASE_DIR, CompetitionTransform
from lpcvc2026.modules.evaluate import evaluate_on_sample
from PIL import Image

ARCH, PRETRAINED = "MobileCLIP-B", "datacompdr"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_DIR = os.path.join(BASE_DIR, "track1_test_eval_v2", "frozen_v1", "calibration")
A_CKPT = "models/mobileclipB_04_vg_fresh/epoch_2.pt"
B_CKPT = "models/mobileclipB_05_vg_from_soup/epoch_1.pt"

def load_sd(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return ck["state_dict"] if "state_dict" in ck else ck

def test_r10(model, tok):
    tm = {}; gt = {}
    with open(os.path.join(EVAL_DIR, "test_texts.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f): tm[int(r["Text_nums"])] = r["Unique_Texts"]
    with open(os.path.join(EVAL_DIR, "test_image_to_texts.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f): gt[r["Image_names"]] = [int(x) for x in r["Text_nums"].split(";")]
    imgs = sorted(gt.keys()); tids = sorted(tm.keys()); txts = [tm[t] for t in tids]
    tf = CompetitionTransform(224)
    with torch.no_grad():
        te = F.normalize(model.encode_text(tok(txts).to(DEVICE)).float(), dim=-1)
        ie = torch.cat([F.normalize(model.encode_image(
            tf(Image.open(os.path.join(EVAL_DIR, "images", n)).convert("RGB")).unsqueeze(0).to(DEVICE)
        ).float(), dim=-1) for n in imgs])
    s = ie @ te.t()
    return float(np.mean([
        len(set(gt[n]) & {tids[s[i].argsort(descending=True)[j].item()] for j in range(10)}) / len(gt[n])
        for i, n in enumerate(imgs)
    ]) * 100)

print("Loading..."); a = load_sd(A_CKPT); b = load_sd(B_CKPT)
model, _, _ = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED)
tok = open_clip.get_tokenizer(ARCH); model = model.to(DEVICE)

ALPHAS = [0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55, 0.60]

print(f"\n{'alpha(fresh2)':>14} {'Test R@10':>10} {'Samp R@10':>10} {'LB_proxy':>10}")
print("-" * 48)

best_lb = 0; best_alpha = 0; best_sd = None
for alpha in ALPHAS:
    soup = {k: alpha * a[k] + (1 - alpha) * b[k] for k in a}
    model.load_state_dict(soup); model.eval()
    tr = test_r10(model, tok)
    _, _, sr = evaluate_on_sample(model, tok, device=DEVICE)
    lb = 0.6 * tr + 0.4 * sr
    tag = " <--" if lb > best_lb else ""
    if lb > best_lb:
        best_lb, best_alpha, best_sd = lb, alpha, {k: v.clone() for k, v in soup.items()}
    print(f"  {alpha:>12.2f} {tr:>10.1f} {sr:>10.1f} {lb:>10.2f}{tag}")

print(f"\nBest alpha={best_alpha:.2f}: LB={best_lb:.2f}")
out = f"models/soup_fresh2_x_s05e1_a{int(best_alpha*100):03d}.pt"
torch.save({"state_dict": best_sd, "arch": ARCH, "pretrained": PRETRAINED,
            "recipe": f"fresh_ep2*{best_alpha:.2f}+soup05_ep1*{1-best_alpha:.2f}",
            "lb_proxy": best_lb}, out)
print(f"Saved -> {out}")
