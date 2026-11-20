#!/usr/bin/env python3
"""
album_mood.py: early-results prototype for "album art -> mood-matched playlist".

Pipeline
  1. Load album art (file, folder, or upload/webcam via --ui)
  2. Extract a dominant color palette (k-means) and low-level visual features
  3. Score valence + energy two ways:
       a) color heuristic (brightness, saturation, warmth, colorfulness, contrast, edges)
       b) CLIP zero-shot (image vs. mood text prompts)
  4. Blend the two, map to a valence/energy quadrant (Q1..Q4)
  5. Emit Spotify-style target attributes + a figure for the demo

Usage
  python album_mood.py cover.jpg               # one image
  python album_mood.py covers/                 # every image in a folder
  python album_mood.py covers/ --no-clip       # color heuristic only (fast, no torch)
  python album_mood.py --eval labeled/         # accuracy on labeled/Q1..Q4 subfolders
  python album_mood.py --ui                    # drag-and-drop / webcam web demo

Install
  pip install numpy pillow scikit-learn matplotlib
  pip install torch transformers               # for CLIP
  pip install gradio                           # for --ui
"""
import argparse
import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

QUADRANTS = {
    "Q1": "Happy / Energetic / Joyful",
    "Q2": "Angry / Frantic / Anxious",
    "Q3": "Sad / Depressed",
    "Q4": "Calm / Relaxed / Content",
}


# ----------------------------------------------------------------------------
# Basics
# ----------------------------------------------------------------------------
def to_quadrant(valence, energy):
    """Map (valence, energy) in [0, 1] to a quadrant using the 0.5 split."""
    if energy >= 0.5:
        return "Q1" if valence >= 0.5 else "Q2"
    return "Q4" if valence >= 0.5 else "Q3"


def load_image(path_or_img, max_side=256):
    """Open an image, force RGB, and shrink it so feature extraction is fast."""
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    img = img.convert("RGB")  # returns a copy, so the caller's image is untouched
    img.thumbnail((max_side, max_side))
    return img


def rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(*[int(c) for c in color])


# ----------------------------------------------------------------------------
# Color features
# ----------------------------------------------------------------------------
def extract_palette(img, k=5):
    """Dominant colors via Pillow's median-cut quantizer. Returns colors (k x 3) and shares, largest first."""
    quantized = img.quantize(colors=k, method=Image.Quantize.MEDIANCUT)
    palette = np.array(quantized.getpalette()[: 3 * k]).reshape(-1, 3)
    counts = np.bincount(np.asarray(quantized).ravel(), minlength=len(palette))
    order = np.argsort(-counts)
    order = order[counts[order] > 0]  # drop unused palette slots
    return palette[order], counts[order] / counts.sum()


def color_features(img):
    """Low-level visual features, each roughly in [0, 1]."""
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    hue_deg, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]

    # Warmth: among clearly colored pixels, fraction with red/orange/yellow/pink hues.
    # Mostly gray images get a neutral 0.5 instead of being punished as "cold".
    colored = sat > 0.2
    warm = (hue_deg < 70) | (hue_deg > 300)
    if colored.mean() > 0.05:
        warmth = (warm & colored).sum() / colored.sum()
    else:
        warmth = 0.5

    # Colorfulness metric from Hasler & Suesstrunk (2003)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())

    # Contrast = spread of luminance
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    contrast = gray.std()

    # Edge density: share of pixels with a sharp brightness jump (busy / textured art)
    edge_x = np.abs(np.diff(gray, axis=1)) > 0.1
    edge_y = np.abs(np.diff(gray, axis=0)) > 0.1
    edge_density = 0.5 * (edge_x.mean() + edge_y.mean())

    return {
        "brightness": float(val.mean()),
        "saturation": float(sat.mean()),
        "warmth": float(warmth),
        "colorfulness": float(colorfulness),
        "contrast": float(contrast),
        "edge_density": float(edge_density),
    }


def color_mood(feats):
    """
    Hand-tuned color heuristic (baseline). Rationale from color-emotion research:
      valence goes up with brightness, saturation, warm hues
      energy  goes up with saturation, colorfulness, contrast, visual busyness
    The /hi values rescale features whose typical max is below 1.
    """
    def norm(x, hi):
        return float(np.clip(x / hi, 0, 1))

    valence = (0.45 * feats["brightness"]
               + 0.25 * feats["saturation"]
               + 0.30 * feats["warmth"])
    energy = (0.30 * feats["saturation"]
              + 0.25 * norm(feats["colorfulness"], 0.4)
              + 0.25 * norm(feats["contrast"], 0.3)
              + 0.20 * norm(feats["edge_density"], 0.25))
    return float(np.clip(valence, 0, 1)), float(np.clip(energy, 0, 1))


# ----------------------------------------------------------------------------
# CLIP zero-shot mood scoring
# ----------------------------------------------------------------------------
class ClipMoodScorer:
    """
    Embeds the image and compares it to mood text prompts.
    Each axis (valence, energy) = probability mass on its positive-side prompts.
    Also gives a direct 4-way quadrant zero-shot prediction.
    """

    PROMPTS = {
        "valence_pos": [
            "album cover art that feels happy and joyful",
            "a cheerful, uplifting, bright album cover",
            "a warm, hopeful, positive image",
        ],
        "valence_neg": [
            "album cover art that feels sad and melancholic",
            "a dark, gloomy, depressing album cover",
            "a bleak, lonely, negative image",
        ],
        "energy_pos": [
            "an intense, energetic, chaotic album cover",
            "loud, aggressive, high-energy artwork",
            "a dynamic, explosive, vibrant image",
        ],
        "energy_neg": [
            "a quiet, calm, minimal album cover",
            "soft, still, peaceful artwork",
            "a slow, sleepy, serene image",
        ],
    }

    QUADRANT_PROMPTS = {
        "Q1": ["a happy, energetic, joyful album cover", "a fun party album cover"],
        "Q2": ["an angry, frantic, anxious album cover", "an aggressive heavy metal album cover"],
        "Q3": ["a sad, depressed, lonely album cover", "a heartbroken melancholy album cover"],
        "Q4": ["a calm, relaxed, content album cover", "a peaceful ambient album cover"],
    }

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.scale = self.model.logit_scale.exp().item()  # CLIP's learned temperature (~100)

        # Embed every prompt once up front
        self.axis_emb = {name: self._embed_text(p) for name, p in self.PROMPTS.items()}
        # Quadrant prompts: average each quadrant's prompt embeddings (prompt ensembling)
        quad_embs = []
        for q in QUADRANTS:
            e = self._embed_text(self.QUADRANT_PROMPTS[q]).mean(dim=0)
            quad_embs.append(e / e.norm())
        self.quad_emb = torch.stack(quad_embs)

    def _as_tensor(self, out):
        # Newer transformers versions may return a model output object instead of a tensor
        return out if isinstance(out, self.torch.Tensor) else out.pooler_output

    def _embed_text(self, prompts):
        with self.torch.no_grad():
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
            emb = self._as_tensor(self.model.get_text_features(**inputs))
        return emb / emb.norm(dim=-1, keepdim=True)

    def _embed_image(self, img):
        with self.torch.no_grad():
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            emb = self._as_tensor(self.model.get_image_features(**inputs))
        return emb / emb.norm(dim=-1, keepdim=True)

    def _axis_score(self, img_emb, pos, neg):
        """Softmax over pos+neg prompts; return total probability on the positive side."""
        logits = self.scale * img_emb @ self.torch.cat([pos, neg]).T  # shape (1, n_prompts)
        probs = logits.softmax(dim=-1)[0]
        return float(probs[: len(pos)].sum())

    def score(self, img):
        img_emb = self._embed_image(img)
        valence = self._axis_score(img_emb, self.axis_emb["valence_pos"], self.axis_emb["valence_neg"])
        energy = self._axis_score(img_emb, self.axis_emb["energy_pos"], self.axis_emb["energy_neg"])
        quad_probs = (self.scale * img_emb @ self.quad_emb.T).softmax(dim=-1)[0].tolist()
        return valence, energy, dict(zip(QUADRANTS, quad_probs))


# ----------------------------------------------------------------------------
# Full analysis
# ----------------------------------------------------------------------------
def spotify_targets(valence, energy):
    """
    Target attributes for track search. Valence/energy come straight from the image;
    danceability/acousticness are rough guesses derived from them (tune or drop).
    """
    return {
        "target_valence": round(valence, 2),
        "target_energy": round(energy, 2),
        "target_danceability": round(0.6 * valence + 0.4 * energy, 2),
        "target_acousticness": round(1 - energy, 2),
    }


def analyze(img, clip_scorer=None, clip_weight=0.6, k=5):
    """Run every method on one image and return a flat result dict."""
    feats = color_features(img)
    color_v, color_e = color_mood(feats)
    colors, shares = extract_palette(img, k)

    result = {
        "color_valence": color_v,
        "color_energy": color_e,
        "color_quadrant": to_quadrant(color_v, color_e),
        "features": feats,
        "palette": [rgb_to_hex(c) for c in colors],
        "palette_shares": [round(float(s), 3) for s in shares],
    }

    if clip_scorer is not None:
        clip_v, clip_e, quad_probs = clip_scorer.score(img)
        result.update({
            "clip_valence": clip_v,
            "clip_energy": clip_e,
            "clip_quadrant": to_quadrant(clip_v, clip_e),
            "clip_zeroshot_quadrant": max(quad_probs, key=quad_probs.get),
            "clip_zeroshot_probs": {q: round(p, 3) for q, p in quad_probs.items()},
        })
        valence = clip_weight * clip_v + (1 - clip_weight) * color_v
        energy = clip_weight * clip_e + (1 - clip_weight) * color_e
    else:
        valence, energy = color_v, color_e

    quadrant = to_quadrant(valence, energy)
    result.update({
        "valence": round(valence, 3),
        "energy": round(energy, 3),
        "quadrant": quadrant,
        "mood": QUADRANTS[quadrant],
        "spotify_targets": spotify_targets(valence, energy),
    })
    return result


# ----------------------------------------------------------------------------
# Visualization (one figure per image, good for slides)
# ----------------------------------------------------------------------------
def make_figure(img, result, title=""):
    fig = plt.figure(figsize=(13, 4.2))

    # Panel 1: the album art
    ax_img = fig.add_subplot(1, 3, 1)
    ax_img.imshow(img)
    ax_img.set_title(title or "Album art", fontsize=11)
    ax_img.axis("off")

    # Panel 2: dominant palette as a proportional bar
    ax_pal = fig.add_subplot(1, 3, 2)
    left = 0.0
    for hex_color, share in zip(result["palette"], result["palette_shares"]):
        ax_pal.barh(0, share, left=left, color=hex_color, edgecolor="white")
        if share > 0.08:
            r, g, b = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
            text_color = "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "white"
            ax_pal.text(left + share / 2, 0, hex_color, ha="center", va="center",
                        fontsize=8, color=text_color, rotation=90)
        left += share
    f = result["features"]
    ax_pal.set_title(
        f"Palette   bright={f['brightness']:.2f}  sat={f['saturation']:.2f}  warm={f['warmth']:.2f}",
        fontsize=9,
    )
    ax_pal.set_xlim(0, 1)
    ax_pal.axis("off")

    # Panel 3: valence/energy plane with each method's point
    ax_ve = fig.add_subplot(1, 3, 3)
    ax_ve.axhline(0.5, color="gray", lw=0.8)
    ax_ve.axvline(0.5, color="gray", lw=0.8)
    corner_labels = {"Q1": (0.75, 0.95), "Q2": (0.25, 0.95), "Q3": (0.25, 0.05), "Q4": (0.75, 0.05)}
    for q, (x, y) in corner_labels.items():
        ax_ve.text(x, y, f"{q}: {QUADRANTS[q].split(' / ')[0]}", ha="center", va="center",
                   fontsize=8, color="gray")
    ax_ve.scatter(result["color_valence"], result["color_energy"], s=60, marker="o", label="color heuristic")
    if "clip_valence" in result:
        ax_ve.scatter(result["clip_valence"], result["clip_energy"], s=60, marker="^", label="CLIP")
    ax_ve.scatter(result["valence"], result["energy"], s=220, marker="*", color="black",
                  label=f"final -> {result['quadrant']}")
    ax_ve.set_xlim(0, 1)
    ax_ve.set_ylim(0, 1)
    ax_ve.set_xlabel("Valence (positivity)")
    ax_ve.set_ylabel("Energy (arousal)")
    ax_ve.set_title(f"{result['quadrant']}: {result['mood']}", fontsize=11)
    ax_ve.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    return fig


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ----------------------------------------------------------------------------
# Modes: batch, eval, UI
# ----------------------------------------------------------------------------
def collect_images(paths):
    images = []
    for p in map(Path, paths):
        if p.is_dir():
            images += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        elif p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
    return images


def run_batch(paths, clip_scorer, clip_weight, out_dir, k):
    images = collect_images(paths)
    if not images:
        print("No images found.")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in images:
        img = load_image(path)
        result = analyze(img, clip_scorer, clip_weight, k)
        fig = make_figure(img, result, title=path.stem)
        fig.savefig(out_dir / f"{path.stem}_mood.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"{path.name:35s} V={result['valence']:.2f} E={result['energy']:.2f} "
              f"-> {result['quadrant']} ({result['mood']})   palette={result['palette'][:3]}")
        rows.append({
            "file": path.name,
            "valence": result["valence"],
            "energy": result["energy"],
            "quadrant": result["quadrant"],
            "mood": result["mood"],
            "color_quadrant": result["color_quadrant"],
            "clip_quadrant": result.get("clip_quadrant", ""),
            "clip_zeroshot_quadrant": result.get("clip_zeroshot_quadrant", ""),
            "palette": " ".join(result["palette"]),
        })

    with open(out_dir / "results.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} figures + results.csv to {out_dir}/")


def run_eval(root, clip_scorer, clip_weight, k):
    """
    Expects root/Q1, root/Q2, root/Q3, root/Q4 folders of album covers.
    Ground truth idea: label each cover by its album's Spotify valence/energy
    (or by hand), then compare every method's quadrant prediction.
    """
    root = Path(root)
    methods = ["color_quadrant", "quadrant"]
    if clip_scorer is not None:
        methods = ["color_quadrant", "clip_quadrant", "clip_zeroshot_quadrant", "quadrant"]

    truths, preds = [], {m: [] for m in methods}
    for true_q in QUADRANTS:
        for path in collect_images([root / true_q]) if (root / true_q).is_dir() else []:
            result = analyze(load_image(path), clip_scorer, clip_weight, k)
            truths.append(true_q)
            for m in methods:
                preds[m].append(result[m])

    if not truths:
        print(f"No labeled images found under {root}/Q1..Q4")
        return

    print(f"Evaluated {len(truths)} images (chance = 25%)\n")
    names = {"color_quadrant": "color heuristic", "clip_quadrant": "CLIP axes",
             "clip_zeroshot_quadrant": "CLIP 4-way zero-shot", "quadrant": "final blend"}
    for m in methods:
        acc = np.mean([t == p for t, p in zip(truths, preds[m])])
        print(f"  {names[m]:22s} accuracy = {acc:.1%}")

    # Confusion matrix for the final blend (rows = true, cols = predicted)
    labels = list(QUADRANTS)
    print("\nConfusion matrix (final blend)   rows=true, cols=pred")
    print("      " + "  ".join(f"{q:>3s}" for q in labels))
    for t in labels:
        counts = [sum(1 for tt, pp in zip(truths, preds["quadrant"]) if tt == t and pp == p) for p in labels]
        print(f"  {t}  " + "  ".join(f"{c:>3d}" for c in counts))


def launch_ui(clip_scorer, clip_weight, k):
    import gradio as gr

    def predict(img):
        if img is None:
            return None, {}
        img = load_image(img)
        result = analyze(img, clip_scorer, clip_weight, k)
        summary = {key: result[key] for key in ("quadrant", "mood", "valence", "energy", "spotify_targets", "palette")}
        if "clip_zeroshot_probs" in result:
            summary["clip_zeroshot_probs"] = result["clip_zeroshot_probs"]
        return fig_to_pil(make_figure(img, result)), summary

    gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Album art (upload or webcam)"),
        outputs=[gr.Image(label="Analysis"), gr.JSON(label="Mood + Spotify targets")],
        title="Album Art -> Mood Quadrant",
        description="Color palette + CLIP mood scoring mapped to Spotify valence/energy.",
    ).launch()


def main():
    parser = argparse.ArgumentParser(description="Classify album art into a valence/energy mood quadrant.")
    parser.add_argument("paths", nargs="*", help="image files and/or folders")
    parser.add_argument("--no-clip", action="store_true", help="color heuristic only")
    parser.add_argument("--clip-weight", type=float, default=0.6, help="CLIP share of the final blend (0..1)")
    parser.add_argument("--k", type=int, default=5, help="palette size")
    parser.add_argument("--out", default="mood_results", help="output folder for figures + CSV")
    parser.add_argument("--eval", metavar="DIR", help="evaluate on DIR/Q1..Q4 labeled folders")
    parser.add_argument("--ui", action="store_true", help="launch the gradio upload/webcam demo")
    args = parser.parse_args()

    clip_scorer = None
    if not args.no_clip:
        print("Loading CLIP (first run downloads ~600MB)...")
        clip_scorer = ClipMoodScorer()

    if args.ui:
        launch_ui(clip_scorer, args.clip_weight, args.k)
    elif args.eval:
        run_eval(args.eval, clip_scorer, args.clip_weight, args.k)
    elif args.paths:
        run_batch(args.paths, clip_scorer, args.clip_weight, args.out, args.k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()