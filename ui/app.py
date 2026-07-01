"""
Gradio interactive review UI — three-tab workflow.

Tab 1 — Culling:   load folder → see keep/reject predictions, override, confirm
Tab 2 — Cropping:  see crop proposals with rule-of-thirds grid overlay, adjust with sliders
Tab 3 — Color:     compare param model vs U-Net side-by-side with judge scores, export

Run:
    python -m ui.app
    python ui/app.py
"""
import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEVELOP_SIZE, THUMB_SIZE
from data.raw_reader import develop_raw, extract_thumbnail
from inference.apply_params import apply_param_vec
from inference.pipeline import (
    apply_color_param, apply_color_unet, apply_crop, judge_score,
    predict_crop, predict_cull, _load_cull, _load_crop,
    _load_color_param, _load_color_unet, _load_judge,
)

# ── Global state (lazy-loaded models) ─────────────────────────────────────────
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_models: dict = {}

def _get(key, loader):
    if key not in _models:
        try:
            _models[key] = loader(_device)
        except FileNotFoundError as e:
            return None
    return _models[key]

def get_cull():   return _get("cull",   lambda d: _load_cull(d))
def get_crop():   return _get("crop",   _load_crop)
def get_param():  return _get("param",  _load_color_param)
def get_unet():   return _get("unet",   _load_color_unet)
def get_judge():  return _get("judge",  _load_judge)


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _draw_thirds(img: Image.Image, crop_box=None, angle_deg=0.0) -> Image.Image:
    """Draw rule-of-thirds grid and optional crop rectangle on a copy of img."""
    out  = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size

    # Rule-of-thirds lines
    for frac in (1/3, 2/3):
        draw.line([(int(w * frac), 0), (int(w * frac), h)], fill=(255, 255, 255, 120), width=2)
        draw.line([(0, int(h * frac)), (w, int(h * frac))], fill=(255, 255, 255, 120), width=2)

    # Crop rectangle
    if crop_box:
        x, y, cw, ch = crop_box
        draw.rectangle([x, y, x + cw, y + ch], outline=(0, 220, 100, 230), width=3)

    return out.convert("RGB")


# ── Tab 1: Culling ─────────────────────────────────────────────────────────────

# Session state stored as module-level dict keyed by session id (Gradio doesn't support
# true server-side sessions in all backends, so we use a simple global for single-user use)
_cull_state: dict = {"entries": [], "folder": ""}


def _load_folder_for_culling(folder_path: str):
    cull_model, threshold = get_cull() or (None, 0.5)
    if cull_model is None:
        return [], "No culling model found. Run models/culling/train.py first.", gr.update()

    raws = sorted(Path(folder_path).glob("*.CR3")) + sorted(Path(folder_path).glob("*.cr3"))
    if not raws:
        return [], f"No CR3 files found in {folder_path}", gr.update()

    entries = []
    for raw in raws:
        keep, conf = predict_cull(cull_model, threshold, raw, _device)
        thumb = extract_thumbnail(raw, size=(256, 256))
        entries.append({
            "raw":   str(raw),
            "thumb": thumb,
            "keep":  keep,
            "conf":  conf,
        })

    _cull_state["entries"] = entries
    _cull_state["folder"]  = folder_path

    kept    = sum(1 for e in entries if e["keep"])
    culled  = len(entries) - kept
    summary = f"Loaded {len(entries)} photos.  Model keeps: {kept}  |  Culls: {culled}"

    gallery = [(e["thumb"], f"{'✓ KEEP' if e['keep'] else '✗ CULL'}  {e['conf']:.0%}  {Path(e['raw']).name}")
               for e in entries]
    return gallery, summary, gr.update(visible=True)


def _confirm_culling():
    entries = _cull_state.get("entries", [])
    kept    = [e for e in entries if e["keep"]]
    _cull_state["kept_entries"] = kept
    return f"Confirmed {len(kept)} photos for cropping. Switch to the Cropping tab."


def _toggle_entry(evt: gr.SelectData):
    entries = _cull_state.get("entries", [])
    if 0 <= evt.index < len(entries):
        entries[evt.index]["keep"] = not entries[evt.index]["keep"]
    gallery = [(e["thumb"], f"{'✓ KEEP' if e['keep'] else '✗ CULL'}  {e['conf']:.0%}  {Path(e['raw']).name}")
               for e in entries]
    kept   = sum(1 for e in entries if e["keep"])
    culled = len(entries) - kept
    return gallery, f"{len(entries)} photos.  Keeping: {kept}  |  Culling: {culled}"


# ── Tab 2: Cropping ────────────────────────────────────────────────────────────

_crop_state: dict = {"entries": [], "idx": 0}


def _load_crops():
    kept = _cull_state.get("kept_entries", [])
    if not kept:
        return None, "No kept photos yet. Complete culling first.", 0, 0, 0, 0, 0.0

    crop_model = get_crop()
    entries = []
    for e in kept:
        box, angle = predict_crop(crop_model, e["raw"], _device) if crop_model else ([0, 0, 512, 512], 0.0)
        raw_img = develop_raw(e["raw"], size=DEVELOP_SIZE)
        entries.append({
            "raw":    e["raw"],
            "img":    raw_img,
            "box":    box,
            "angle":  angle,
        })
    _crop_state["entries"] = entries
    _crop_state["idx"]     = 0
    return _show_crop(0)


def _show_crop(idx: int):
    entries = _crop_state["entries"]
    if not entries:
        return None, "No images", 0, 0, 0, 0, 0.0
    idx  = max(0, min(idx, len(entries) - 1))
    _crop_state["idx"] = idx
    e   = entries[idx]
    img = _draw_thirds(e["img"], e["box"], e["angle"])
    label = f"Image {idx+1}/{len(entries)}  —  {Path(e['raw']).name}"
    x, y, w, h = e["box"]
    return img, label, x, y, w, h, e["angle"]


def _update_crop(x, y, w, h, angle):
    entries = _crop_state["entries"]
    idx     = _crop_state["idx"]
    if not entries:
        return None
    entries[idx]["box"]   = [int(x), int(y), int(w), int(h)]
    entries[idx]["angle"] = float(angle)
    img = _draw_thirds(entries[idx]["img"], entries[idx]["box"], entries[idx]["angle"])
    return img


def _confirm_crops():
    entries = _crop_state["entries"]
    _crop_state["confirmed"] = True
    return f"Confirmed crops for {len(entries)} photos. Switch to Color Correction tab."


# ── Tab 3: Color Correction ────────────────────────────────────────────────────

_color_state: dict = {"idx": 0}


def _load_color_tab():
    entries = _crop_state.get("entries", [])
    if not entries:
        return None, None, "Complete cropping first.", 0.0, 0.0
    return _show_color(0)


def _show_color(idx: int):
    entries = _crop_state["entries"]
    if not entries:
        return None, None, "No images", 0.0, 0.0
    idx = max(0, min(idx, len(entries) - 1))
    _color_state["idx"] = idx
    e   = entries[idx]
    cropped = apply_crop(e["img"], e["box"], e["angle"])

    param_model = get_param()
    unet_model  = get_unet()
    judge       = get_judge()

    if param_model:
        param_out, _ = apply_color_param(param_model, cropped, _device)
        p_score      = judge_score(judge, param_out, _device)
    else:
        param_out, p_score = cropped, float("nan")

    if unet_model:
        unet_out = apply_color_unet(unet_model, cropped, _device)
        u_score  = judge_score(judge, unet_out, _device)
    else:
        unet_out, u_score = cropped, float("nan")

    label = (f"Image {idx+1}/{len(entries)} — {Path(e['raw']).name}\n"
             f"Param model judge score: {p_score:.1%}  |  U-Net judge score: {u_score:.1%}")
    return param_out, unet_out, label, p_score, u_score


def _export_all(output_folder: str, use_unet: bool, quality: int):
    entries = _crop_state.get("entries", [])
    if not entries:
        return "No images to export."
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    model   = get_unet() if use_unet else get_param()
    judge   = get_judge()
    saved   = 0
    for e in entries:
        try:
            cropped = apply_crop(e["img"], e["box"], e["angle"])
            if use_unet and model:
                corrected = apply_color_unet(model, cropped, _device)
            elif model:
                corrected, _ = apply_color_param(model, cropped, _device)
            else:
                corrected = cropped
            out_path = out_dir / (Path(e["raw"]).stem + ".jpg")
            corrected.save(out_path, format="JPEG", quality=int(quality))
            saved += 1
        except Exception as exc:
            print(f"Export failed for {e['raw']}: {exc}")

    return f"Exported {saved} images to {out_dir}"


# ── Build UI ───────────────────────────────────────────────────────────────────

with gr.Blocks(title="Badminton Photo Editor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Jay Ma Photography — AI Editing Assistant")

    with gr.Tabs():

        # ── Tab 1: Culling ─────────────────────────────────────────────────
        with gr.Tab("1 · Culling"):
            gr.Markdown(
                "Load a folder of CR3 raws. The model keeps ~8–12% of photos "
                "(tuned to never miss Jay's picks). Click any thumbnail to toggle keep/cull."
            )
            with gr.Row():
                folder_input = gr.Textbox(label="Raw folder path", placeholder=r"C:\photos\event_raws")
                load_btn     = gr.Button("Load & Predict", variant="primary")
            cull_status   = gr.Textbox(label="Status", interactive=False)
            cull_gallery  = gr.Gallery(label="Photos (click to toggle)", columns=5, height=600)
            confirm_cull  = gr.Button("Confirm selections → proceed to Cropping", visible=False)
            cull_confirm_status = gr.Textbox(label="", interactive=False)

            load_btn.click(_load_folder_for_culling, folder_input,
                           [cull_gallery, cull_status, confirm_cull])
            cull_gallery.select(_toggle_entry, None, [cull_gallery, cull_status])
            confirm_cull.click(_confirm_culling, None, cull_confirm_status)

        # ── Tab 2: Cropping ────────────────────────────────────────────────
        with gr.Tab("2 · Cropping"):
            gr.Markdown(
                "Rule-of-thirds grid overlay. Sliders let you fine-tune the proposed crop and angle."
            )
            load_crops_btn = gr.Button("Load cropping predictions", variant="primary")
            crop_label     = gr.Textbox(label="", interactive=False)
            with gr.Row():
                prev_btn = gr.Button("← Prev")
                next_btn = gr.Button("Next →")
            crop_img = gr.Image(label="Crop preview", type="pil", height=600)
            with gr.Row():
                sl_x = gr.Slider(0, 2000, label="X", step=1)
                sl_y = gr.Slider(0, 2000, label="Y", step=1)
            with gr.Row():
                sl_w = gr.Slider(1, 2000, label="Width",  step=1)
                sl_h = gr.Slider(1, 2000, label="Height", step=1)
            sl_a = gr.Slider(-45, 45, label="Rotation (°)", step=0.5)
            confirm_crop_btn    = gr.Button("Confirm all crops → proceed to Color", variant="primary")
            crop_confirm_status = gr.Textbox(label="", interactive=False)

            _crop_outputs = [crop_img, crop_label, sl_x, sl_y, sl_w, sl_h, sl_a]
            load_crops_btn.click(_load_crops, None, _crop_outputs)
            prev_btn.click(lambda: _show_crop(_crop_state["idx"] - 1), None, _crop_outputs)
            next_btn.click(lambda: _show_crop(_crop_state["idx"] + 1), None, _crop_outputs)
            for sl in (sl_x, sl_y, sl_w, sl_h, sl_a):
                sl.change(_update_crop, [sl_x, sl_y, sl_w, sl_h, sl_a], crop_img)
            confirm_crop_btn.click(_confirm_crops, None, crop_confirm_status)

        # ── Tab 3: Color Correction ────────────────────────────────────────
        with gr.Tab("3 · Color Correction"):
            gr.Markdown(
                "Side-by-side comparison. Judge score = % confidence this looks like a finished edit."
            )
            load_color_btn = gr.Button("Load color predictions", variant="primary")
            color_label    = gr.Textbox(label="", interactive=False)
            with gr.Row():
                prev_color_btn = gr.Button("← Prev")
                next_color_btn = gr.Button("Next →")
            with gr.Row():
                param_img  = gr.Image(label="Param model", type="pil")
                unet_img   = gr.Image(label="U-Net",       type="pil")
            with gr.Row():
                param_score = gr.Number(label="Param judge score", precision=3)
                unet_score  = gr.Number(label="U-Net judge score", precision=3)

            gr.Markdown("### Export")
            with gr.Row():
                export_dir    = gr.Textbox(label="Output folder", placeholder=r"C:\photos\edited_output")
                use_unet_cb   = gr.Checkbox(label="Use U-Net (uncheck = param model)", value=False)
                quality_sl    = gr.Slider(70, 100, value=95, label="JPEG quality")
            export_btn    = gr.Button("Export all to JPEG", variant="primary")
            export_status = gr.Textbox(label="", interactive=False)

            _color_outputs = [param_img, unet_img, color_label, param_score, unet_score]
            load_color_btn.click(_load_color_tab, None, _color_outputs)
            prev_color_btn.click(lambda: _show_color(_color_state["idx"] - 1), None, _color_outputs)
            next_color_btn.click(lambda: _show_color(_color_state["idx"] + 1), None, _color_outputs)
            export_btn.click(_export_all, [export_dir, use_unet_cb, quality_sl], export_status)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
