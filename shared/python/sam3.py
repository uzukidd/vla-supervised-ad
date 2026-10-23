"""Shared SAM3 prediction, class prompts, palette and image helpers."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import inspect
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

@dataclass(frozen=True)
class Category:
    id: int
    name: str
    prompts: tuple[str, ...]


def parse_class_file(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    entries = []
    seen = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" in line:
            name, text = (part.strip() for part in line.split("=", 1))
        else:
            name = text = line
        prompts = tuple(part.strip() for part in text.split("|"))
        if not name or not all(prompts):
            raise ValueError(f"{path}:{line_number}: empty class name or prompt")
        if name.casefold() in seen:
            raise ValueError(f"{path}:{line_number}: duplicate class {name!r}")
        if name.casefold() in {"unlabeled", "unassigned"}:
            raise ValueError(f"{path}:{line_number}: class 0 is reserved for unassigned pixels")
        seen.add(name.casefold())
        entries.append((name, tuple(dict.fromkeys(prompts))))
    if not entries:
        raise ValueError(f"No classes found in {path}")
    if len(entries) > 254:
        raise ValueError("At most 254 classes are supported; IDs 0 and 255 are reserved.")
    return entries


def merge_instance(prediction: np.ndarray, best_score: np.ndarray, mask: np.ndarray, score: float, class_id: int) -> None:
    """Highest instance confidence wins; equal scores preserve the earlier class."""
    if mask.shape != prediction.shape or best_score.shape != prediction.shape:
        raise ValueError("Mask and semantic maps must have matching dimensions")
    update = mask.astype(bool) & (score > best_score)
    prediction[update] = class_id
    best_score[update] = score


def _load_local_sam3_checkpoint(model, checkpoint: Path, torch) -> None:
    """Map a local ZIP checkpoint and assign its storages without a CPU copy."""
    if not torch.load.__module__.startswith('torch') or 'mmap' not in inspect.signature(torch.load).parameters:
        raise RuntimeError('此PyTorch版本不支持SAM3检查点内存映射加载 / PyTorch mmap checkpoint loading is unavailable')
    checkpoint_state=torch.load(str(checkpoint),map_location='cpu',weights_only=True,mmap=True)
    if isinstance(checkpoint_state.get('model'),dict):checkpoint_state=checkpoint_state['model']
    state={key.replace('detector.',''):value for key,value in checkpoint_state.items() if 'detector' in key}
    if model.inst_interactive_predictor is not None:
        state.update({key.replace('tracker.','inst_interactive_predictor.model.'):value
                      for key,value in checkpoint_state.items() if 'tracker' in key})
    if 'assign' not in inspect.signature(model.load_state_dict).parameters:
        raise RuntimeError('此PyTorch版本不支持零拷贝权重分配 / PyTorch assign checkpoint loading is unavailable')
    missing,unexpected=model.load_state_dict(state,strict=False,assign=True)
    if missing:print(f'loaded {checkpoint} with missing keys: {missing=}')
    if unexpected:print(f'loaded {checkpoint} with unexpected keys: {unexpected=}')


class Sam3Segmenter:
    def __init__(self, checkpoint: Path, device: str, precision: str, score_threshold: float, mask_threshold: float):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Local official SAM3 checkpoint not found: {checkpoint}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in vla-supervised; check WSL GPU access or use --device cpu --precision fp32.")
        if device == "cpu" and precision != "fp32":
            raise ValueError("CPU inference requires --precision fp32")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("GPU does not support bfloat16; use --precision fp32 or fp16")
        self.torch = torch
        self.device = device
        self.precision = precision
        self.mask_threshold = mask_threshold
        model_builder=build_sam3_image_model
        model=model_builder(checkpoint_path=None,load_from_HF=False,device='cpu',eval_mode=True,compile=False)
        _load_local_sam3_checkpoint(model,checkpoint,torch)
        # Move assigned checkpoint storages to the target device after loading.
        if device=='cuda':model=model.cuda()
        model.eval()
        self.processor=Sam3Processor(model,device=device,confidence_threshold=score_threshold)

    def predict(self, image: Image.Image, categories: list[Category], *, instance_callback=None) -> tuple[np.ndarray, list[dict]]:
        """Predict semantic labels; optionally stream each original instance mask.

        The callback receives (mask, metadata), with one-based camera-local
        instance_id in metadata. Masks precede semantic overlap fusion; callbacks
        must copy arrays they need to retain and must not mutate the mask.
        Existing callers still receive the same semantic/instance metadata pair.
        """
        torch = self.torch
        context = nullcontext() if self.precision == "fp32" else torch.autocast(
            device_type="cuda", dtype=torch.bfloat16 if self.precision == "bf16" else torch.float16
        )
        prediction = np.zeros((image.height, image.width), dtype=np.uint8)
        best_score = np.full(prediction.shape, -np.inf, dtype=np.float32)
        instances = []
        with torch.inference_mode(), context:
            state = self.processor.set_image(image)
            for category in categories:
                for prompt in category.prompts:
                    self.processor.reset_all_prompts(state)
                    result = self.processor.set_text_prompt(prompt=prompt, state=state)
                    scores = result["scores"].detach().float().cpu().numpy().reshape(-1)
                    boxes = result["boxes"].detach().float().cpu().numpy()
                    probabilities = result["masks_logits"]  # Official API: sigmoid probabilities, despite the name.
                    for instance_index, score in enumerate(scores):
                        mask = (probabilities[instance_index, 0] > self.mask_threshold).cpu().numpy()
                        if not mask.any():
                            continue
                        merge_instance(prediction, best_score, mask, float(score), category.id)
                        instances.append({
                            "class_id": category.id, "class_name": category.name, "prompt": prompt,
                            "score": float(score), "box_xyxy": boxes[instance_index].tolist(),
                            "mask_pixels_before_fusion": int(mask.sum()),
                        })
                        if instance_callback is not None:
                            instance_callback(mask, {"instance_id": len(instances), **instances[-1]})
        return prediction, instances


def make_palette() -> np.ndarray:
    palette = np.zeros((256, 3), dtype=np.uint8)
    for label in range(1, 256):
        value = label
        for bit in range(8):
            for channel in range(3):
                palette[label, channel] |= ((value >> channel) & 1) << (7 - bit)
            value >>= 3
    palette[255] = (100, 100, 100)
    return palette


def colorize(mask: np.ndarray, palette: np.ndarray) -> Image.Image:
    if mask.size and (mask.min() < 0 or mask.max() > 255):
        raise ValueError("Visualization label IDs must be in 0..255")
    return Image.fromarray(palette[mask.astype(np.uint8)])


def overlay(image: Image.Image, mask: np.ndarray, palette: np.ndarray, alpha: float) -> Image.Image:
    pixels = np.asarray(image).astype(np.float32)
    blended = pixels * (1 - alpha) + np.asarray(colorize(mask, palette)) * alpha
    active = (mask != 0) & (mask != 255)
    pixels[active] = blended[active]
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=(20, 20, 20)) -> None:
    # Pillow's bundled font supports English; common WSL fonts also handle user class names.
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=fill, font=font)
