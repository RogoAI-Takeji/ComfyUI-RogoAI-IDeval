import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


BACKENDS = ["facenet_vggface2", "insightface_buffalo_l"]


def _lazy_import_facenet():
    try:
        import torch
        from facenet_pytorch import InceptionResnetV1, MTCNN
    except Exception as exc:
        raise RuntimeError(
            "facenet-pytorch, torch, and their dependencies are required. "
            "Install this node's requirements into the ComfyUI Python environment."
        ) from exc
    return torch, MTCNN, InceptionResnetV1


def _lazy_import_insightface():
    try:
        import cv2
        from insightface.app import FaceAnalysis
    except Exception as exc:
        raise RuntimeError(
            "insightface, opencv-python, and onnxruntime are required for insightface_buffalo_l backend. "
            "Install insightface in the ComfyUI Python environment."
        ) from exc
    return cv2, FaceAnalysis


def _iter_images(folder: str, extensions: str = "png,jpg,jpeg,webp,bmp") -> List[Path]:
    root = Path(folder).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    allowed = {"." + ext.lower().lstrip(".") for ext in extensions.split(",") if ext.strip()}
    if not allowed:
        allowed = IMAGE_EXTENSIONS
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed]
    return sorted(files)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= 1e-12:
        return v
    return v / norm


def _to_numpy_image(image):
    arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[None, ...]
    raise ValueError(f"Unsupported image tensor shape: {arr.shape}")


def _pil_from_comfy_frame(frame: np.ndarray) -> Image.Image:
    frame = np.clip(frame, 0.0, 1.0)
    frame = (frame * 255.0).astype(np.uint8)
    return Image.fromarray(frame)


class FacenetEmbedder:
    backend_name = "facenet_vggface2"

    def __init__(self, device: str = "auto"):
        torch, MTCNN, InceptionResnetV1 = _lazy_import_facenet()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = device
        self.detector = MTCNN(image_size=160, margin=20, post_process=True, device=device)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def embed_pil(self, image: Image.Image) -> Tuple[Optional[np.ndarray], str]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        face = self.detector(image)
        if face is None:
            return None, "no_face"
        with self.torch.no_grad():
            emb = self.model(face.unsqueeze(0).to(self.device)).detach().cpu().numpy()[0]
        return _normalize(emb.astype(np.float32)), "ok"

    def embed_file(self, path: Path) -> Tuple[Optional[np.ndarray], str]:
        with Image.open(path) as img:
            return self.embed_pil(img)


class InsightFaceEmbedder:
    backend_name = "insightface_buffalo_l"

    def __init__(self, device: str = "auto"):
        cv2, FaceAnalysis = _lazy_import_insightface()
        self.cv2 = cv2
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        if device in ("auto", "cuda"):
            try:
                # Load the CUDA/cuDNN DLLs bundled with PyTorch before ONNX Runtime.
                import torch
                import onnxruntime

                if not torch.cuda.is_available():
                    raise RuntimeError("PyTorch CUDA is unavailable.")
                available = onnxruntime.get_available_providers()
                if "CUDAExecutionProvider" in available:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    ctx_id = 0
            except Exception:
                providers = ["CPUExecutionProvider"]
                ctx_id = -1
        self.requested_device = device
        self.providers = providers
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def embed_pil(self, image: Image.Image) -> Tuple[Optional[np.ndarray], str]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        rgb = np.asarray(image)
        bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        faces = self.app.get(bgr)
        if not faces:
            return None, "no_face"
        face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
        if emb is None:
            return None, "no_embedding"
        return _normalize(np.asarray(emb, dtype=np.float32)), "ok"

    def embed_file(self, path: Path) -> Tuple[Optional[np.ndarray], str]:
        with Image.open(path) as img:
            return self.embed_pil(img)


def _make_embedder(backend: str, device: str):
    if backend == "facenet_vggface2":
        return FacenetEmbedder(device)
    if backend == "insightface_buffalo_l":
        return InsightFaceEmbedder(device)
    raise ValueError(f"Unknown backend: {backend}")


def _score_folder(embedder, folder: str, prototype: np.ndarray, extensions: str) -> List[Dict]:
    rows = []
    for path in _iter_images(folder, extensions):
        emb, status = embedder.embed_file(path)
        score = _cosine(emb, prototype) if emb is not None else float("nan")
        rows.append({"file": str(path), "status": status, "csim": score})
    return rows


def _safe_mean(values: Sequence[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def _safe_std(values: Sequence[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return float(np.std(values)) if values else float("nan")


def _safe_percentile(values: Sequence[float], p: float) -> float:
    values = [v for v in values if not math.isnan(v)]
    return float(np.percentile(values, p)) if values else float("nan")


def _linear_slope(values: Sequence[float]) -> float:
    ys = np.asarray([v for v in values], dtype=np.float32)
    mask = ~np.isnan(ys)
    if int(mask.sum()) < 2:
        return float("nan")
    x = np.linspace(0.0, 1.0, len(ys), dtype=np.float32)[mask]
    y = ys[mask]
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _normalize_score(score: float, calibration: Optional[Dict]) -> float:
    if not calibration or math.isnan(score):
        return float("nan")
    genuine_mean = calibration.get("genuine_mean", float("nan"))
    impostor_mean = calibration.get("impostor_mean", float("nan"))
    denom = genuine_mean - impostor_mean
    if math.isnan(denom) or abs(denom) < 1e-6:
        return float("nan")
    return float((score - impostor_mean) / denom)


def _make_report_image(title: str, lines: List[str], path: str) -> str:
    width = 1280
    line_height = 34
    height = max(720, 110 + line_height * (len(lines) + 1))
    img = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 34)
        text_font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
    draw.text((42, 34), title, fill=(20, 24, 28), font=title_font)
    y = 96
    for line in lines:
        draw.text((42, y), line, fill=(35, 39, 44), font=text_font)
        y += line_height
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return str(out)


def _image_file_to_comfy_tensor(path: str):
    import torch

    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _make_calibration_bar(calibration: Dict, path: str) -> str:
    width, height = 1280, 520
    img = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 34)
        text_font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    draw.text((42, 32), "ID Calibration: face similarity ruler", fill=(20, 24, 28), font=title_font)
    left, right, y = 120, 1160, 230
    draw.line((left, y, right, y), fill=(35, 39, 44), width=8)
    draw.text((left, y + 28), "別人らしい", fill=(35, 39, 44), font=text_font)
    draw.text((520, y + 28), "判断が難しい", fill=(35, 39, 44), font=text_font)
    draw.text((1000, y + 28), "本人らしい", fill=(35, 39, 44), font=text_font)

    values = [
        ("impostor_mean", calibration.get("impostor_mean", float("nan")), (205, 72, 72)),
        ("stress_same_mean", calibration.get("stress_same_mean", float("nan")), (214, 145, 45)),
        ("genuine_mean", calibration.get("genuine_mean", float("nan")), (52, 133, 83)),
    ]
    clean = [v for _, v, _ in values if not math.isnan(v)]
    min_v = min(clean) if clean else 0.0
    max_v = max(clean) if clean else 1.0
    if abs(max_v - min_v) < 1e-6:
        min_v -= 0.1
        max_v += 0.1

    for label, value, color in values:
        if math.isnan(value):
            continue
        x = int(left + (value - min_v) / (max_v - min_v) * (right - left))
        draw.ellipse((x - 12, y - 44, x + 12, y - 20), fill=color)
        draw.line((x, y - 20, x, y + 8), fill=color, width=3)
        draw.text((max(40, x - 120), y - 84), f"{label}: {value:.4f}", fill=color, font=text_font)

    lines = [
        "CSIM is a relative proxy, not a certified identity score.",
        "First measure the same-person range and the different-person range.",
        "Use this ruler before comparing LTX, Wan, Krea, LoRA, or 10S outputs.",
    ]
    yy = 350
    for line in lines:
        draw.text((72, yy), line, fill=(35, 39, 44), font=text_font)
        yy += 34
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return str(out)


def _make_drift_graph(scores: Sequence[float], summary: Dict, path: str) -> str:
    width, height = 1280, 720
    img = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 34)
        text_font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
    draw.text((42, 30), "ID Drift Over Video Time", fill=(20, 24, 28), font=title_font)

    plot_left, plot_top, plot_right, plot_bottom = 100, 120, 1180, 560
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(80, 84, 89), width=2)
    draw.text((plot_left, plot_bottom + 32), "0% video", fill=(35, 39, 44), font=text_font)
    draw.text((plot_right - 110, plot_bottom + 32), "100% video", fill=(35, 39, 44), font=text_font)
    draw.text((28, plot_top - 8), "CSIM", fill=(35, 39, 44), font=text_font)

    clean = [s for s in scores if not math.isnan(s)]
    if clean:
        min_v = min(clean)
        max_v = max(clean)
        pad = max(0.02, (max_v - min_v) * 0.2)
        min_v -= pad
        max_v += pad
        points = []
        for idx, score in enumerate(scores):
            if math.isnan(score):
                continue
            x = plot_left + int((idx / max(1, len(scores) - 1)) * (plot_right - plot_left))
            y = plot_bottom - int((score - min_v) / (max_v - min_v) * (plot_bottom - plot_top))
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=(52, 101, 164), width=4)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(52, 101, 164))

    lines = [
        f"mean: {summary.get('mean_csim', float('nan')):.4f}",
        f"p05: {summary.get('p05_csim', float('nan')):.4f}",
        f"late_drop: {summary.get('late_drop', float('nan')):.4f}",
        f"drift_slope: {summary.get('drift_slope', float('nan')):.4f}",
        f"face_detect_rate: {summary.get('face_detect_rate', float('nan')):.2%}",
    ]
    yy = 596
    for line in lines:
        draw.text((100, yy), line, fill=(35, 39, 44), font=text_font)
        yy += 26
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return str(out)


def _average_aligned_faces(folder: str, device: str, extensions: str, image_size: int = 256) -> Tuple[Optional[np.ndarray], Dict]:
    torch, MTCNN, _ = _lazy_import_facenet()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = MTCNN(image_size=image_size, margin=20, post_process=False, device=device)
    faces = []
    rows = []
    for path in _iter_images(folder, extensions):
        with Image.open(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            face = detector(img)
        if face is None:
            rows.append({"file": str(path), "status": "no_face"})
            continue
        arr = face.detach().cpu().numpy()
        arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr, 0, 255).astype(np.float32)
        faces.append(arr)
        rows.append({"file": str(path), "status": "ok"})
    if not faces:
        return None, {"count_total": len(rows), "count_detected": 0, "rows": rows}
    avg = np.mean(np.stack(faces, axis=0), axis=0)
    avg = np.clip(avg, 0, 255).astype(np.uint8)
    return avg, {"count_total": len(rows), "count_detected": len(faces), "rows": rows}


def _save_average_face(avg: np.ndarray, path: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(avg).save(out)
    return str(out)


def _make_average_face_panel(
    candidates_avg: Optional[np.ndarray],
    candidates_meta: Dict,
    impostors_avg: Optional[np.ndarray],
    impostors_meta: Dict,
    output_png: str,
) -> str:
    width, height = 1280, 720
    img = Image.new("RGB", (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 34)
        label_font = ImageFont.truetype("arial.ttf", 28)
        text_font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        text_font = ImageFont.load_default()

    draw.text((42, 34), "Average aligned faces", fill=(20, 24, 28), font=title_font)
    panels = [
        ("candidates", candidates_avg, candidates_meta, 120),
        ("impostors", impostors_avg, impostors_meta, 720),
    ]
    for label, avg, meta, x in panels:
        draw.text((x, 100), label, fill=(35, 39, 44), font=label_font)
        box = (x, 150, x + 440, 590)
        draw.rectangle(box, outline=(80, 84, 89), width=2)
        if avg is None:
            draw.text((x + 70, 340), "No detectable faces", fill=(205, 72, 72), font=text_font)
        else:
            face_img = Image.fromarray(avg).resize((440, 440), Image.Resampling.LANCZOS)
            img.paste(face_img, (x, 150))
        count_text = f"detected {meta.get('count_detected', 0)} / {meta.get('count_total', 0)}"
        draw.text((x, 612), count_text, fill=(35, 39, 44), font=text_font)

    draw.text(
        (42, 666),
        "This is a visual explanation aid. The CSIM prototype is still the embedding average, not this bitmap average.",
        fill=(35, 39, 44),
        font=text_font,
    )
    out = Path(output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return str(out)


class RogoAIIDGallery:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gallery_dir": ("STRING", {"default": ""}),
                "output_json": ("STRING", {"default": "id_prototype.json"}),
                "backend": (BACKENDS, {"default": "facenet_vggface2"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "extensions": ("STRING", {"default": "png,jpg,jpeg,webp,bmp"}),
            }
        }

    RETURN_TYPES = ("ID_PROTOTYPE", "STRING", "ID_BACKEND")
    RETURN_NAMES = ("prototype", "report_text", "backend")
    FUNCTION = "build"
    CATEGORY = "RogoAI/IDeval"

    def build(self, gallery_dir, output_json, backend, device, extensions):
        embedder = _make_embedder(backend, device)
        embeddings = []
        rows = []
        for path in _iter_images(gallery_dir, extensions):
            emb, status = embedder.embed_file(path)
            if emb is not None:
                embeddings.append(emb)
            rows.append({"file": str(path), "status": status})
        if not embeddings:
            raise RuntimeError("No detectable faces were found in the gallery.")
        prototype_vec = _normalize(np.mean(np.stack(embeddings, axis=0), axis=0))
        prototype = {
            "backend": embedder.backend_name,
            "prototype": prototype_vec.tolist(),
            "gallery_dir": gallery_dir,
            "count_detected": len(embeddings),
            "count_total": len(rows),
            "rows": rows,
        }
        if output_json:
            out = Path(output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(prototype, ensure_ascii=False, indent=2), encoding="utf-8")
        text = f"Gallery prototype built: {len(embeddings)}/{len(rows)} faces detected. backend={embedder.backend_name}"
        return (prototype, text, embedder.backend_name)


class RogoAIIDCalibration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prototype": ("ID_PROTOTYPE",),
                "genuine_dir": ("STRING", {"default": ""}),
                "impostor_dir": ("STRING", {"default": ""}),
                "stress_same_dir": ("STRING", {"default": ""}),
                "output_json": ("STRING", {"default": "id_calibration.json"}),
                "report_png": ("STRING", {"default": "id_calibration_report.png"}),
                "backend": ("ID_BACKEND",),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "extensions": ("STRING", {"default": "png,jpg,jpeg,webp,bmp"}),
            }
        }

    RETURN_TYPES = ("ID_CALIBRATION", "IMAGE", "STRING", "ID_BACKEND")
    RETURN_NAMES = ("calibration", "report_image", "report_text", "backend")
    FUNCTION = "calibrate"
    CATEGORY = "RogoAI/IDeval"

    def calibrate(self, prototype, genuine_dir, impostor_dir, stress_same_dir, output_json, report_png, backend, device, extensions):
        prototype_backend = prototype.get("backend")
        if prototype_backend and prototype_backend != backend:
            raise ValueError(f"Prototype backend mismatch: prototype={prototype_backend}, requested={backend}")
        embedder = _make_embedder(backend, device)
        proto = np.asarray(prototype["prototype"], dtype=np.float32)
        genuine = _score_folder(embedder, genuine_dir, proto, extensions)
        impostor = _score_folder(embedder, impostor_dir, proto, extensions) if impostor_dir else []
        stress = _score_folder(embedder, stress_same_dir, proto, extensions) if stress_same_dir else []
        genuine_scores = [r["csim"] for r in genuine]
        impostor_scores = [r["csim"] for r in impostor]
        stress_scores = [r["csim"] for r in stress]
        calibration = {
            "backend": embedder.backend_name,
            "genuine_mean": _safe_mean(genuine_scores),
            "genuine_std": _safe_std(genuine_scores),
            "genuine_p05": _safe_percentile(genuine_scores, 5),
            "impostor_mean": _safe_mean(impostor_scores),
            "impostor_std": _safe_std(impostor_scores),
            "impostor_p95": _safe_percentile(impostor_scores, 95),
            "stress_same_mean": _safe_mean(stress_scores),
            "stress_same_p05": _safe_percentile(stress_scores, 5),
            "genuine": genuine,
            "impostor": impostor,
            "stress_same": stress,
        }
        if output_json:
            out = Path(output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
        text = (
            f"Calibration ({embedder.backend_name}): genuine_mean={calibration['genuine_mean']:.4f}, "
            f"impostor_mean={calibration['impostor_mean']:.4f}, "
            f"stress_same_mean={calibration['stress_same_mean']:.4f}"
        )
        report_path = _make_calibration_bar(calibration, report_png)
        return (calibration, _image_file_to_comfy_tensor(report_path), text, embedder.backend_name)


class RogoAIIDEvalImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prototype": ("ID_PROTOTYPE",),
                "calibration": ("ID_CALIBRATION",),
                "backend": ("ID_BACKEND",),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("csim", "normalized_csim", "report_text")
    FUNCTION = "score"
    CATEGORY = "RogoAI/IDeval"

    def score(self, image, prototype, calibration, backend, device):
        if prototype.get("backend") and prototype.get("backend") != backend:
            raise ValueError(f"Prototype backend mismatch: prototype={prototype.get('backend')}, requested={backend}")
        if calibration.get("backend") and calibration.get("backend") != backend:
            raise ValueError(f"Calibration backend mismatch: calibration={calibration.get('backend')}, requested={backend}")
        embedder = _make_embedder(backend, device)
        frames = _to_numpy_image(image)
        pil = _pil_from_comfy_frame(frames[0])
        emb, status = embedder.embed_pil(pil)
        if emb is None:
            return (float("nan"), float("nan"), "No face detected.")
        proto = np.asarray(prototype["prototype"], dtype=np.float32)
        csim = _cosine(emb, proto)
        norm = _normalize_score(csim, calibration)
        return (csim, norm, f"backend={embedder.backend_name}, CSIM={csim:.4f}, normalized={norm:.4f}, status={status}")


class RogoAIIDEvalVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "prototype": ("ID_PROTOTYPE",),
                "calibration": ("ID_CALIBRATION",),
                "output_json": ("STRING", {"default": "id_video_summary.json"}),
                "report_png": ("STRING", {"default": "id_video_drift_report.png"}),
                "backend": ("ID_BACKEND",),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("ID_SUMMARY", "IMAGE", "STRING")
    RETURN_NAMES = ("summary", "report_image", "report_text")
    FUNCTION = "score_video"
    CATEGORY = "RogoAI/IDeval"

    def score_video(self, frames, prototype, calibration, output_json, report_png, backend, device):
        if prototype.get("backend") and prototype.get("backend") != backend:
            raise ValueError(f"Prototype backend mismatch: prototype={prototype.get('backend')}, requested={backend}")
        if calibration.get("backend") and calibration.get("backend") != backend:
            raise ValueError(f"Calibration backend mismatch: calibration={calibration.get('backend')}, requested={backend}")
        embedder = _make_embedder(backend, device)
        proto = np.asarray(prototype["prototype"], dtype=np.float32)
        arr = _to_numpy_image(frames)
        rows = []
        scores = []
        normalized = []
        for idx, frame in enumerate(arr):
            emb, status = embedder.embed_pil(_pil_from_comfy_frame(frame))
            score = _cosine(emb, proto) if emb is not None else float("nan")
            norm = _normalize_score(score, calibration)
            rows.append({"frame": idx, "status": status, "csim": score, "normalized_csim": norm})
            scores.append(score)
            normalized.append(norm)
        detected = [s for s in scores if not math.isnan(s)]
        midpoint = max(1, len(scores) // 2)
        early = _safe_mean(scores[:midpoint])
        late = _safe_mean(scores[midpoint:])
        summary = {
            "backend": embedder.backend_name,
            "frame_count": len(scores),
            "detected_count": len(detected),
            "face_detect_rate": float(len(detected) / len(scores)) if scores else 0.0,
            "mean_csim": _safe_mean(scores),
            "p05_csim": _safe_percentile(scores, 5),
            "min_csim": float(np.nanmin(scores)) if detected else float("nan"),
            "drift_slope": _linear_slope(scores),
            "late_drop": float(early - late) if not math.isnan(early) and not math.isnan(late) else float("nan"),
            "mean_normalized_csim": _safe_mean(normalized),
            "per_frame": rows,
        }
        if output_json:
            out = Path(output_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        text = (
            f"backend={embedder.backend_name}, mean={summary['mean_csim']:.4f}, p05={summary['p05_csim']:.4f}, "
            f"late_drop={summary['late_drop']:.4f}, drift={summary['drift_slope']:.4f}, "
            f"detect={summary['face_detect_rate']:.2%}"
        )
        report_path = _make_drift_graph(scores, summary, report_png)
        return (summary, _image_file_to_comfy_tensor(report_path), text)


class RogoAIIDBatchReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "summary_json_dir": ("STRING", {"default": ""}),
                "report_png": ("STRING", {"default": "id_eval_report.png"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("report_image", "report_text")
    FUNCTION = "report"
    CATEGORY = "RogoAI/IDeval"

    def report(self, summary_json_dir, report_png):
        rows = []
        for path in sorted(Path(summary_json_dir).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if "mean_csim" not in data:
                continue
            rows.append((path.stem, data))
        rows.sort(key=lambda item: item[1].get("p05_csim", float("-inf")), reverse=True)
        lines = ["condition | mean | p05 | late_drop | drift | detect"]
        for name, data in rows:
            lines.append(
                f"{name} | {data.get('mean_csim', float('nan')):.4f} | "
                f"{data.get('p05_csim', float('nan')):.4f} | "
                f"{data.get('late_drop', float('nan')):.4f} | "
                f"{data.get('drift_slope', float('nan')):.4f} | "
                f"{data.get('face_detect_rate', float('nan')):.2%}"
            )
        _make_report_image("RogoAI ID Evaluation Report", lines, report_png)
        return (_image_file_to_comfy_tensor(report_png), "\n".join(lines) + f"\nReport PNG: {report_png}")


class RogoAIIDAverageFace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_dir": ("STRING", {"default": ""}),
                "output_png": ("STRING", {"default": "average_face.png"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "extensions": ("STRING", {"default": "png,jpg,jpeg,webp,bmp"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("average_face", "report_text")
    FUNCTION = "build"
    CATEGORY = "RogoAI/IDeval"

    def build(self, image_dir, output_png, device, extensions):
        avg, meta = _average_aligned_faces(image_dir, device, extensions)
        if avg is None:
            panel_path = _make_average_face_panel(None, meta, None, {"count_total": 0, "count_detected": 0}, output_png)
            return (_image_file_to_comfy_tensor(panel_path), f"No detectable faces in {image_dir}")
        out = _save_average_face(avg, output_png)
        text = f"Average face: detected {meta['count_detected']}/{meta['count_total']} faces. Output: {out}"
        return (_image_file_to_comfy_tensor(out), text)


class RogoAIIDAverageFaceCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "candidates_dir": ("STRING", {"default": ""}),
                "impostors_dir": ("STRING", {"default": ""}),
                "output_png": ("STRING", {"default": "average_face_compare.png"}),
                "candidates_avg_png": ("STRING", {"default": "candidates_average_face.png"}),
                "impostors_avg_png": ("STRING", {"default": "impostors_average_face.png"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "extensions": ("STRING", {"default": "png,jpg,jpeg,webp,bmp"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("comparison_image", "report_text")
    FUNCTION = "compare"
    CATEGORY = "RogoAI/IDeval"

    def compare(self, candidates_dir, impostors_dir, output_png, candidates_avg_png, impostors_avg_png, device, extensions):
        candidates_avg, candidates_meta = _average_aligned_faces(candidates_dir, device, extensions)
        impostors_avg, impostors_meta = _average_aligned_faces(impostors_dir, device, extensions) if impostors_dir else (
            None,
            {"count_total": 0, "count_detected": 0, "rows": []},
        )
        if candidates_avg is not None and candidates_avg_png:
            _save_average_face(candidates_avg, candidates_avg_png)
        if impostors_avg is not None and impostors_avg_png:
            _save_average_face(impostors_avg, impostors_avg_png)
        panel_path = _make_average_face_panel(candidates_avg, candidates_meta, impostors_avg, impostors_meta, output_png)
        text = (
            f"candidates detected {candidates_meta['count_detected']}/{candidates_meta['count_total']}; "
            f"impostors detected {impostors_meta['count_detected']}/{impostors_meta['count_total']}. "
            "Bitmap average faces are for visual explanation; use embedding CSIM for scoring."
        )
        return (_image_file_to_comfy_tensor(panel_path), text)


NODE_CLASS_MAPPINGS = {
    "RogoAIIDGallery": RogoAIIDGallery,
    "RogoAIIDCalibration": RogoAIIDCalibration,
    "RogoAIIDEvalImage": RogoAIIDEvalImage,
    "RogoAIIDEvalVideo": RogoAIIDEvalVideo,
    "RogoAIIDBatchReport": RogoAIIDBatchReport,
    "RogoAIIDAverageFace": RogoAIIDAverageFace,
    "RogoAIIDAverageFaceCompare": RogoAIIDAverageFaceCompare,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RogoAIIDGallery": "RogoAI ID Gallery",
    "RogoAIIDCalibration": "RogoAI ID Calibration",
    "RogoAIIDEvalImage": "RogoAI ID Eval Image",
    "RogoAIIDEvalVideo": "RogoAI ID Eval Video",
    "RogoAIIDBatchReport": "RogoAI ID Batch Report",
    "RogoAIIDAverageFace": "RogoAI ID Average Face",
    "RogoAIIDAverageFaceCompare": "RogoAI ID Average Face Compare",
}


