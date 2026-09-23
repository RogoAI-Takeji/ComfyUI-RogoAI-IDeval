import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .nodes import (
    BACKENDS,
    _average_aligned_faces,
    _cosine,
    _image_file_to_comfy_tensor,
    _iter_images,
    _linear_slope,
    _make_drift_graph,
    _make_embedder,
    _normalize,
    _normalize_score,
    _pil_from_comfy_frame,
    _safe_mean,
    _safe_percentile,
    _safe_std,
    _score_folder,
    _to_numpy_image,
)


CATEGORY = "RogoAI/IDeval"
IMAGE_EXTENSIONS = "png,jpg,jpeg,webp,bmp"


def _comfy_input_path(*parts: str) -> str:
    try:
        import folder_paths

        root = Path(folder_paths.get_input_directory())
    except Exception:
        root = Path.cwd() / "input"
    return str(root.joinpath(*parts))


def _comfy_output_path(*parts: str) -> str:
    try:
        import folder_paths

        root = Path(folder_paths.get_output_directory())
    except Exception:
        root = Path.cwd() / "output"
    return str(root.joinpath(*parts))


def _font(size: int):
    candidates = [
        "C:/Windows/Fonts/YuGothM.ttc",
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _pil_to_tensor(image: Image.Image):
    import torch

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _rgb_to_tensor(rgb: np.ndarray):
    return _pil_to_tensor(Image.fromarray(np.asarray(rgb, dtype=np.uint8), "RGB"))


def _placeholder(title: str, detail: str = "", size: Tuple[int, int] = (1280, 720)):
    image = Image.new("RGB", size, (246, 247, 248))
    draw = ImageDraw.Draw(image)
    draw.text((48, 42), title, fill=(28, 32, 36), font=_font(34))
    if detail:
        draw.text((48, 104), detail, fill=(72, 76, 82), font=_font(24))
    return _pil_to_tensor(image)


def _save_pil(image: Image.Image, path: str) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)


def _average_face_tensor(folder: str, device: str, extensions: str):
    avg, meta = _average_aligned_faces(folder, device, extensions)
    if avg is None:
        return _placeholder("Visual average face", "No detectable faces"), meta
    image = Image.fromarray(avg, "RGB")
    return _pil_to_tensor(image), meta


def _classification(score: float, normalized: float, ruler: Dict) -> str:
    if math.isnan(score):
        return "顔検出失敗"
    if ruler.get("threshold_mode", "distribution") == "normalized_manual":
        if math.isnan(normalized):
            return "判定不能"
        if normalized >= ruler.get("same_threshold_normalized", 0.80):
            return "本人範囲"
        if normalized <= ruler.get("different_threshold_normalized", 0.50):
            return "別人範囲"
        return "判定保留"
    same_floor = ruler.get("same_person_p05", float("nan"))
    different_ceiling = ruler.get("different_people_p95", float("nan"))
    if math.isnan(same_floor) or math.isnan(different_ceiling):
        return "判定不能"
    if score >= same_floor:
        return "本人範囲"
    if score <= different_ceiling:
        return "別人範囲"
    return "判定保留"


def _raw_from_normalized(value: float, ruler: Dict) -> float:
    different_mean = float(ruler.get("different_people_mean", float("nan")))
    same_mean = float(ruler.get("same_person_mean", float("nan")))
    if math.isnan(value) or math.isnan(different_mean) or math.isnan(same_mean):
        return float("nan")
    return different_mean + value * (same_mean - different_mean)


def _active_raw_thresholds(ruler: Dict) -> Tuple[float, float]:
    if ruler.get("threshold_mode", "distribution") == "normalized_manual":
        different = _raw_from_normalized(
            float(ruler.get("different_threshold_normalized", 0.50)), ruler
        )
        same = _raw_from_normalized(
            float(ruler.get("same_threshold_normalized", 0.80)), ruler
        )
        return different, same
    return (
        float(ruler.get("different_people_p95", float("nan"))),
        float(ruler.get("same_person_p05", float("nan"))),
    )


def _check_backend(reference: Dict, backend: str, ruler: Optional[Dict] = None) -> None:
    reference_backend = reference.get("backend")
    if reference_backend != backend:
        raise ValueError(
            f"ID Reference backend mismatch: reference={reference_backend}, connected={backend}"
        )
    if ruler is not None and ruler.get("backend") != backend:
        raise ValueError(
            f"ID Ruler backend mismatch: ruler={ruler.get('backend')}, connected={backend}"
        )


def _make_ruler_chart(ruler: Dict) -> Image.Image:
    width, height = 1280, 600
    image = Image.new("RGB", (width, height), (246, 247, 248))
    draw = ImageDraw.Draw(image)
    title_font = _font(34)
    text_font = _font(24)
    small_font = _font(20)
    draw.text((44, 32), "ID Ruler", fill=(24, 28, 32), font=title_font)

    left, right, y = 110, 1170, 250
    draw.line((left, y, right, y), fill=(55, 60, 66), width=8)
    draw.text((left, y + 32), "別人範囲", fill=(166, 55, 55), font=text_font)
    draw.text((525, y + 32), "判定保留", fill=(166, 116, 40), font=text_font)
    draw.text((1000, y + 32), "本人範囲", fill=(45, 126, 76), font=text_font)

    values = [
        ("different mean", ruler["different_people_mean"], (185, 62, 62)),
        ("different p95", ruler["different_people_p95"], (220, 105, 80)),
        ("same p05", ruler["same_person_p05"], (82, 155, 96)),
        ("same mean", ruler["same_person_mean"], (42, 119, 72)),
    ]
    clean = [value for _, value, _ in values if not math.isnan(value)]
    minimum = min(clean) if clean else 0.0
    maximum = max(clean) if clean else 1.0
    padding = max(0.03, (maximum - minimum) * 0.15)
    minimum -= padding
    maximum += padding
    for index, (label, value, color) in enumerate(values):
        if math.isnan(value):
            continue
        x = int(left + (value - minimum) / max(1e-6, maximum - minimum) * (right - left))
        offset = -92 if index % 2 == 0 else -145
        draw.line((x, y - 22, x, y + 16), fill=color, width=4)
        draw.ellipse((x - 9, y - 31, x + 9, y - 13), fill=color)
        draw.text((max(30, x - 100), y + offset), f"{label}: {value:.4f}", fill=color, font=small_font)

    separation = ruler["same_person_p05"] - ruler["different_people_p95"]
    draw.text(
        (44, 430),
        f"境界間隔: {separation:.4f}  /  backend: {ruler['backend']}",
        fill=(55, 60, 66),
        font=text_font,
    )
    mode_text = (
        f"manual normalized: different <= {ruler.get('different_threshold_normalized', 0.50):.2f}, "
        f"same >= {ruler.get('same_threshold_normalized', 0.80):.2f}"
        if ruler.get("threshold_mode") == "normalized_manual"
        else "distribution: different p95 / same p05"
    )
    draw.text((44, 468), mode_text, fill=(55, 60, 66), font=small_font)
    draw.text(
        (44, 510),
        "CSIMは認証済み本人確認ではなく、この評価条件内の相対的な物差しです。",
        fill=(55, 60, 66),
        font=small_font,
    )
    return image


def _letterbox(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (size, size), (238, 240, 242))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def _result_gallery(frames: np.ndarray, rows: List[Dict]) -> Image.Image:
    card_w, card_h = 330, 360
    columns = min(4, max(1, len(rows)))
    row_count = max(1, math.ceil(len(rows) / columns))
    image = Image.new("RGB", (columns * card_w, row_count * card_h), (235, 237, 239))
    draw = ImageDraw.Draw(image)
    title_font = _font(23)
    text_font = _font(19)
    colors = {
        "本人範囲": (42, 126, 74),
        "判定保留": (177, 119, 37),
        "別人範囲": (180, 58, 58),
        "顔検出失敗": (110, 110, 115),
        "判定不能": (110, 110, 115),
    }
    for index, row in enumerate(rows):
        x = (index % columns) * card_w
        y = (index // columns) * card_h
        pil = _pil_from_comfy_frame(frames[index]).convert("RGB")
        thumb = _letterbox(pil, 240)
        image.paste(thumb, (x + 45, y + 12))
        label = row["classification"]
        draw.text((x + 20, y + 262), label, fill=colors.get(label, (40, 40, 40)), font=title_font)
        score = row["csim"]
        normalized = row["normalized_csim"]
        score_text = "nan" if math.isnan(score) else f"{score:.4f}"
        normalized_text = "nan" if math.isnan(normalized) else f"{normalized:.3f}"
        draw.text((x + 20, y + 300), f"CSIM {score_text} / normalized {normalized_text}", fill=(42, 46, 50), font=text_font)
        draw.text((x + 20, y + 330), Path(row["file"]).name[:34], fill=(70, 74, 80), font=text_font)
    return image


def _aligned_saliency(embedder, pil_image: Image.Image, prototype: np.ndarray, patch: int, stride: int):
    backend = embedder.backend_name
    if backend == "insightface_buffalo_l":
        from insightface.utils import face_align

        cv2 = embedder.cv2
        rgb = np.asarray(pil_image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = embedder.app.get(bgr)
        if not faces:
            return None, float("nan")
        face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
        aligned_bgr = face_align.norm_crop(bgr, landmark=face.kps, image_size=112)
        recognition = embedder.app.models["recognition"]

        def encode(face_rgb):
            face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
            vector = recognition.get_feat(face_bgr).reshape(-1)
            return _normalize(np.asarray(vector, dtype=np.float32))

        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    else:
        torch = embedder.torch
        face = embedder.detector(pil_image.convert("RGB"))
        if face is None:
            return None, float("nan")
        aligned_rgb = np.clip(
            face.detach().cpu().numpy().transpose(1, 2, 0) * 128.0 + 127.5,
            0,
            255,
        ).astype(np.uint8)

        def encode(face_rgb):
            tensor = torch.from_numpy(face_rgb.astype(np.float32).transpose(2, 0, 1))
            tensor = (tensor - 127.5) / 128.0
            with torch.no_grad():
                vector = embedder.model(tensor.unsqueeze(0).to(embedder.device)).detach().cpu().numpy()[0]
            return _normalize(vector.astype(np.float32))

    height, width = aligned_rgb.shape[:2]
    patch = max(8, min(patch, height))
    stride = max(4, stride)
    base = _cosine(encode(aligned_rgb), prototype)
    ys = list(range(0, height - patch + 1, stride))
    xs = list(range(0, width - patch + 1, stride))
    grid = np.zeros((len(ys), len(xs)), dtype=np.float32)
    fill = np.mean(aligned_rgb.reshape(-1, 3), axis=0).astype(np.uint8)
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            occluded = aligned_rgb.copy()
            occluded[y : y + patch, x : x + patch] = fill
            grid[yi, xi] = base - _cosine(encode(occluded), prototype)

    import cv2

    heat = cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)
    scale = float(np.max(np.abs(heat))) + 1e-9
    signed = np.clip(heat / scale, -1.0, 1.0)
    overlay = aligned_rgb.astype(np.float32)
    positive = np.clip(signed, 0, 1)[..., None]
    negative = np.clip(-signed, 0, 1)[..., None]
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    blue = np.zeros_like(overlay)
    blue[..., 2] = 255
    overlay = overlay * (1 - 0.55 * positive - 0.55 * negative) + red * (0.55 * positive) + blue * (0.55 * negative)
    return np.clip(overlay, 0, 255).astype(np.uint8), base


def _saliency_gallery(
    frames: np.ndarray,
    rows: List[Dict],
    embedder,
    prototype: np.ndarray,
    mode: str,
    selected_index: int,
    patch: int,
    stride: int,
):
    if mode == "off":
        return _placeholder("Saliency", "off")
    if mode == "selected":
        indices = [max(0, min(selected_index, len(rows) - 1))]
    else:
        indices = list(range(len(rows)))

    panels = []
    labels = []
    for index in indices:
        saliency, base = _aligned_saliency(
            embedder,
            _pil_from_comfy_frame(frames[index]),
            prototype,
            patch,
            stride,
        )
        if saliency is None:
            saliency = np.asarray(_placeholder("Saliency", "Face not detected")[0].mul(255).byte())
        panels.append(Image.fromarray(saliency, "RGB").resize((280, 280), Image.Resampling.LANCZOS))
        labels.append(f"{Path(rows[index]['file']).name[:28]}  CSIM {base:.4f}")

    width = max(360, 320 * len(panels))
    image = Image.new("RGB", (width, 390), (240, 242, 244))
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), "赤: 類似性を支える / 青: 不一致を支える", fill=(38, 42, 46), font=_font(22))
    for index, panel in enumerate(panels):
        x = 20 + index * 320
        image.paste(panel, (x, 64))
        draw.text((x, 352), labels[index], fill=(55, 60, 65), font=_font(17))
    return _pil_to_tensor(image)


class RogoAIIDReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_person_folder": ("STRING", {"default": _comfy_input_path("RogoAI_IDeval", "target_person")}),
                "backend": (BACKENDS, {"default": "insightface_buffalo_l"}),
                "json_output_path": ("STRING", {"default": _comfy_output_path("RogoAI_IDeval", "id_reference.json")}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "extensions": ("STRING", {"default": IMAGE_EXTENSIONS}),
            }
        }

    RETURN_TYPES = ("ID_REFERENCE", "IMAGE", "ID_BACKEND", "STRING")
    RETURN_NAMES = ("id_reference", "visual_average_face", "backend", "qc_summary")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, target_person_folder, backend, json_output_path, device, extensions):
        embedder = _make_embedder(backend, device)
        vectors = []
        rows = []
        for path in _iter_images(target_person_folder, extensions):
            vector, status = embedder.embed_file(path)
            row = {"file": str(path), "status": status}
            if vector is not None:
                vectors.append(vector)
                row["embedding"] = vector.tolist()
            rows.append(row)
        if not vectors:
            raise RuntimeError("ID Referenceを作れる顔がありません。")
        reference = {
            "backend": embedder.backend_name,
            "prototype": _normalize(np.mean(np.stack(vectors), axis=0)).tolist(),
            "target_person_folder": target_person_folder,
            "count_detected": len(vectors),
            "count_total": len(rows),
            "rows": rows,
        }
        if json_output_path:
            path = Path(json_output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
        average_face, average_meta = _average_face_tensor(
            target_person_folder, device, extensions
        )
        summary = (
            f"ID Reference: {len(vectors)}/{len(rows)} faces detected; "
            f"visual average: {average_meta['count_detected']}/{average_meta['count_total']}; "
            f"backend={embedder.backend_name}"
        )
        return reference, average_face, embedder.backend_name, summary


def _reference_embedding_rows(
    id_reference: Dict,
    embedder,
    extensions: str,
) -> List[Tuple[str, np.ndarray]]:
    embedded = []
    for row in id_reference.get("rows", []):
        values = row.get("embedding")
        if values is None:
            continue
        vector = _normalize(np.asarray(values, dtype=np.float32))
        embedded.append((row.get("file", ""), vector))
    if embedded:
        return embedded
    source_folder = id_reference.get("target_person_folder")
    if not source_folder:
        raise RuntimeError(
            "旧ID Referenceに埋め込みもtarget_person_folderもありません。Referenceを再作成してください。"
        )
    for path in _iter_images(source_folder, extensions):
        vector, status = embedder.embed_file(path)
        if vector is not None:
            embedded.append((str(path), vector))
    return embedded


def _leave_one_out_scores(
    id_reference: Dict,
    embedder,
    extensions: str,
) -> List[Dict]:
    embedded = _reference_embedding_rows(id_reference, embedder, extensions)
    if len(embedded) < 2:
        raise RuntimeError("Leave-one-outには顔検出済み本人画像が2枚以上必要です。")
    vectors = np.stack([vector for _, vector in embedded], axis=0)
    vector_sum = np.sum(vectors, axis=0)
    rows = []
    for index, (file_name, vector) in enumerate(embedded):
        loo_prototype = _normalize((vector_sum - vector) / (len(embedded) - 1))
        rows.append(
            {
                "file": file_name,
                "status": "ok",
                "csim": _cosine(vector, loo_prototype),
                "excluded_index": index,
                "prototype_count": len(embedded) - 1,
            }
        )
    return rows


class RogoAIIDRuler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "id_reference": ("ID_REFERENCE",),
                "backend": ("ID_BACKEND",),
                "same_person_mode": (["leave_one_out_reference", "external_folder"], {"default": "leave_one_out_reference"}),
                "same_person_dir": ("STRING", {"default": _comfy_input_path("RogoAI_IDeval", "same_person")}),
                "different_people_dir": ("STRING", {"default": _comfy_input_path("RogoAI_IDeval", "different_people")}),
                "threshold_mode": (["normalized_manual", "distribution"], {"default": "normalized_manual"}),
                "same_threshold_normalized": ("FLOAT", {"default": 0.80, "min": -1.0, "max": 2.0, "step": 0.01}),
                "different_threshold_normalized": ("FLOAT", {"default": 0.50, "min": -1.0, "max": 2.0, "step": 0.01}),
                "json_output_path": ("STRING", {"default": _comfy_output_path("RogoAI_IDeval", "id_ruler.json")}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "extensions": ("STRING", {"default": IMAGE_EXTENSIONS}),
            }
        }

    RETURN_TYPES = ("ID_RULER", "IMAGE", "IMAGE", "IMAGE", "ID_BACKEND")
    RETURN_NAMES = (
        "id_ruler",
        "ruler_chart",
        "same_person_average_face",
        "different_people_average_face",
        "backend",
    )
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(
        self,
        id_reference,
        backend,
        same_person_mode,
        same_person_dir,
        different_people_dir,
        threshold_mode,
        same_threshold_normalized,
        different_threshold_normalized,
        json_output_path,
        device,
        extensions,
    ):
        _check_backend(id_reference, backend)
        embedder = _make_embedder(backend, device)
        prototype = np.asarray(id_reference["prototype"], dtype=np.float32)
        if same_person_mode == "leave_one_out_reference":
            same_rows = _leave_one_out_scores(id_reference, embedder, extensions)
            same_average_source = id_reference.get("target_person_folder", same_person_dir)
        else:
            same_rows = _score_folder(embedder, same_person_dir, prototype, extensions)
            same_average_source = same_person_dir
        different_rows = _score_folder(embedder, different_people_dir, prototype, extensions)
        same_scores = [row["csim"] for row in same_rows]
        different_scores = [row["csim"] for row in different_rows]
        if not any(not math.isnan(value) for value in same_scores):
            raise RuntimeError("same_person_dirに測定可能な顔がありません。")
        if not any(not math.isnan(value) for value in different_scores):
            raise RuntimeError("different_people_dirに測定可能な顔がありません。")
        if different_threshold_normalized >= same_threshold_normalized:
            raise ValueError("different_threshold_normalized must be lower than same_threshold_normalized.")
        ruler = {
            "backend": backend,
            "same_person_mode": same_person_mode,
            "threshold_mode": threshold_mode,
            "same_threshold_normalized": float(same_threshold_normalized),
            "different_threshold_normalized": float(different_threshold_normalized),
            "same_person_mean": _safe_mean(same_scores),
            "same_person_std": _safe_std(same_scores),
            "same_person_p05": _safe_percentile(same_scores, 5),
            "different_people_mean": _safe_mean(different_scores),
            "different_people_std": _safe_std(different_scores),
            "different_people_p95": _safe_percentile(different_scores, 95),
            "same_person": same_rows,
            "different_people": different_rows,
        }
        if json_output_path:
            path = Path(json_output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(ruler, ensure_ascii=False, indent=2), encoding="utf-8")
        chart = _make_ruler_chart(ruler)
        same_average, _ = _average_face_tensor(
            same_average_source, device, extensions
        )
        different_average, _ = _average_face_tensor(
            different_people_dir, device, extensions
        )
        return ruler, _pil_to_tensor(chart), same_average, different_average, backend


class RogoAIIDImageFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_folder": ("STRING", {"default": _comfy_input_path("RogoAI_IDeval", "eval_images")}),
                "condition_label": ("STRING", {"default": ""}),
                "image_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "extensions": ("STRING", {"default": IMAGE_EXTENSIONS}),
            }
        }

    RETURN_TYPES = ("IMAGE", "ID_IMAGE_META")
    RETURN_NAMES = ("images", "image_metadata")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, image_folder, condition_label, image_size, extensions):
        import torch

        paths = _iter_images(image_folder, extensions)
        if not paths:
            raise RuntimeError("画像フォルダが空です。")
        tensors = []
        files = []
        for path in paths:
            with Image.open(path) as image:
                prepared = _letterbox(image, image_size)
            tensors.append(_pil_to_tensor(prepared))
            files.append(str(path))
        metadata = {"files": files, "condition_label": condition_label}
        return torch.cat(tensors, dim=0), metadata


class RogoAIIDEvalImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "id_reference": ("ID_REFERENCE",),
                "id_ruler": ("ID_RULER",),
                "backend": ("ID_BACKEND",),
                "json_output_path": ("STRING", {"default": _comfy_output_path("RogoAI_IDeval", "id_image_results.json")}),
                "saliency_mode": (["off", "selected", "all"], {"default": "off"}),
                "saliency_selected_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "saliency_patch": ("INT", {"default": 28, "min": 8, "max": 96}),
                "saliency_stride": ("INT", {"default": 14, "min": 4, "max": 48}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "image_metadata": ("ID_IMAGE_META",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "ID_EVAL_RESULTS", "ID_BACKEND")
    RETURN_NAMES = ("result_gallery", "saliency_gallery", "evaluation_results", "backend")
    FUNCTION = "evaluate"
    CATEGORY = CATEGORY

    def evaluate(
        self,
        images,
        id_reference,
        id_ruler,
        backend,
        json_output_path,
        saliency_mode,
        saliency_selected_index,
        saliency_patch,
        saliency_stride,
        device,
        image_metadata=None,
    ):
        _check_backend(id_reference, backend, id_ruler)
        frames = _to_numpy_image(images)
        metadata = image_metadata or {}
        files = list(metadata.get("files", []))
        while len(files) < len(frames):
            files.append(f"image_{len(files) + 1:03d}")
        condition = metadata.get("condition_label", "")
        embedder = _make_embedder(backend, device)
        prototype = np.asarray(id_reference["prototype"], dtype=np.float32)
        rows = []
        for index, frame in enumerate(frames):
            vector, status = embedder.embed_pil(_pil_from_comfy_frame(frame))
            score = _cosine(vector, prototype) if vector is not None else float("nan")
            normalized = _normalize_score(
                score,
                {
                    "genuine_mean": id_ruler["same_person_mean"],
                    "impostor_mean": id_ruler["different_people_mean"],
                },
            )
            rows.append(
                {
                    "index": index,
                    "file": files[index],
                    "condition": condition,
                    "status": status,
                    "face_detected": vector is not None,
                    "csim": score,
                    "normalized_csim": normalized,
                    "classification": _classification(score, normalized, id_ruler),
                }
            )
        evaluation_results = {
            "backend": backend,
            "count_total": len(rows),
            "count_detected": sum(1 for row in rows if row["face_detected"]),
            "results": rows,
        }
        if json_output_path:
            path = Path(json_output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(evaluation_results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        result_image = _result_gallery(frames, rows)
        saliency = _saliency_gallery(
            frames,
            rows,
            embedder,
            prototype,
            saliency_mode,
            saliency_selected_index,
            saliency_patch,
            saliency_stride,
        )
        return _pil_to_tensor(result_image), saliency, evaluation_results, backend


def _draw_dashed_hline(draw, left, right, y, fill, width=2, dash=12, gap=8):
    x = left
    while x < right:
        draw.line((x, y, min(right, x + dash), y), fill=fill, width=width)
        x += dash + gap


def _video_sample_indices(
    mode: str,
    scores: Sequence[float],
    fps: float,
    interval_seconds: float,
    max_samples: int,
) -> Tuple[List[int], bool]:
    if not scores or mode == "off":
        return [], False
    if mode == "first_worst_last":
        finite = [(index, score) for index, score in enumerate(scores) if not math.isnan(score)]
        worst = min(finite, key=lambda item: item[1])[0] if finite else 0
        return list(dict.fromkeys([0, worst, len(scores) - 1])), False

    step = max(1, int(round(max(0.001, fps) * max(0.01, interval_seconds))))
    indices = list(range(0, len(scores), step))
    if indices[-1] != len(scores) - 1:
        indices.append(len(scores) - 1)
    truncated = len(indices) > max_samples
    if truncated:
        indices = indices[: max(1, max_samples - 1)] + [len(scores) - 1]
        indices = list(dict.fromkeys(indices))
    return indices, truncated


def _video_saliency_gallery(
    frames: np.ndarray,
    rows: List[Dict],
    indices: Sequence[int],
    embedder,
    prototype: np.ndarray,
    patch: int,
    stride: int,
    fps: float,
    truncated: bool,
):
    if not indices:
        return _placeholder("Video Saliency", "off")

    card_w, card_h = 330, 420
    columns = min(4, max(1, len(indices)))
    row_count = max(1, math.ceil(len(indices) / columns))
    header_h = 72
    image = Image.new(
        "RGB",
        (columns * card_w, header_h + row_count * card_h),
        (238, 240, 242),
    )
    draw = ImageDraw.Draw(image)
    draw.text(
        (22, 15),
        "赤: 類似性を支える / 青: 不一致を支える",
        fill=(38, 42, 46),
        font=_font(21),
    )
    if truncated:
        draw.text(
            (22, 43),
            "表示上限に達したため、途中のサンプルを省略しています。",
            fill=(150, 94, 31),
            font=_font(16),
        )

    colors = {
        "本人範囲": (42, 126, 74),
        "判定保留": (177, 119, 37),
        "別人範囲": (180, 58, 58),
        "顔検出失敗": (110, 110, 115),
        "判定不能": (110, 110, 115),
    }
    for card_index, frame_index in enumerate(indices):
        row = rows[frame_index]
        x = (card_index % columns) * card_w
        y = header_h + (card_index // columns) * card_h
        saliency, _ = _aligned_saliency(
            embedder,
            _pil_from_comfy_frame(frames[frame_index]),
            prototype,
            patch,
            stride,
        )
        if saliency is None:
            panel = Image.new("RGB", (280, 280), (225, 227, 230))
            panel_draw = ImageDraw.Draw(panel)
            panel_draw.text((52, 126), "Face not detected", fill=(90, 94, 98), font=_font(18))
        else:
            panel = Image.fromarray(saliency, "RGB").resize(
                (280, 280), Image.Resampling.LANCZOS
            )
        image.paste(panel, (x + 25, y + 8))

        label = row["classification"]
        draw.text((x + 20, y + 298), label, fill=colors.get(label, (50, 50, 50)), font=_font(23))
        score = row["csim"]
        normalized = row["normalized_csim"]
        score_text = "nan" if math.isnan(score) else f"{score:.4f}"
        normalized_text = "nan" if math.isnan(normalized) else f"{normalized:.3f}"
        draw.text(
            (x + 20, y + 330),
            f"CSIM {score_text} / normalized {normalized_text}",
            fill=(42, 46, 50),
            font=_font(17),
        )
        draw.text(
            (x + 20, y + 358),
            f"t={frame_index / max(0.001, fps):.2f}s / frame {frame_index}",
            fill=(70, 74, 80),
            font=_font(17),
        )
        yaw = row.get("yaw", float("nan"))
        yaw_text = "nan" if math.isnan(yaw) else f"{yaw:+.1f} deg"
        draw.text(
            (x + 20, y + 386),
            f"pose: {row.get('pose_class', '未評価')} / yaw {yaw_text}",
            fill=(70, 74, 80),
            font=_font(17),
        )
    return _pil_to_tensor(image)


def _pose_class(yaw: float, frontal_max_abs_yaw: float, profile_min_abs_yaw: float) -> str:
    if math.isnan(yaw):
        return "検出不能"
    absolute = abs(yaw)
    if absolute <= frontal_max_abs_yaw:
        return "正面"
    if absolute < profile_min_abs_yaw:
        return "斜め"
    return "横顔"


def _embed_and_pose_pil(
    embedder,
    pil_image: Image.Image,
    pose_detector,
    frontal_max_abs_yaw: float,
    profile_min_abs_yaw: float,
):
    pose = None
    det_score = float("nan")
    if embedder.backend_name == "insightface_buffalo_l":
        rgb = np.asarray(pil_image.convert("RGB"))
        bgr = embedder.cv2.cvtColor(rgb, embedder.cv2.COLOR_RGB2BGR)
        faces = embedder.app.get(bgr)
        if not faces:
            vector, status = None, "no_face"
        else:
            face = max(
                faces,
                key=lambda item: float(
                    (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
                ),
            )
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            vector = (
                _normalize(np.asarray(embedding, dtype=np.float32))
                if embedding is not None
                else None
            )
            status = "ok" if vector is not None else "no_embedding"
            raw_pose = getattr(face, "pose", None)
            if raw_pose is not None and len(raw_pose) >= 3:
                pose = [float(raw_pose[0]), float(raw_pose[1]), float(raw_pose[2])]
            det_score = float(getattr(face, "det_score", float("nan")))
    else:
        vector, status = embedder.embed_pil(pil_image)
        if pose_detector is not None:
            rgb = np.asarray(pil_image.convert("RGB"))
            bgr = pose_detector.cv2.cvtColor(rgb, pose_detector.cv2.COLOR_RGB2BGR)
            faces = pose_detector.app.get(bgr)
            if faces:
                face = max(
                    faces,
                    key=lambda item: float(
                        (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
                    ),
                )
                raw_pose = getattr(face, "pose", None)
                if raw_pose is not None and len(raw_pose) >= 3:
                    pose = [float(raw_pose[0]), float(raw_pose[1]), float(raw_pose[2])]
                det_score = float(getattr(face, "det_score", float("nan")))

    pitch, yaw, roll = pose if pose is not None else (float("nan"),) * 3
    return {
        "vector": vector,
        "status": status,
        "pose_detected": pose is not None,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "abs_yaw": abs(yaw) if not math.isnan(yaw) else float("nan"),
        "pose_class": _pose_class(
            yaw, frontal_max_abs_yaw, profile_min_abs_yaw
        ),
        "pose_det_score": det_score,
    }


def _pose_score_summary(rows: Sequence[Dict], pose_class: str) -> Dict:
    selected = [row for row in rows if row.get("pose_class") == pose_class]
    scores = [row.get("csim", float("nan")) for row in selected]
    detected_scores = [value for value in scores if not math.isnan(value)]
    classifications = [row.get("classification") for row in selected]
    count = len(selected)
    return {
        "frame_count": count,
        "score_count": len(detected_scores),
        "mean_csim": _safe_mean(scores),
        "p05_csim": _safe_percentile(scores, 5),
        "min_csim": min(detected_scores) if detected_scores else float("nan"),
        "same_range_rate": classifications.count("本人範囲") / count if count else float("nan"),
        "hold_rate": classifications.count("判定保留") / count if count else float("nan"),
        "different_range_rate": classifications.count("別人範囲") / count if count else float("nan"),
    }


def _pose_recovery_summary(
    rows: Sequence[Dict],
    fps: float,
    same_threshold: float,
    window_seconds: float,
    tolerance_csim: float,
    minimum_profile_seconds: float,
) -> Dict:
    window = max(1, int(round(window_seconds * fps)))
    minimum_profile_frames = max(1, int(round(minimum_profile_seconds * fps)))
    events = []
    unreturned = 0
    index = 0
    while index < len(rows):
        if rows[index].get("pose_class") != "横顔":
            index += 1
            continue
        start = index
        cursor = index
        profile_indices = []
        while cursor < len(rows) and rows[cursor].get("pose_class") != "正面":
            if rows[cursor].get("pose_class") == "横顔":
                profile_indices.append(cursor)
            cursor += 1
        if len(profile_indices) < minimum_profile_frames:
            index = max(cursor, index + 1)
            continue
        if cursor >= len(rows):
            unreturned += 1
            events.append(
                {
                    "profile_start_frame": start,
                    "profile_last_frame": profile_indices[-1],
                    "returned_to_frontal": False,
                }
            )
            break

        return_frame = cursor
        pre_rows = [
            row for row in rows[:start] if row.get("pose_class") == "正面"
        ][-window:]
        post_rows = [
            row
            for row in rows[return_frame : min(len(rows), return_frame + window)]
            if row.get("pose_class") == "正面"
        ]
        pre_mean = _safe_mean([row.get("csim", float("nan")) for row in pre_rows])
        post_scores = [row.get("csim", float("nan")) for row in post_rows]
        post_mean = _safe_mean(post_scores)
        post_p05 = _safe_percentile(post_scores, 5)
        recovery_drop = (
            pre_mean - post_mean
            if not math.isnan(pre_mean) and not math.isnan(post_mean)
            else float("nan")
        )
        absolute_failure = (
            not math.isnan(post_mean)
            and not math.isnan(same_threshold)
            and post_mean < same_threshold
        )
        relative_failure = (
            not math.isnan(recovery_drop) and recovery_drop > tolerance_csim
        )
        events.append(
            {
                "profile_start_frame": start,
                "profile_last_frame": profile_indices[-1],
                "return_frame": return_frame,
                "return_time_seconds": return_frame / max(0.001, fps),
                "returned_to_frontal": True,
                "pre_turn_frontal_mean": pre_mean,
                "post_return_frontal_mean": post_mean,
                "post_return_frontal_p05": post_p05,
                "recovery_drop": recovery_drop,
                "absolute_recovery_failure": absolute_failure,
                "relative_recovery_failure": relative_failure,
            }
        )
        index = return_frame + 1

    returned = [event for event in events if event.get("returned_to_frontal")]
    return {
        "profile_episode_count": len(events),
        "returned_profile_episode_count": len(returned),
        "unreturned_profile_episode_count": unreturned,
        "return_to_frontal_mean_csim": _safe_mean(
            [event.get("post_return_frontal_mean", float("nan")) for event in returned]
        ),
        "return_to_frontal_p05_csim": _safe_percentile(
            [event.get("post_return_frontal_p05", float("nan")) for event in returned],
            5,
        ),
        "mean_recovery_drop": _safe_mean(
            [event.get("recovery_drop", float("nan")) for event in returned]
        ),
        "absolute_recovery_failure_count": sum(
            bool(event.get("absolute_recovery_failure")) for event in returned
        ),
        "relative_recovery_failure_count": sum(
            bool(event.get("relative_recovery_failure")) for event in returned
        ),
        "events": events,
    }


def _video_drift_chart(
    rows: Sequence[Dict], summary: Dict, ruler: Dict, fps: float
) -> Image.Image:
    width, height = 1280, 910
    image = Image.new("RGB", (width, height), (246, 247, 248))
    draw = ImageDraw.Draw(image)
    draw.text((42, 24), "Pose-aware ID Drift Over Video Time", fill=(24, 28, 32), font=_font(34))
    left, top, right, bottom = 100, 100, 1180, 530
    scores = [row["csim"] for row in rows]
    different_threshold, same_threshold = _active_raw_thresholds(ruler)
    different_p95 = float(ruler.get("different_people_p95", float("nan")))
    same_p05 = float(ruler.get("same_person_p05", float("nan")))
    clean = [score for score in scores if not math.isnan(score)]
    if clean:
        scale_values = clean + [
            value
            for value in (different_threshold, same_threshold, different_p95, same_p05)
            if not math.isnan(value)
        ]
        minimum = min(scale_values)
        maximum = max(scale_values)
        padding = max(0.02, (maximum - minimum) * 0.2)
        minimum -= padding
        maximum += padding

        def y_for(value):
            return bottom - int(
                (value - minimum) / max(1e-6, maximum - minimum) * (bottom - top)
            )

        if not math.isnan(same_threshold) and not math.isnan(different_threshold):
            same_y = y_for(same_threshold)
            different_y = y_for(different_threshold)
            draw.rectangle((left, top, right, same_y), fill=(227, 242, 232))
            draw.rectangle((left, same_y, right, different_y), fill=(250, 242, 220))
            draw.rectangle((left, different_y, right, bottom), fill=(248, 229, 229))

        mode = ruler.get("threshold_mode", "distribution")
        if not math.isnan(same_threshold):
            y = y_for(same_threshold)
            draw.line((left, y, right, y), fill=(42, 126, 74), width=3)
            suffix = "same p05" if mode == "distribution" else "active"
            draw.text((left + 8, y - 27), f"本人境界 {same_threshold:.4f} ({suffix})", fill=(35, 105, 62), font=_font(17))
        if not math.isnan(different_threshold):
            y = y_for(different_threshold)
            draw.line((left, y, right, y), fill=(180, 58, 58), width=3)
            suffix = "different p95" if mode == "distribution" else "active"
            draw.text((left + 8, y + 5), f"別人境界 {different_threshold:.4f} ({suffix})", fill=(145, 44, 44), font=_font(17))

        if mode != "distribution":
            if not math.isnan(same_p05):
                y = y_for(same_p05)
                _draw_dashed_hline(draw, left, right, y, (70, 142, 84))
                draw.text((right - 230, y - 24), f"same p05 {same_p05:.4f}", fill=(55, 116, 69), font=_font(16))
            if not math.isnan(different_p95):
                y = y_for(different_p95)
                _draw_dashed_hline(draw, left, right, y, (207, 91, 75))
                draw.text((right - 270, y + 4), f"different p95 {different_p95:.4f}", fill=(164, 67, 56), font=_font(16))

        points = []
        for index, (score, row) in enumerate(zip(scores, rows)):
            if math.isnan(score):
                continue
            x = left + int(index / max(1, len(scores) - 1) * (right - left))
            y = y_for(score)
            points.append((index, x, y, row))
        segment = []
        previous_index = None
        for frame_index, x, y, row in points:
            if previous_index is not None and frame_index != previous_index + 1:
                if len(segment) >= 2:
                    draw.line(segment, fill=(52, 101, 164), width=4)
                segment = []
            segment.append((x, y))
            previous_index = frame_index
        if len(segment) >= 2:
            draw.line(segment, fill=(52, 101, 164), width=4)
        point_colors = {
            "本人範囲": (42, 126, 74),
            "判定保留": (177, 119, 37),
            "別人範囲": (180, 58, 58),
        }
        for _, x, y, row in points:
            color = point_colors.get(row["classification"], (90, 94, 98))
            pose_class = row.get("pose_class", "検出不能")
            if pose_class == "正面":
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            elif pose_class == "斜め":
                draw.polygon([(x, y - 6), (x - 6, y + 5), (x + 6, y + 5)], fill=color)
            elif pose_class == "横顔":
                draw.polygon([(x, y - 6), (x - 6, y), (x, y + 6), (x + 6, y)], fill=color)
            else:
                draw.line((x - 5, y - 5, x + 5, y + 5), fill=(90, 94, 98), width=2)
                draw.line((x - 5, y + 5, x + 5, y - 5), fill=(90, 94, 98), width=2)

        for index, row in enumerate(rows):
            if not math.isnan(row.get("csim", float("nan"))):
                continue
            x = left + int(index / max(1, len(rows) - 1) * (right - left))
            y = bottom - 8
            draw.line((x - 5, y - 5, x + 5, y + 5), fill=(90, 94, 98), width=2)
            draw.line((x - 5, y + 5, x + 5, y - 5), fill=(90, 94, 98), width=2)

        draw.text((18, top - 8), f"{maximum:.3f}", fill=(70, 74, 80), font=_font(16))
        draw.text((18, bottom - 12), f"{minimum:.3f}", fill=(70, 74, 80), font=_font(16))
        duration = (len(scores) - 1) / max(0.001, fps) if scores else 0.0
        draw.text((left, bottom + 8), "0.0s", fill=(70, 74, 80), font=_font(16))
        draw.text((right - 70, bottom + 8), f"{duration:.1f}s", fill=(70, 74, 80), font=_font(16))
    draw.rectangle((left, top, right, bottom), outline=(80, 84, 89), width=2)

    yaw_top, yaw_bottom = 585, 710
    draw.rectangle((left, yaw_top, right, yaw_bottom), outline=(80, 84, 89), width=2)
    yaw_values = [row.get("abs_yaw", float("nan")) for row in rows]
    clean_yaw = [value for value in yaw_values if not math.isnan(value)]
    yaw_max = max(90.0, max(clean_yaw) + 10.0 if clean_yaw else 90.0)

    def yaw_y(value):
        return yaw_bottom - int(min(yaw_max, value) / yaw_max * (yaw_bottom - yaw_top))

    frontal_limit = float(summary.get("frontal_max_abs_yaw", 15.0))
    profile_limit = float(summary.get("profile_min_abs_yaw", 35.0))
    draw.line((left, yaw_y(frontal_limit), right, yaw_y(frontal_limit)), fill=(76, 142, 92), width=2)
    draw.line((left, yaw_y(profile_limit), right, yaw_y(profile_limit)), fill=(190, 119, 45), width=2)
    draw.text((left + 7, yaw_y(frontal_limit) - 22), f"frontal <= {frontal_limit:.0f} deg", fill=(55, 116, 69), font=_font(15))
    draw.text((left + 7, yaw_y(profile_limit) - 22), f"profile >= {profile_limit:.0f} deg", fill=(153, 91, 34), font=_font(15))
    pose_colors = {"正面": (42, 126, 74), "斜め": (177, 119, 37), "横顔": (180, 58, 58)}
    yaw_segment = []
    previous_index = None
    for index, (value, row) in enumerate(zip(yaw_values, rows)):
        if math.isnan(value):
            if len(yaw_segment) >= 2:
                draw.line(yaw_segment, fill=(86, 104, 135), width=3)
            yaw_segment = []
            previous_index = None
            x = left + int(index / max(1, len(rows) - 1) * (right - left))
            draw.line((x - 5, yaw_bottom - 10, x + 5, yaw_bottom), fill=(90, 94, 98), width=2)
            draw.line((x - 5, yaw_bottom, x + 5, yaw_bottom - 10), fill=(90, 94, 98), width=2)
            continue
        x = left + int(index / max(1, len(rows) - 1) * (right - left))
        y = yaw_y(value)
        if previous_index is not None and index != previous_index + 1:
            if len(yaw_segment) >= 2:
                draw.line(yaw_segment, fill=(86, 104, 135), width=3)
            yaw_segment = []
        yaw_segment.append((x, y))
        previous_index = index
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=pose_colors.get(row.get("pose_class"), (90, 94, 98)))
    if len(yaw_segment) >= 2:
        draw.line(yaw_segment, fill=(86, 104, 135), width=3)
    draw.text((22, yaw_top + 44), "abs yaw", fill=(70, 74, 80), font=_font(16))

    frontal = summary.get("pose_summary", {}).get("frontal", {})
    profile = summary.get("pose_summary", {}).get("profile", {})
    recovery = summary.get("pose_recovery", {})
    labels = [
        f"all mean/p05: {summary.get('mean_csim', float('nan')):.4f} / {summary.get('p05_csim', float('nan')):.4f}",
        f"frontal mean/p05: {frontal.get('mean_csim', float('nan')):.4f} / {frontal.get('p05_csim', float('nan')):.4f}",
        f"profile mean: {profile.get('mean_csim', float('nan')):.4f}",
        f"detect: {summary.get('face_detect_rate', float('nan')):.1%}",
    ]
    for index, label in enumerate(labels):
        draw.text((55 + index * 300, 752), label, fill=(55, 60, 66), font=_font(17))
    recovery_labels = [
        f"return frontal mean: {recovery.get('return_to_frontal_mean_csim', float('nan')):.4f}",
        f"recovery failures(abs/rel): {recovery.get('absolute_recovery_failure_count', 0)} / {recovery.get('relative_recovery_failure_count', 0)}",
        f"unreturned profile: {recovery.get('unreturned_profile_episode_count', 0)}",
        f"pose detect: {summary.get('pose_detection_rate', float('nan')):.1%}",
    ]
    for index, label in enumerate(recovery_labels):
        draw.text((55 + index * 300, 785), label, fill=(55, 60, 66), font=_font(17))
    draw.text(
        (55, 842),
        "ID色: 緑=本人範囲 / 黄=判定保留 / 赤=別人範囲    姿勢形: ○正面 / △斜め / ◇横顔 / ×検出不能",
        fill=(55, 60, 66),
        font=_font(18),
    )
    return image


def _prompt_video_source(prompt, unique_id) -> Tuple[str, str]:
    """Resolve the upstream video widget connected to this node's frames input."""
    if not isinstance(prompt, dict):
        return "", ""
    queue = [str(unique_id)] if unique_id is not None else []
    visited = set()
    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        node = prompt.get(node_id) or prompt.get(int(node_id) if node_id.isdigit() else node_id)
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        video = inputs.get("video")
        if isinstance(video, str) and video.strip():
            return Path(video).name, node_id
        preferred = []
        if "frames" in inputs:
            preferred.append(inputs.get("frames"))
        preferred.extend(value for key, value in inputs.items() if key != "frames")
        for value in preferred:
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                upstream_id = value[0]
                if isinstance(upstream_id, (str, int)):
                    queue.append(str(upstream_id))
    return "", ""


def _safe_video_stem(video_name: str) -> str:
    stem = Path(str(video_name or "")).stem.strip()
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in stem)
    return safe.strip("._") or "unknown_video"


def _result_number_tag(value: float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(number):
        return "nan"
    text = f"{number:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _evaluation_result_tag(
    threshold_mode: str,
    same_threshold: float,
    different_threshold: float,
    start_seconds: float,
    end_seconds: float,
) -> str:
    safe_mode = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(threshold_mode or "unknown_mode")
    ).strip("_-") or "unknown_mode"
    return (
        f"{safe_mode}"
        f"__same-{_result_number_tag(same_threshold)}"
        f"__diff-{_result_number_tag(different_threshold)}"
        f"__win-{_result_number_tag(start_seconds)}-{_result_number_tag(end_seconds)}"
    )


def _video_json_path(
    json_output_path: str,
    video_name: str,
    append_video_name: str,
    evaluation_tag: str = "",
) -> Path:
    path = Path(json_output_path)
    if append_video_name == "on":
        stem = _safe_video_stem(video_name)
        suffix = path.suffix or ".json"
        tag = f"__{stem}"
        if evaluation_tag:
            tag += f"__{evaluation_tag}"
        if not path.name.endswith(f"{tag}{suffix}"):
            path = path.with_name(f"{path.stem}{tag}{suffix}")
    return path


def _identity_evaluation_window(
    rows: Sequence[Dict],
    fps: float,
    same_threshold: float,
    start_seconds: float,
    end_seconds: float,
    stable_identity_seconds: float,
) -> Dict:
    if end_seconds <= start_seconds:
        raise ValueError("evaluation_end_seconds must be greater than evaluation_start_seconds.")
    selected = [
        row for row in rows
        if float(start_seconds) <= float(row.get("time_seconds", 0.0)) < float(end_seconds)
    ]
    detected = [row for row in selected if row.get("face_detected")]
    same_flags = [
        bool(row.get("face_detected"))
        and not math.isnan(float(row.get("csim", float("nan"))))
        and float(row.get("csim")) >= float(same_threshold)
        for row in selected
    ]
    same_count = sum(same_flags)
    longest = current = 0
    first_same_index = None
    stable_start_index = None
    stable_frames = max(1, int(math.ceil(float(stable_identity_seconds) * fps)))
    for local_index, flag in enumerate(same_flags):
        if flag:
            if first_same_index is None:
                first_same_index = local_index
            current += 1
            longest = max(longest, current)
            if stable_start_index is None and current >= stable_frames:
                stable_start_index = local_index - stable_frames + 1
        else:
            current = 0
    def frame_info(local_index):
        if local_index is None or not selected:
            return None, float("nan")
        row = selected[local_index]
        return int(row.get("frame", local_index)), float(row.get("time_seconds", 0.0))
    first_frame, first_time = frame_info(first_same_index)
    stable_frame, stable_time = frame_info(stable_start_index)
    return {
        "start_seconds": float(start_seconds),
        "end_seconds": float(end_seconds),
        "frame_count": len(selected),
        "detected_frame_count": len(detected),
        "undetected_frame_count": len(selected) - len(detected),
        "same_person_frame_count": same_count,
        "same_person_seconds": same_count / fps if fps else 0.0,
        "same_person_rate_strict": same_count / len(selected) if selected else 0.0,
        "same_person_rate_detected": same_count / len(detected) if detected else 0.0,
        "first_same_person_frame": first_frame,
        "first_same_person_time_seconds": first_time,
        "first_same_person_latency_seconds": (
            first_time - float(start_seconds) if not math.isnan(first_time) else float("nan")
        ),
        "stable_identity_seconds_required": float(stable_identity_seconds),
        "first_stable_same_person_frame": stable_frame,
        "first_stable_same_person_time_seconds": stable_time,
        "first_stable_same_person_latency_seconds": (
            stable_time - float(start_seconds) if not math.isnan(stable_time) else float("nan")
        ),
        "longest_same_person_run_frames": longest,
        "longest_same_person_run_seconds": longest / fps if fps else 0.0,
        "recovery_success": stable_start_index is not None,
    }


class RogoAIIDEvalVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "id_reference": ("ID_REFERENCE",),
                "id_ruler": ("ID_RULER",),
                "backend": ("ID_BACKEND",),
                "json_output_path": ("STRING", {"default": _comfy_output_path("RogoAI_IDeval", "id_video_results.json")}),
                "representative_saliency": (["off", "first_worst_last", "interval_seconds"], {"default": "off"}),
                "saliency_patch": ("INT", {"default": 28, "min": 8, "max": 96}),
                "saliency_stride": ("INT", {"default": 14, "min": 4, "max": 48}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "frames_per_second": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.1}),
                "saliency_interval_seconds": ("FLOAT", {"default": 5.0, "min": 0.25, "max": 600.0, "step": 0.25}),
                "saliency_max_samples": ("INT", {"default": 12, "min": 1, "max": 240}),
                "pose_analysis": (["on", "off"], {"default": "on"}),
                "frontal_max_abs_yaw": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "profile_min_abs_yaw": ("FLOAT", {"default": 35.0, "min": 5.0, "max": 89.0, "step": 1.0}),
                "recovery_window_seconds": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "recovery_tolerance_csim": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.50, "step": 0.01}),
                "minimum_profile_seconds": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 5.0, "step": 0.05}),
                "analyzed_video_name": ("STRING", {"default": ""}),
                "json_append_video_name": (["on", "off"], {"default": "on"}),
                "evaluation_start_seconds": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 3600.0, "step": 0.125}),
                "evaluation_end_seconds": ("FLOAT", {"default": 30.0, "min": 0.125, "max": 3600.0, "step": 0.125}),
                "stable_identity_seconds": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.125}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("ID_VIDEO_RESULTS", "IMAGE", "IMAGE", "ID_BACKEND")
    RETURN_NAMES = ("video_results", "drift_chart", "representative_saliency", "backend")
    FUNCTION = "evaluate"
    CATEGORY = CATEGORY

    def evaluate(
        self,
        frames,
        id_reference,
        id_ruler,
        backend,
        json_output_path,
        representative_saliency,
        saliency_patch,
        saliency_stride,
        device,
        frames_per_second,
        saliency_interval_seconds,
        saliency_max_samples,
        pose_analysis,
        frontal_max_abs_yaw,
        profile_min_abs_yaw,
        recovery_window_seconds,
        recovery_tolerance_csim,
        minimum_profile_seconds,
        analyzed_video_name="",
        json_append_video_name="on",
        evaluation_start_seconds=20.0,
        evaluation_end_seconds=30.0,
        stable_identity_seconds=0.5,
        prompt=None,
        unique_id=None,
    ):
        _check_backend(id_reference, backend, id_ruler)
        if profile_min_abs_yaw <= frontal_max_abs_yaw:
            raise ValueError("profile_min_abs_yaw must be greater than frontal_max_abs_yaw.")
        array = _to_numpy_image(frames)
        embedder = _make_embedder(backend, device)
        pose_enabled = pose_analysis == "on"
        pose_detector = None
        if pose_enabled and backend != "insightface_buffalo_l":
            pose_detector = _make_embedder("insightface_buffalo_l", device)
        prototype = np.asarray(id_reference["prototype"], dtype=np.float32)
        rows = []
        scores = []
        fps = max(0.001, float(frames_per_second))
        prompt_video_name, source_video_node_id = _prompt_video_source(prompt, unique_id)
        analyzed_video_name = str(analyzed_video_name or prompt_video_name or "unknown_video").strip()
        for index, frame in enumerate(array):
            pil_frame = _pil_from_comfy_frame(frame)
            if pose_enabled:
                face_info = _embed_and_pose_pil(
                    embedder,
                    pil_frame,
                    pose_detector,
                    float(frontal_max_abs_yaw),
                    float(profile_min_abs_yaw),
                )
                vector = face_info.pop("vector")
                status = face_info.pop("status")
            else:
                vector, status = embedder.embed_pil(pil_frame)
                face_info = {
                    "pose_detected": False,
                    "pitch": float("nan"),
                    "yaw": float("nan"),
                    "roll": float("nan"),
                    "abs_yaw": float("nan"),
                    "pose_class": "未評価",
                    "pose_det_score": float("nan"),
                }
            score = _cosine(vector, prototype) if vector is not None else float("nan")
            normalized = _normalize_score(
                score,
                {
                    "genuine_mean": id_ruler["same_person_mean"],
                    "impostor_mean": id_ruler["different_people_mean"],
                },
            )
            rows.append(
                {
                    "frame": index,
                    "time_seconds": index / fps,
                    "status": status,
                    "face_detected": vector is not None,
                    "csim": score,
                    "normalized_csim": normalized,
                    "classification": _classification(score, normalized, id_ruler),
                    **face_info,
                }
            )
            scores.append(score)
        detected = [score for score in scores if not math.isnan(score)]
        midpoint = max(1, len(scores) // 2)
        early = _safe_mean(scores[:midpoint])
        late = _safe_mean(scores[midpoint:])
        different_threshold, same_threshold = _active_raw_thresholds(id_ruler)
        classification_counts = {}
        for row in rows:
            label = row["classification"]
            classification_counts[label] = classification_counts.get(label, 0) + 1
        frontal_summary = _pose_score_summary(rows, "正面")
        oblique_summary = _pose_score_summary(rows, "斜め")
        profile_summary = _pose_score_summary(rows, "横顔")
        pose_undetected_summary = _pose_score_summary(rows, "検出不能")
        recovery_summary = _pose_recovery_summary(
            rows,
            fps,
            same_threshold,
            float(recovery_window_seconds),
            float(recovery_tolerance_csim),
            float(minimum_profile_seconds),
        ) if pose_enabled else {
            "profile_episode_count": 0,
            "returned_profile_episode_count": 0,
            "unreturned_profile_episode_count": 0,
            "return_to_frontal_mean_csim": float("nan"),
            "return_to_frontal_p05_csim": float("nan"),
            "mean_recovery_drop": float("nan"),
            "absolute_recovery_failure_count": 0,
            "relative_recovery_failure_count": 0,
            "events": [],
        }
        sample_indices, samples_truncated = _video_sample_indices(
            representative_saliency,
            scores,
            fps,
            float(saliency_interval_seconds),
            int(saliency_max_samples),
        )
        evaluation_window = _identity_evaluation_window(
            rows,
            fps,
            same_threshold,
            float(evaluation_start_seconds),
            float(evaluation_end_seconds),
            float(stable_identity_seconds),
        )
        threshold_mode = id_ruler.get("threshold_mode", "distribution")
        evaluation_tag = _evaluation_result_tag(
            threshold_mode,
            same_threshold,
            different_threshold,
            float(evaluation_start_seconds),
            float(evaluation_end_seconds),
        )
        evaluation_id = f"{_safe_video_stem(analyzed_video_name)}__{evaluation_tag}"
        video_results = {
            "analyzed_video_name": analyzed_video_name,
            "analyzed_video_stem": _safe_video_stem(analyzed_video_name),
            "evaluation_id": evaluation_id,
            "evaluation_tag": evaluation_tag,
            "source_video_node_id": source_video_node_id,
            "backend": backend,
            "frames_per_second": fps,
            "duration_seconds": (len(scores) - 1) / fps if scores else 0.0,
            "frame_count": len(scores),
            "detected_count": len(detected),
            "face_detect_rate": len(detected) / len(scores) if scores else 0.0,
            "pose_analysis_enabled": pose_enabled,
            "pose_detection_rate": (
                sum(bool(row.get("pose_detected")) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "frontal_max_abs_yaw": float(frontal_max_abs_yaw),
            "profile_min_abs_yaw": float(profile_min_abs_yaw),
            "mean_csim": _safe_mean(scores),
            "p05_csim": _safe_percentile(scores, 5),
            "min_csim": min(detected) if detected else float("nan"),
            "drift_slope": _linear_slope(scores),
            "late_drop": early - late if not math.isnan(early) and not math.isnan(late) else float("nan"),
            "threshold_mode": threshold_mode,
            "same_threshold_csim": same_threshold,
            "different_threshold_csim": different_threshold,
            "same_person_p05": id_ruler.get("same_person_p05", float("nan")),
            "different_people_p95": id_ruler.get("different_people_p95", float("nan")),
            "classification_counts": classification_counts,
            "primary_identity_metric": "evaluation_window.same_person_rate_strict",
            "evaluation_window": evaluation_window,
            "legacy_pose_controlled_metric": "frontal_p05_csim",
            "primary_pose_controlled_metric": "frontal_p05_csim",
            "frontal_frame_count": frontal_summary["frame_count"],
            "frontal_mean_csim": frontal_summary["mean_csim"],
            "frontal_p05_csim": frontal_summary["p05_csim"],
            "frontal_min_csim": frontal_summary["min_csim"],
            "oblique_frame_count": oblique_summary["frame_count"],
            "oblique_mean_csim": oblique_summary["mean_csim"],
            "oblique_p05_csim": oblique_summary["p05_csim"],
            "profile_frame_count": profile_summary["frame_count"],
            "profile_mean_csim": profile_summary["mean_csim"],
            "profile_p05_csim": profile_summary["p05_csim"],
            "pose_penalty_oblique": (
                frontal_summary["mean_csim"] - oblique_summary["mean_csim"]
                if not math.isnan(frontal_summary["mean_csim"])
                and not math.isnan(oblique_summary["mean_csim"])
                else float("nan")
            ),
            "pose_penalty_profile": (
                frontal_summary["mean_csim"] - profile_summary["mean_csim"]
                if not math.isnan(frontal_summary["mean_csim"])
                and not math.isnan(profile_summary["mean_csim"])
                else float("nan")
            ),
            "pose_summary": {
                "frontal": frontal_summary,
                "oblique": oblique_summary,
                "profile": profile_summary,
                "undetected": pose_undetected_summary,
            },
            "pose_recovery": recovery_summary,
            "saliency_sampling": {
                "mode": representative_saliency,
                "interval_seconds": float(saliency_interval_seconds),
                "max_samples": int(saliency_max_samples),
                "truncated": samples_truncated,
                "sampled_frame_indices": sample_indices,
            },
            "sampled_frames": [rows[index] for index in sample_indices],
            "per_frame": rows,
        }
        if json_output_path:
            path = _video_json_path(
                json_output_path,
                analyzed_video_name,
                json_append_video_name,
                evaluation_tag,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            video_results["json_output_path"] = str(path)
            path.write_text(json.dumps(video_results, ensure_ascii=False, indent=2), encoding="utf-8")
        drift_chart = _video_drift_chart(rows, video_results, id_ruler, fps)

        if representative_saliency == "off" or not scores:
            saliency = _placeholder("Video Saliency", "off")
        else:
            saliency = _video_saliency_gallery(
                array,
                rows,
                sample_indices,
                embedder,
                prototype,
                saliency_patch,
                saliency_stride,
                fps,
                samples_truncated,
            )
        return video_results, _pil_to_tensor(drift_chart), saliency, backend


COLOR_STABLE = "色安定"
COLOR_CAUTION = "色変化注意"
COLOR_UNSTABLE = "色不安定"
COLOR_HOLD = "色評価保留"
COLOR_SKIN_REGIONS = {
    "forehead": (34, 22, 78, 42),
    "left_cheek": (16, 60, 46, 84),
    "right_cheek": (66, 60, 96, 84),
}


def _aligned_color_face(detector, pil_image: Image.Image):
    from insightface.utils import face_align

    cv2 = detector.cv2
    rgb = np.asarray(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    faces = detector.app.get(bgr)
    if not faces:
        return None, {
            "pose_detected": False,
            "pitch": float("nan"),
            "yaw": float("nan"),
            "roll": float("nan"),
        }
    face = max(
        faces,
        key=lambda item: float(
            (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
        ),
    )
    aligned_bgr = face_align.norm_crop(bgr, landmark=face.kps, image_size=112)
    raw_pose = getattr(face, "pose", None)
    if raw_pose is not None and len(raw_pose) >= 3:
        pitch, yaw, roll = (float(raw_pose[0]), float(raw_pose[1]), float(raw_pose[2]))
        pose_detected = True
    else:
        pitch = yaw = roll = float("nan")
        pose_detected = False
    return cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB), {
        "pose_detected": pose_detected,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }


def _skin_patch_labs(aligned_rgb: np.ndarray, minimum_skin_ratio: float = 0.20):
    import cv2
    from skimage.color import rgb2lab

    ycrcb = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2YCrCb)
    full_mask = np.zeros(aligned_rgb.shape[:2], dtype=bool)
    labs = {}
    validity = {}
    for name, (x1, y1, x2, y2) in COLOR_SKIN_REGIONS.items():
        patch_ycrcb = ycrcb[y1:y2, x1:x2]
        candidate = (
            (patch_ycrcb[..., 1] >= 128)
            & (patch_ycrcb[..., 1] <= 182)
            & (patch_ycrcb[..., 2] >= 72)
            & (patch_ycrcb[..., 2] <= 138)
        )
        ratio = float(np.mean(candidate)) if candidate.size else 0.0
        validity[name] = ratio
        patch_rgb = aligned_rgb[y1:y2, x1:x2]
        if ratio >= minimum_skin_ratio:
            full_mask[y1:y2, x1:x2] |= candidate
            lab_patch = rgb2lab(patch_rgb.astype(np.float32) / 255.0)
            values = lab_patch[candidate]
            if values.size:
                labs[name] = np.median(values, axis=0).astype(np.float32)
    return labs, full_mask, validity


def _patch_distances(current: Dict, reference: Dict, region_names: Optional[Sequence[str]] = None):
    from skimage.color import deltaE_ciede2000

    delta_e = []
    delta_l = []
    delta_chroma = []
    common = set(current) & set(reference)
    if region_names is not None:
        common &= set(region_names)
    for name in sorted(common):
        value = np.asarray(current[name], dtype=np.float32)
        base = np.asarray(reference[name], dtype=np.float32)
        delta_e.append(float(deltaE_ciede2000(value, base)))
        delta_l.append(float(value[0] - base[0]))
        delta_chroma.append(float(np.linalg.norm(value[1:3] - base[1:3])))
    if not delta_e:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(np.median(delta_e)),
        float(np.max(delta_e)),
        float(np.median(delta_l)),
        float(np.median(delta_chroma)),
    )


def _color_classification(delta_e: float, stable: float, unstable: float, valid: bool = True) -> str:
    if not valid:
        return COLOR_HOLD
    if math.isnan(delta_e):
        return "顔検出失敗"
    if delta_e <= stable:
        return COLOR_STABLE
    if delta_e >= unstable:
        return COLOR_UNSTABLE
    return COLOR_CAUTION


def _color_regions_for_yaw(
    yaw: float, frontal_max_abs_yaw: float, profile_hold_abs_yaw: float
) -> List[str]:
    """Use both cheeks frontally and only the camera-visible cheek when oblique."""
    if math.isnan(yaw) or abs(yaw) >= profile_hold_abs_yaw:
        return []
    if abs(yaw) <= frontal_max_abs_yaw:
        return ["left_cheek", "right_cheek"]
    if yaw > 0.0:
        return ["left_cheek"]
    if yaw < 0.0:
        return ["right_cheek"]
    return ["left_cheek", "right_cheek"]


def _color_metric_summary(rows: Sequence[Dict], pose_class: Optional[str] = None) -> Dict:
    selected = [
        row
        for row in rows
        if row.get("color_valid")
        and (pose_class is None or row.get("pose_class") == pose_class)
    ]
    values = [row.get("skin_delta_e_temporal", float("nan")) for row in selected]
    flicker = [row.get("flicker_delta_e", float("nan")) for row in selected]
    return {
        "frame_count": len(selected),
        "mean_delta_e": _safe_mean(values),
        "p95_delta_e": _safe_percentile(values, 95),
        "max_delta_e": max([v for v in values if not math.isnan(v)], default=float("nan")),
        "flicker_p95": _safe_percentile(flicker, 95),
    }


def _time_slope(rows: Sequence[Dict], value_key: str, pose_class: Optional[str] = None) -> float:
    points = [
        (float(row.get("time_seconds", 0.0)), float(row.get(value_key, float("nan"))))
        for row in rows
        if row.get("color_valid")
        and (pose_class is None or row.get("pose_class") == pose_class)
    ]
    points = [(x, y) for x, y in points if not math.isnan(y)]
    if len(points) < 2 or abs(points[-1][0] - points[0][0]) < 1e-9:
        return float("nan")
    slope, _ = np.polyfit(
        np.asarray([point[0] for point in points], dtype=np.float32),
        np.asarray([point[1] for point in points], dtype=np.float32),
        1,
    )
    return float(slope)


def _color_drift_chart(rows: Sequence[Dict], summary: Dict) -> Image.Image:
    width, height = 1280, 910
    image = Image.new("RGB", (width, height), (246, 247, 248))
    draw = ImageDraw.Draw(image)
    draw.text((42, 24), "Pose-aware Face Color Difference", fill=(24, 28, 32), font=_font(34))
    left, top, right, bottom = 100, 105, 1180, 505
    values = [row.get("skin_delta_e_temporal", float("nan")) for row in rows]
    stable = float(summary["stable_delta_e"])
    unstable = float(summary["unstable_delta_e"])
    clean = [value for value in values if not math.isnan(value)]
    maximum = max(clean + [unstable]) if clean else unstable
    maximum = max(1.0, maximum * 1.18)

    def y_for(value):
        return bottom - int(value / maximum * (bottom - top))

    stable_y = y_for(stable)
    unstable_y = y_for(unstable)
    draw.rectangle((left, top, right, unstable_y), fill=(248, 229, 229))
    draw.rectangle((left, unstable_y, right, stable_y), fill=(250, 242, 220))
    draw.rectangle((left, stable_y, right, bottom), fill=(227, 242, 232))
    draw.line((left, stable_y, right, stable_y), fill=(42, 126, 74), width=3)
    draw.line((left, unstable_y, right, unstable_y), fill=(180, 58, 58), width=3)
    draw.text((left + 8, stable_y + 5), f"色安定 <= ΔE00 {stable:.1f}", fill=(35, 105, 62), font=_font(17))
    draw.text((left + 8, unstable_y - 27), f"色不安定 >= ΔE00 {unstable:.1f}", fill=(145, 44, 44), font=_font(17))

    points = []
    colors = {
        COLOR_STABLE: (42, 126, 74),
        COLOR_CAUTION: (177, 119, 37),
        COLOR_UNSTABLE: (180, 58, 58),
        COLOR_HOLD: (112, 116, 122),
    }
    for index, (value, row) in enumerate(zip(values, rows)):
        if math.isnan(value):
            continue
        x = left + int(index / max(1, len(values) - 1) * (right - left))
        y = y_for(value)
        points.append((x, y, row, index))
    for first, second in zip(points, points[1:]):
        x1, y1, row1, index1 = first
        x2, y2, row2, index2 = second
        if index2 == index1 + 1 and row1.get("color_valid") and row2.get("color_valid"):
            draw.line((x1, y1, x2, y2), fill=(52, 101, 164), width=4)
    for x, y, row, _ in points:
        color = colors.get(row.get("color_classification"), (95, 99, 104))
        if row.get("color_valid"):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        else:
            draw.line((x - 5, y - 5, x + 5, y + 5), fill=color, width=2)
            draw.line((x - 5, y + 5, x + 5, y - 5), fill=color, width=2)
    draw.rectangle((left, top, right, bottom), outline=(80, 84, 89), width=2)
    duration = float(summary.get("duration_seconds", 0.0))
    draw.text((left, bottom + 12), "0.0s", fill=(70, 74, 80), font=_font(16))
    draw.text((right - 75, bottom + 12), f"{duration:.1f}s", fill=(70, 74, 80), font=_font(16))
    # Lower panel: yaw versus color difference. It exposes pose-correlated false alarms.
    scatter_top, scatter_bottom = 585, 760
    draw.rectangle((left, scatter_top, right, scatter_bottom), outline=(80, 84, 89), width=2)
    frontal_limit = float(summary.get("frontal_max_abs_yaw", 15.0))
    profile_limit = float(summary.get("profile_hold_abs_yaw", 35.0))
    for limit, color, label in (
        (frontal_limit, (42, 126, 74), "frontal"),
        (profile_limit, (177, 119, 37), "profile hold"),
    ):
        x = left + int(min(90.0, limit) / 90.0 * (right - left))
        draw.line((x, scatter_top, x, scatter_bottom), fill=color, width=2)
        draw.text((x + 5, scatter_top + 5), f"{label} {limit:.0f}°", fill=color, font=_font(15))
    for row in rows:
        yaw = float(row.get("abs_yaw", float("nan")))
        value = float(row.get("skin_delta_e_temporal", float("nan")))
        if math.isnan(yaw) or math.isnan(value):
            continue
        x = left + int(min(90.0, yaw) / 90.0 * (right - left))
        y = scatter_bottom - int(min(maximum, value) / maximum * (scatter_bottom - scatter_top))
        color = colors.get(row.get("color_classification"), (112, 116, 122))
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    draw.text((left, scatter_bottom + 8), "abs yaw 0°", fill=(70, 74, 80), font=_font(15))
    draw.text((right - 90, scatter_bottom + 8), "90°", fill=(70, 74, 80), font=_font(15))
    draw.text((left, scatter_top - 25), "Yaw vs ΔE00", fill=(55, 60, 66), font=_font(18))
    draw.text(
        (right - 390, scatter_top - 25),
        "● 評価対象 / × 色評価保留",
        fill=(75, 79, 84),
        font=_font(16),
    )

    labels = [
        f"frontal p95: {summary.get('frontal_color_p95_delta_e', float('nan')):.3f}",
        f"frontal mean: {summary.get('frontal_color_mean_delta_e', float('nan')):.3f}",
        f"valid: {summary.get('color_valid_rate', float('nan')):.1%}",
        f"hold: {summary.get('color_hold_rate', float('nan')):.1%}",
        f"ref QC p95: {summary.get('reference_self_delta_e_p95', float('nan')):.3f}",
    ]
    for index, label in enumerate(labels):
        draw.text((52 + index * 245, 818), label, fill=(55, 60, 66), font=_font(17))
    sync = summary.get("sampling", {})
    draw.text(
        (52, 862),
        f"sampling: {sync.get('source', 'manual')} / {len(sync.get('sampled_frame_indices', []))} frames",
        fill=(55, 60, 66),
        font=_font(16),
    )
    return image


def _color_gallery(
    aligned_faces: Sequence[Optional[np.ndarray]],
    rows: Sequence[Dict],
    indices: Sequence[int],
    fps: float,
    show_mask: bool,
    masks: Sequence[Optional[np.ndarray]],
):
    if not indices:
        return _placeholder("Color samples", "No sampled frames")
    card_w, card_h = 330, 445
    columns = min(4, len(indices))
    row_count = math.ceil(len(indices) / columns)
    image = Image.new("RGB", (columns * card_w, row_count * card_h + 54), (238, 240, 242))
    draw = ImageDraw.Draw(image)
    title = "Measured skin regions" if show_mask else "Color evaluation samples"
    draw.text((22, 12), title, fill=(38, 42, 46), font=_font(23))
    colors = {
        COLOR_STABLE: (42, 126, 74),
        COLOR_CAUTION: (177, 119, 37),
        COLOR_UNSTABLE: (180, 58, 58),
        COLOR_HOLD: (110, 110, 115),
        "顔検出失敗": (110, 110, 115),
    }
    for card_index, frame_index in enumerate(indices):
        x = card_index % columns * card_w
        y = 54 + card_index // columns * card_h
        face = aligned_faces[frame_index]
        if face is None:
            panel = Image.new("RGB", (280, 280), (225, 227, 230))
            ImageDraw.Draw(panel).text((52, 126), "Face not detected", fill=(90, 94, 98), font=_font(18))
        else:
            display = face.copy()
            if show_mask and masks[frame_index] is not None:
                mask = masks[frame_index]
                tint = np.zeros_like(display)
                tint[..., 1] = 235
                display = np.where(mask[..., None], (display * 0.45 + tint * 0.55).astype(np.uint8), display)
            panel = Image.fromarray(display, "RGB").resize((280, 280), Image.Resampling.NEAREST)
        image.paste(panel, (x + 25, y + 6))
        row = rows[frame_index]
        label = row["color_classification"]
        draw.text((x + 20, y + 296), label, fill=colors.get(label, (70, 70, 70)), font=_font(22))
        value = row.get("skin_delta_e_temporal", float("nan"))
        value_text = "nan" if math.isnan(value) else f"{value:.3f}"
        draw.text((x + 20, y + 328), f"ΔE00 {value_text}", fill=(42, 46, 50), font=_font(18))
        draw.text((x + 20, y + 356), f"t={frame_index / max(0.001, fps):.2f}s / frame {frame_index}", fill=(70, 74, 80), font=_font(17))
        yaw = row.get("yaw", float("nan"))
        yaw_text = "nan" if math.isnan(yaw) else f"{yaw:+.1f} deg"
        draw.text((x + 20, y + 384), f"pose: {row.get('pose_class', '未評価')} / yaw {yaw_text}", fill=(70, 74, 80), font=_font(16))
        detail = row.get("color_hold_reason") or ",".join(row.get("active_skin_regions", []))
        draw.text((x + 20, y + 410), detail[:34], fill=(70, 74, 80), font=_font(15))
    return _pil_to_tensor(image)


def _color_evaluation_window(
    rows: Sequence[Dict],
    start_seconds: float,
    end_seconds: float,
    require_same_person: bool,
) -> Dict:
    if end_seconds <= start_seconds:
        raise ValueError("evaluation_end_seconds must be greater than evaluation_start_seconds.")
    selected = [
        row for row in rows
        if float(start_seconds) <= float(row.get("time_seconds", 0.0)) < float(end_seconds)
    ]
    same_person = [row for row in selected if row.get("id_same_person")]
    eligible = [
        row for row in selected
        if row.get("color_valid")
        and (not require_same_person or row.get("id_same_person"))
    ]
    frontal = [row for row in eligible if row.get("pose_class") == "正面"]
    all_values = [float(row.get("skin_delta_e_temporal", float("nan"))) for row in eligible]
    frontal_values = [float(row.get("skin_delta_e_temporal", float("nan"))) for row in frontal]
    return {
        "start_seconds": float(start_seconds),
        "end_seconds": float(end_seconds),
        "frame_count": len(selected),
        "require_same_person": bool(require_same_person),
        "same_person_frame_count": len(same_person),
        "same_person_rate_strict": len(same_person) / len(selected) if selected else 0.0,
        "color_valid_frame_count": len(eligible),
        "color_valid_rate_strict": len(eligible) / len(selected) if selected else 0.0,
        "frontal_color_valid_frame_count": len(frontal),
        "mean_delta_e": _safe_mean(all_values),
        "p95_delta_e": _safe_percentile(all_values, 95),
        "frontal_mean_delta_e": _safe_mean(frontal_values),
        "frontal_p95_delta_e": _safe_percentile(frontal_values, 95),
        "color_evaluation_available": bool(frontal_values),
    }


class RogoAIColorEvalVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "sampling_source": (["sync_id_eval", "manual_interval"], {"default": "sync_id_eval"}),
                "frames_per_second": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0, "step": 0.1}),
                "sample_interval_seconds": ("FLOAT", {"default": 5.0, "min": 0.25, "max": 600.0, "step": 0.25}),
                "max_samples": ("INT", {"default": 12, "min": 1, "max": 240}),
                "reference_mode": (["first_stable_frames", "external_reference"], {"default": "first_stable_frames"}),
                "reference_start_seconds": ("FLOAT", {"default": 0.125, "min": 0.0, "max": 30.0, "step": 0.125}),
                "reference_window_seconds": ("FLOAT", {"default": 0.5, "min": 0.125, "max": 30.0, "step": 0.125}),
                "stable_delta_e": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "unstable_delta_e": ("FLOAT", {"default": 6.0, "min": 0.2, "max": 40.0, "step": 0.1}),
                "json_output_path": ("STRING", {"default": _comfy_output_path("RogoAI_IDeval", "color_video_results.json")}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "pose_aware_color": (["on", "off"], {"default": "on"}),
                "frontal_max_abs_yaw": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "profile_hold_abs_yaw": ("FLOAT", {"default": 35.0, "min": 5.0, "max": 89.0, "step": 1.0}),
                "minimum_patch_skin_ratio": ("FLOAT", {"default": 0.20, "min": 0.05, "max": 0.90, "step": 0.05}),
                "json_append_video_name": (["on", "off"], {"default": "on"}),
                "evaluation_start_seconds": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 3600.0, "step": 0.125}),
                "evaluation_end_seconds": ("FLOAT", {"default": 30.0, "min": 0.125, "max": 3600.0, "step": 0.125}),
                "require_same_person": (["on", "off"], {"default": "on"}),
            },
            "optional": {
                "id_video_results": ("ID_VIDEO_RESULTS",),
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("COLOR_VIDEO_RESULTS", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("color_results", "color_drift_chart", "representative_color_gallery", "skin_mask_gallery")
    FUNCTION = "evaluate"
    CATEGORY = CATEGORY

    def evaluate(
        self,
        frames,
        sampling_source,
        frames_per_second,
        sample_interval_seconds,
        max_samples,
        reference_mode,
        reference_start_seconds,
        reference_window_seconds,
        stable_delta_e,
        unstable_delta_e,
        json_output_path,
        device,
        pose_aware_color,
        frontal_max_abs_yaw,
        profile_hold_abs_yaw,
        minimum_patch_skin_ratio,
        json_append_video_name="on",
        evaluation_start_seconds=20.0,
        evaluation_end_seconds=30.0,
        require_same_person="on",
        id_video_results=None,
        reference_image=None,
    ):
        if unstable_delta_e <= stable_delta_e:
            raise ValueError("unstable_delta_e must be greater than stable_delta_e.")
        if profile_hold_abs_yaw <= frontal_max_abs_yaw:
            raise ValueError("profile_hold_abs_yaw must be greater than frontal_max_abs_yaw.")
        array = _to_numpy_image(frames)
        if not len(array):
            raise RuntimeError("Color Eval Videoにフレームがありません。")
        fps = max(0.001, float(frames_per_second))
        sync_indices = []
        sync_truncated = False
        exact_id_sampling = False
        if sampling_source == "sync_id_eval":
            if id_video_results is None:
                raise ValueError("sync_id_evalにはID Eval Videoのvideo_resultsを接続してください。")
            if int(id_video_results.get("frame_count", -1)) != len(array):
                raise ValueError(
                    f"Frame count mismatch: color={len(array)}, id={id_video_results.get('frame_count')}"
                )
            fps = max(0.001, float(id_video_results.get("frames_per_second", fps)))
            sampling = id_video_results.get("saliency_sampling", {})
            sync_indices = [
                int(index)
                for index in sampling.get("sampled_frame_indices", [])
                if 0 <= int(index) < len(array)
            ]
            exact_id_sampling = bool(sync_indices)
            sync_truncated = bool(sampling.get("truncated", False))
            if not sync_indices:
                interval = float(sampling.get("interval_seconds", sample_interval_seconds))
                sync_indices, sync_truncated = _video_sample_indices(
                    "interval_seconds", [0.0] * len(array), fps, interval, int(max_samples)
                )
        else:
            sync_indices, sync_truncated = _video_sample_indices(
                "interval_seconds",
                [0.0] * len(array),
                fps,
                float(sample_interval_seconds),
                int(max_samples),
            )

        id_pose_by_frame = {}
        id_same_threshold = float("nan")
        if id_video_results is not None:
            id_pose_by_frame = {
                int(row.get("frame", index)): row
                for index, row in enumerate(id_video_results.get("per_frame", []))
            }
            id_same_threshold = float(id_video_results.get("same_threshold_csim", float("nan")))
        if require_same_person == "on" and id_video_results is None:
            raise ValueError("require_same_person=onにはID Eval Videoのvideo_resultsを接続してください。")

        detector = _make_embedder("insightface_buffalo_l", device)
        aligned_faces = []
        patch_labs = []
        masks = []
        validities = []
        pose_rows = []
        for index, frame in enumerate(array):
            aligned, detected_pose = _aligned_color_face(detector, _pil_from_comfy_frame(frame))
            aligned_faces.append(aligned)
            synced_pose = id_pose_by_frame.get(index, {})
            if synced_pose.get("pose_detected"):
                pose = {
                    "pose_detected": True,
                    "pitch": float(synced_pose.get("pitch", float("nan"))),
                    "yaw": float(synced_pose.get("yaw", float("nan"))),
                    "roll": float(synced_pose.get("roll", float("nan"))),
                    "pose_source": "id_eval",
                }
            else:
                pose = dict(detected_pose)
                pose["pose_source"] = "color_detector"
            pose["abs_yaw"] = abs(pose["yaw"]) if not math.isnan(pose["yaw"]) else float("nan")
            pose["pose_class"] = _pose_class(
                pose["yaw"], float(frontal_max_abs_yaw), float(profile_hold_abs_yaw)
            )
            pose_rows.append(pose)
            if aligned is None:
                patch_labs.append(None)
                masks.append(None)
                validities.append({})
            else:
                labs, mask, validity = _skin_patch_labs(
                    aligned, float(minimum_patch_skin_ratio)
                )
                patch_labs.append(labs)
                masks.append(mask)
                validities.append(validity)

        if reference_mode == "external_reference":
            if reference_image is None:
                raise ValueError("external_referenceにはreference_imageを接続してください。")
            reference_frames = _to_numpy_image(reference_image)
            reference_face, reference_pose = _aligned_color_face(
                detector, _pil_from_comfy_frame(reference_frames[0])
            )
            if reference_face is None:
                raise RuntimeError("reference_imageから顔を検出できません。")
            if (
                pose_aware_color == "on"
                and (
                    not reference_pose.get("pose_detected")
                    or abs(float(reference_pose.get("yaw", float("nan")))) > float(frontal_max_abs_yaw)
                )
            ):
                raise RuntimeError("reference_imageは正面顔を使用してください。")
            reference_labs, _, reference_validity = _skin_patch_labs(
                reference_face, float(minimum_patch_skin_ratio)
            )
            if not all(name in reference_labs for name in ("left_cheek", "right_cheek")):
                raise RuntimeError("reference_imageの左右頬から有効な肌色を取得できません。")
            reference_indices = []
            reference_fallback_used = False
            reference_self_values = [0.0]
        else:
            start = min(len(array) - 1, max(0, int(round(reference_start_seconds * fps))))
            end = min(len(array), max(start + 1, int(round((reference_start_seconds + reference_window_seconds) * fps))))
            def is_reference_candidate(index):
                labs = patch_labs[index]
                pose = pose_rows[index]
                return (
                    labs is not None
                    and all(name in labs for name in ("left_cheek", "right_cheek"))
                    and (
                        pose_aware_color == "off"
                        or (
                            pose.get("pose_detected")
                            and pose.get("pose_class") == "正面"
                        )
                    )
                )

            reference_indices = [index for index in range(start, end) if is_reference_candidate(index)]
            reference_fallback_used = False
            if not reference_indices:
                reference_indices = [index for index in range(len(array)) if is_reference_candidate(index)][:8]
                reference_fallback_used = True
            if not reference_indices:
                raise RuntimeError("正面かつ肌マスク有効な基準色フレームがありません。")
            reference_labs = {}
            for name in ("left_cheek", "right_cheek"):
                values = [patch_labs[index][name] for index in reference_indices if name in patch_labs[index]]
                if values:
                    reference_labs[name] = np.median(np.stack(values), axis=0).astype(np.float32)
            reference_self_values = [
                _patch_distances(
                    patch_labs[index], reference_labs, ("left_cheek", "right_cheek")
                )[0]
                for index in reference_indices
            ]

        rows = []
        previous_labs = None
        previous_regions = None
        for index, labs in enumerate(patch_labs):
            pose = pose_rows[index]
            yaw = float(pose.get("yaw", float("nan")))
            if pose_aware_color == "off":
                active_regions = ["left_cheek", "right_cheek"]
                pose_valid = True
                hold_reason = ""
            elif not pose.get("pose_detected"):
                active_regions = []
                pose_valid = False
                hold_reason = "姿勢検出不能"
            elif abs(yaw) >= float(profile_hold_abs_yaw):
                active_regions = []
                pose_valid = False
                hold_reason = "横顔のため保留"
            else:
                active_regions = _color_regions_for_yaw(
                    yaw, float(frontal_max_abs_yaw), float(profile_hold_abs_yaw)
                )
                pose_valid = True
                hold_reason = ""

            available_regions = [
                name for name in active_regions if labs is not None and name in labs
            ]
            required_region_count = 2 if pose.get("pose_class") == "正面" else 1
            mask_valid = len(available_regions) >= required_region_count
            if labs is None:
                hold_reason = "顔検出失敗"
            elif pose_valid and not mask_valid:
                hold_reason = "肌マスク不足"
            color_valid = bool(labs is not None and pose_valid and mask_valid)

            if labs is None:
                delta_e = delta_e_max = delta_l = delta_chroma = flicker = float("nan")
                previous_labs = None
                previous_regions = None
            else:
                diagnostic_regions = available_regions or [
                    name for name in ("left_cheek", "right_cheek") if name in labs
                ]
                delta_e, delta_e_max, delta_l, delta_chroma = _patch_distances(
                    labs, reference_labs, diagnostic_regions
                )
                if color_valid and previous_labs is not None and previous_regions:
                    flicker_regions = sorted(set(available_regions) & set(previous_regions))
                    flicker = _patch_distances(labs, previous_labs, flicker_regions)[0]
                else:
                    flicker = 0.0 if color_valid else float("nan")
                if color_valid:
                    previous_labs = labs
                    previous_regions = available_regions
                else:
                    previous_labs = None
                    previous_regions = None
            validity_values = [validities[index].get(name, 0.0) for name in active_regions]
            mask_valid = float(np.mean(validity_values)) if validity_values else 0.0
            if masks[index] is not None:
                visible_mask = np.zeros_like(masks[index], dtype=bool)
                for name in available_regions:
                    x1, y1, x2, y2 = COLOR_SKIN_REGIONS[name]
                    visible_mask[y1:y2, x1:x2] = masks[index][y1:y2, x1:x2]
                masks[index] = visible_mask
            id_row = id_pose_by_frame.get(index, {})
            id_csim = float(id_row.get("csim", float("nan")))
            id_same_person = bool(
                id_row.get("face_detected")
                and not math.isnan(id_csim)
                and not math.isnan(id_same_threshold)
                and id_csim >= id_same_threshold
            )
            rows.append(
                {
                    "frame": index,
                    "time_seconds": index / fps,
                    "face_detected": labs is not None,
                    "id_face_detected": bool(id_row.get("face_detected")),
                    "id_csim": id_csim,
                    "id_classification": id_row.get("classification", "未接続"),
                    "id_same_person": id_same_person,
                    "skin_delta_e_temporal": delta_e,
                    "skin_delta_e_patch_max": delta_e_max,
                    "lightness_shift": delta_l,
                    "chroma_drift": delta_chroma,
                    "flicker_delta_e": flicker,
                    "skin_mask_valid_rate": mask_valid,
                    "color_valid": color_valid,
                    "color_hold_reason": hold_reason,
                    "active_skin_regions": available_regions,
                    "excluded_skin_regions": [
                        name for name in ("left_cheek", "right_cheek") if name not in available_regions
                    ],
                    "pose_detected": pose.get("pose_detected", False),
                    "pose_source": pose.get("pose_source"),
                    "pitch": pose.get("pitch", float("nan")),
                    "yaw": yaw,
                    "roll": pose.get("roll", float("nan")),
                    "abs_yaw": pose.get("abs_yaw", float("nan")),
                    "pose_class": pose.get("pose_class", "検出不能"),
                    "color_classification": _color_classification(
                        delta_e, stable_delta_e, unstable_delta_e, color_valid
                    ),
                }
            )

        delta_values = [
            row["skin_delta_e_temporal"] if row["color_valid"] else float("nan")
            for row in rows
        ]
        flicker_values = [
            row["flicker_delta_e"] if row["color_valid"] else float("nan")
            for row in rows
        ]
        mask_values = [
            row["skin_mask_valid_rate"]
            for row in rows
            if row["face_detected"]
            and row.get("pose_class") != "横顔"
            and (pose_aware_color == "off" or row.get("pose_detected"))
        ]
        detected_delta = [value for value in delta_values if not math.isnan(value)]
        frontal_summary = _color_metric_summary(rows, "正面")
        oblique_summary = _color_metric_summary(rows, "斜め")
        profile_summary = _color_metric_summary(rows, "横顔")
        all_valid_summary = _color_metric_summary(rows)
        hold_reasons = {}
        for row in rows:
            reason = row.get("color_hold_reason")
            if reason:
                hold_reasons[reason] = hold_reasons.get(reason, 0) + 1
        color_valid_count = sum(1 for row in rows if row.get("color_valid"))
        reference_qc_p95 = _safe_percentile(reference_self_values, 95)
        duration = (len(array) - 1) / fps
        early_end = duration / 3.0
        late_start = duration * 2.0 / 3.0
        early_frontal = [
            row["skin_delta_e_temporal"]
            for row in rows
            if row.get("color_valid")
            and row.get("pose_class") == "正面"
            and row["time_seconds"] <= early_end
        ]
        late_frontal = [
            row["skin_delta_e_temporal"]
            for row in rows
            if row.get("color_valid")
            and row.get("pose_class") == "正面"
            and row["time_seconds"] >= late_start
        ]
        early_frontal_mean = _safe_mean(early_frontal)
        late_frontal_mean = _safe_mean(late_frontal)
        evaluation_window = _color_evaluation_window(
            rows,
            float(evaluation_start_seconds),
            float(evaluation_end_seconds),
            require_same_person == "on",
        )
        analyzed_video_name = str(
            (id_video_results or {}).get("analyzed_video_name", "unknown_video")
        )
        threshold_mode = str(
            (id_video_results or {}).get("threshold_mode", "unknown_mode")
        )
        same_threshold = float(
            (id_video_results or {}).get("same_threshold_csim", float("nan"))
        )
        different_threshold = float(
            (id_video_results or {}).get("different_threshold_csim", float("nan"))
        )
        evaluation_tag = str((id_video_results or {}).get("evaluation_tag", ""))
        if not evaluation_tag:
            evaluation_tag = _evaluation_result_tag(
                threshold_mode,
                same_threshold,
                different_threshold,
                float(evaluation_start_seconds),
                float(evaluation_end_seconds),
            )
        evaluation_id = str(
            (id_video_results or {}).get(
                "evaluation_id",
                f"{_safe_video_stem(analyzed_video_name)}__{evaluation_tag}",
            )
        )
        color_results = {
            "analyzed_video_name": analyzed_video_name,
            "analyzed_video_stem": _safe_video_stem(analyzed_video_name),
            "evaluation_id": evaluation_id,
            "evaluation_tag": evaluation_tag,
            "id_threshold_mode": threshold_mode,
            "id_same_threshold_csim": same_threshold,
            "id_different_threshold_csim": different_threshold,
            "metric": "CIEDE2000",
            "face_detector": "insightface_buffalo_l",
            "frames_per_second": fps,
            "duration_seconds": duration,
            "frame_count": len(array),
            "detected_count": sum(1 for row in rows if row["face_detected"]),
            "face_detect_rate": sum(1 for row in rows if row["face_detected"]) / len(array),
            "reference_mode": reference_mode,
            "reference_frame_indices": reference_indices,
            "reference_frontal_only": pose_aware_color == "on",
            "reference_fallback_used": reference_fallback_used,
            "reference_self_delta_e_p95": reference_qc_p95,
            "reference_qc_status": (
                "良好" if not math.isnan(reference_qc_p95) and reference_qc_p95 <= stable_delta_e
                else "要確認"
            ),
            "stable_delta_e": float(stable_delta_e),
            "unstable_delta_e": float(unstable_delta_e),
            "pose_aware_color": pose_aware_color == "on",
            "frontal_max_abs_yaw": float(frontal_max_abs_yaw),
            "profile_hold_abs_yaw": float(profile_hold_abs_yaw),
            "minimum_patch_skin_ratio": float(minimum_patch_skin_ratio),
            "primary_color_metric": "evaluation_window.frontal_p95_delta_e",
            "evaluation_window": evaluation_window,
            "legacy_primary_color_metric": "frontal_color_p95_delta_e",
            "skin_delta_e_mean": all_valid_summary["mean_delta_e"],
            "skin_delta_e_p95": all_valid_summary["p95_delta_e"],
            "skin_delta_e_max": all_valid_summary["max_delta_e"],
            "frontal_color_frame_count": frontal_summary["frame_count"],
            "frontal_color_mean_delta_e": frontal_summary["mean_delta_e"],
            "frontal_color_p95_delta_e": frontal_summary["p95_delta_e"],
            "frontal_color_max_delta_e": frontal_summary["max_delta_e"],
            "frontal_color_flicker_p95": frontal_summary["flicker_p95"],
            "frontal_early_mean_delta_e": early_frontal_mean,
            "frontal_late_mean_delta_e": late_frontal_mean,
            "frontal_late_delta_e_rise": (
                late_frontal_mean - early_frontal_mean
                if not math.isnan(early_frontal_mean) and not math.isnan(late_frontal_mean)
                else float("nan")
            ),
            "oblique_color_summary": oblique_summary,
            "profile_color_summary": profile_summary,
            "color_drift_slope_per_second": _time_slope(rows, "skin_delta_e_temporal"),
            "frontal_color_drift_slope_per_second": _time_slope(
                rows, "skin_delta_e_temporal", "正面"
            ),
            "flicker_p95": _safe_percentile(flicker_values, 95),
            "lightness_shift_p95_abs": _safe_percentile(
                [abs(row["lightness_shift"]) for row in rows if row["color_valid"]], 95
            ),
            "chroma_drift_p95": _safe_percentile(
                [row["chroma_drift"] for row in rows if row["color_valid"]], 95
            ),
            "skin_mask_valid_rate": _safe_mean(mask_values),
            "color_valid_count": color_valid_count,
            "color_valid_rate": color_valid_count / len(rows),
            "color_hold_count": len(rows) - color_valid_count,
            "color_hold_rate": (len(rows) - color_valid_count) / len(rows),
            "color_hold_reasons": hold_reasons,
            "yaw_delta_e_scatter": [
                {
                    "frame": row["frame"],
                    "abs_yaw": row["abs_yaw"],
                    "delta_e": row["skin_delta_e_temporal"],
                    "color_valid": row["color_valid"],
                }
                for row in rows
                if not math.isnan(row.get("abs_yaw", float("nan")))
                and not math.isnan(row.get("skin_delta_e_temporal", float("nan")))
            ],
            "sampling": {
                "source": sampling_source,
                "synced_to_id_eval": exact_id_sampling,
                "truncated": sync_truncated,
                "sampled_frame_indices": sync_indices,
            },
            "sampled_frames": [rows[index] for index in sync_indices],
            "per_frame": rows,
        }
        if json_output_path:
            path = _video_json_path(
                json_output_path,
                analyzed_video_name,
                json_append_video_name,
                evaluation_tag,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            color_results["json_output_path"] = str(path)
            path.write_text(json.dumps(color_results, ensure_ascii=False, indent=2), encoding="utf-8")
        chart = _color_drift_chart(rows, color_results)
        gallery = _color_gallery(aligned_faces, rows, sync_indices, fps, False, masks)
        mask_gallery = _color_gallery(aligned_faces, rows, sync_indices, fps, True, masks)
        return color_results, _pil_to_tensor(chart), gallery, mask_gallery


def _float_metric(results: Dict, key: str, fallback_key: Optional[str] = None) -> float:
    value = results.get(key)
    if value is None and fallback_key is not None:
        value = results.get(fallback_key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _quality_state_id_value(p05: float, same: float, different: float) -> str:
    if math.isnan(p05) or math.isnan(same) or math.isnan(different):
        return "判定不能"
    if p05 >= same:
        return "ID安定"
    if p05 <= different:
        return "ID不安定"
    return "ID注意"


def _quality_state_id(results: Dict) -> str:
    p05 = _float_metric(results, "frontal_p05_csim", "p05_csim")
    same = _float_metric(results, "same_threshold_csim")
    different = _float_metric(results, "different_threshold_csim")
    return _quality_state_id_value(p05, same, different)


def _quality_state_id_all_pose(results: Dict) -> str:
    return _quality_state_id_value(
        _float_metric(results, "p05_csim"),
        _float_metric(results, "same_threshold_csim"),
        _float_metric(results, "different_threshold_csim"),
    )


def _quality_state_color(results: Dict) -> str:
    p95 = _float_metric(results, "frontal_color_p95_delta_e", "skin_delta_e_p95")
    stable = _float_metric(results, "stable_delta_e")
    unstable = _float_metric(results, "unstable_delta_e")
    if math.isnan(stable):
        stable = 3.0
    if math.isnan(unstable):
        unstable = 6.0
    frontal_count = int(results.get("frontal_color_frame_count", 3))
    reference_qc = _float_metric(results, "reference_self_delta_e_p95")
    if math.isnan(reference_qc):
        reference_qc = 0.0
    if frontal_count < 3 or (not math.isnan(reference_qc) and reference_qc >= unstable):
        return "判定不能"
    if math.isnan(p95):
        return "判定不能"
    if p95 <= stable:
        return COLOR_STABLE
    if p95 >= unstable:
        return COLOR_UNSTABLE
    return COLOR_CAUTION


def _quality_report_image(id_results: Dict, color_results: Dict, combined: Dict) -> Image.Image:
    width, height = 1440, 960
    image = Image.new("RGB", (width, height), (246, 247, 248))
    draw = ImageDraw.Draw(image)
    draw.text((42, 28), "RogoAI Video Quality Report", fill=(24, 28, 32), font=_font(34))
    draw.text(
        (44, 76),
        "主判定: frontal p05 × frontal color p95",
        fill=(55, 60, 66),
        font=_font(20),
    )
    left, top, cell_w, cell_h = 220, 155, 350, 170
    id_states = ["ID安定", "ID注意", "ID不安定"]
    color_states = [COLOR_STABLE, COLOR_CAUTION, COLOR_UNSTABLE]
    id_labels = ["ID安定", "ID注意", "ID不安定"]
    color_labels = ["色安定", "色注意", "色不安定"]
    fills = [
        [(222, 241, 228), (238, 242, 220), (250, 238, 220)],
        [(238, 242, 220), (250, 242, 220), (250, 232, 218)],
        [(250, 238, 220), (250, 232, 218), (247, 225, 225)],
    ]
    primary_id = combined.get("id_state", "判定不能")
    all_id = combined.get("id_all_pose_state", "判定不能")
    color_state = combined.get("color_state", "判定不能")
    primary_cell = (
        (id_states.index(primary_id), color_states.index(color_state))
        if primary_id in id_states and color_state in color_states
        else None
    )
    all_cell = (
        (id_states.index(all_id), color_states.index(color_state))
        if all_id in id_states and color_state in color_states
        else None
    )

    for col, id_label in enumerate(id_labels):
        x = left + col * cell_w + cell_w // 2
        draw.text((x, top - 38), id_label, fill=(55, 60, 66), font=_font(21), anchor="mm")
    for row, color_label in enumerate(color_labels):
        y = top + row * cell_h + cell_h // 2
        draw.text((left - 28, y), color_label, fill=(55, 60, 66), font=_font(20), anchor="rm")

    for row, color_label in enumerate(color_labels):
        for col, id_label in enumerate(id_labels):
            label = f"{id_label}・{color_label}"
            fill = fills[row][col]
            selected = primary_cell == (col, row)
            reference = all_cell == (col, row)
            x1, y1 = left + col * cell_w, top + row * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            draw.rectangle(
                (x1, y1, x2, y2),
                fill=fill,
                outline=(39, 105, 150) if selected else (88, 92, 97),
                width=6 if selected else 2,
            )
            draw.text((x1 + 20, y1 + 112), label, fill=(40, 44, 48), font=_font(21))
            if selected and reference:
                draw.ellipse((x1 + 20, y1 + 20, x1 + 42, y1 + 42), fill=(39, 105, 150))
                draw.ellipse((x1 + 20, y1 + 52, x1 + 42, y1 + 74), outline=(55, 60, 66), width=3)
                draw.text((x1 + 54, y1 + 18), "主判定", fill=(39, 105, 150), font=_font(19))
                draw.text((x1 + 54, y1 + 50), "全姿勢参考", fill=(55, 60, 66), font=_font(18))
            elif selected:
                draw.ellipse((x1 + 20, y1 + 20, x1 + 42, y1 + 42), fill=(39, 105, 150))
                draw.text((x1 + 54, y1 + 18), "主判定", fill=(39, 105, 150), font=_font(19))
            elif reference:
                draw.ellipse((x1 + 20, y1 + 20, x1 + 42, y1 + 42), outline=(55, 60, 66), width=3)
                draw.text((x1 + 54, y1 + 18), "全姿勢参考", fill=(55, 60, 66), font=_font(18))

    if primary_cell is None:
        draw.text((left, top + 3 * cell_h + 12), "主判定: 判定不能", fill=(180, 58, 58), font=_font(20))
    draw.text(
        (80, 700),
        f"ID主判定: {combined['id_state']} / frontal p05 {combined.get('id_primary_value', float('nan')):.4f}",
        fill=(42, 46, 50),
        font=_font(22),
    )
    draw.text(
        (80, 736),
        f"ID全姿勢参考: {combined.get('id_all_pose_state', '判定不能')} / all p05 {combined.get('id_all_pose_value', float('nan')):.4f}",
        fill=(42, 46, 50),
        font=_font(22),
    )
    draw.text(
        (80, 772),
        f"Color主判定: {combined['color_state']} / frontal p95 ΔE00 {combined.get('color_primary_value', float('nan')):.3f}",
        fill=(42, 46, 50),
        font=_font(22),
    )
    draw.text(
        (80, 808),
        f"Color valid {color_results.get('color_valid_rate', float('nan')):.1%} / hold {color_results.get('color_hold_rate', float('nan')):.1%} / reference QC {color_results.get('reference_qc_status', '未評価')}",
        fill=(42, 46, 50),
        font=_font(20),
    )
    sync_text = "同期済み" if combined.get("sampling_synchronized") else "サンプリング不一致"
    draw.text((80, 848), f"sampling: {sync_text}", fill=(42, 126, 74) if combined.get("sampling_synchronized") else (180, 58, 58), font=_font(20))
    draw.text((80, 886), "● 主判定（正面統制）  ○ 全姿勢参考", fill=(55, 60, 66), font=_font(18))
    return image


class RogoAIVideoQualityReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "id_video_results": ("ID_VIDEO_RESULTS",),
                "color_results": ("COLOR_VIDEO_RESULTS",),
            }
        }

    RETURN_TYPES = ("IMAGE", "VIDEO_QUALITY_RESULTS")
    RETURN_NAMES = ("quality_report", "quality_results")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, id_video_results, color_results):
        id_state = _quality_state_id(id_video_results)
        id_all_pose_state = _quality_state_id_all_pose(id_video_results)
        color_state = _quality_state_color(color_results)
        id_indices = id_video_results.get("saliency_sampling", {}).get("sampled_frame_indices", [])
        color_indices = color_results.get("sampling", {}).get("sampled_frame_indices", [])
        synchronized = (
            int(id_video_results.get("frame_count", -1)) == int(color_results.get("frame_count", -2))
            and abs(float(id_video_results.get("frames_per_second", 0.0)) - float(color_results.get("frames_per_second", 1.0))) < 1e-6
            and list(id_indices) == list(color_indices)
        )
        primary_cell = f"{id_state}・{color_state}"
        all_pose_reference_cell = f"{id_all_pose_state}・{color_state}"
        quality_results = {
            "id_state": id_state,
            "id_all_pose_state": id_all_pose_state,
            "color_state": color_state,
            "quality_map": "3x3",
            "primary_cell": primary_cell,
            "all_pose_reference_cell": all_pose_reference_cell,
            "quadrant": primary_cell,
            "id_primary_metric": "frontal_p05_csim",
            "id_reference_metric": "p05_csim",
            "color_primary_metric": "frontal_color_p95_delta_e",
            "id_primary_value": _float_metric(id_video_results, "frontal_p05_csim", "p05_csim"),
            "id_all_pose_value": _float_metric(id_video_results, "p05_csim"),
            "color_primary_value": _float_metric(
                color_results, "frontal_color_p95_delta_e", "skin_delta_e_p95"
            ),
            "sampling_synchronized": synchronized,
            "id_sampled_frame_indices": id_indices,
            "color_sampled_frame_indices": color_indices,
            "id_summary": {
                "mean_csim": id_video_results.get("mean_csim"),
                "p05_csim": id_video_results.get("p05_csim"),
                "frontal_mean_csim": id_video_results.get("frontal_mean_csim"),
                "frontal_p05_csim": id_video_results.get("frontal_p05_csim"),
                "profile_mean_csim": id_video_results.get("profile_mean_csim"),
                "return_to_frontal_mean_csim": id_video_results.get("pose_recovery", {}).get("return_to_frontal_mean_csim"),
                "unreturned_profile_episode_count": id_video_results.get("pose_recovery", {}).get("unreturned_profile_episode_count"),
                "drift_slope": id_video_results.get("drift_slope"),
            },
            "color_summary": {
                "skin_delta_e_mean": color_results.get("skin_delta_e_mean"),
                "skin_delta_e_p95": color_results.get("skin_delta_e_p95"),
                "frontal_color_mean_delta_e": color_results.get("frontal_color_mean_delta_e"),
                "frontal_color_p95_delta_e": color_results.get("frontal_color_p95_delta_e"),
                "frontal_color_max_delta_e": color_results.get("frontal_color_max_delta_e"),
                "frontal_color_drift_slope_per_second": color_results.get("frontal_color_drift_slope_per_second"),
                "frontal_early_mean_delta_e": color_results.get("frontal_early_mean_delta_e"),
                "frontal_late_mean_delta_e": color_results.get("frontal_late_mean_delta_e"),
                "frontal_late_delta_e_rise": color_results.get("frontal_late_delta_e_rise"),
                "flicker_p95": color_results.get("flicker_p95"),
                "frontal_color_flicker_p95": color_results.get("frontal_color_flicker_p95"),
                "color_valid_rate": color_results.get("color_valid_rate"),
                "color_hold_rate": color_results.get("color_hold_rate"),
                "color_hold_reasons": color_results.get("color_hold_reasons"),
                "reference_self_delta_e_p95": color_results.get("reference_self_delta_e_p95"),
                "reference_qc_status": color_results.get("reference_qc_status"),
            },
        }
        return _pil_to_tensor(_quality_report_image(id_video_results, color_results, quality_results)), quality_results


class RogoAIIDBatchReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "results_json_dir": ("STRING", {"default": _comfy_output_path("RogoAI_IDeval")}),
                "backend": ("ID_BACKEND",),
            }
        }

    RETURN_TYPES = ("IMAGE", "ID_BATCH_RESULTS", "ID_BACKEND")
    RETURN_NAMES = ("report_gallery", "batch_results", "backend")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, results_json_dir, backend):
        rows = []
        for path in sorted(Path(results_json_dir).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("backend") != backend or "mean_csim" not in data:
                continue
            rows.append({"name": path.stem, **data})
        rows.sort(key=lambda row: row.get("p05_csim", float("-inf")), reverse=True)
        width, height = 1800, max(520, 150 + 58 * max(1, len(rows)))
        image = Image.new("RGB", (width, height), (246, 247, 248))
        draw = ImageDraw.Draw(image)
        draw.text((42, 30), "RogoAI ID Batch Report", fill=(24, 28, 32), font=_font(34))
        draw.text((42, 82), f"backend: {backend}", fill=(55, 60, 66), font=_font(22))
        headers = ["condition", "mean", "all p05", "frontal p05", "profile mean", "return mean", "unreturned", "detect"]
        xs = [42, 520, 650, 790, 960, 1130, 1310, 1490]
        for x, label in zip(xs, headers):
            draw.text((x, 132), label, fill=(40, 44, 48), font=_font(22))
        for index, row in enumerate(rows):
            y = 180 + index * 58
            values = [
                row["name"][:42],
                f"{row.get('mean_csim', float('nan')):.4f}",
                f"{row.get('p05_csim', float('nan')):.4f}",
                f"{row.get('frontal_p05_csim', float('nan')):.4f}",
                f"{row.get('profile_mean_csim', float('nan')):.4f}",
                f"{row.get('pose_recovery', {}).get('return_to_frontal_mean_csim', float('nan')):.4f}",
                str(row.get('pose_recovery', {}).get('unreturned_profile_episode_count', 0)),
                f"{row.get('face_detect_rate', float('nan')):.1%}",
            ]
            for x, value in zip(xs, values):
                draw.text((x, y), value, fill=(55, 60, 66), font=_font(20))
        batch_results = {"backend": backend, "count": len(rows), "results": rows}
        return _pil_to_tensor(image), batch_results, backend


NODE_CLASS_MAPPINGS = {
    "RogoAIIDReference": RogoAIIDReference,
    "RogoAIIDRuler": RogoAIIDRuler,
    "RogoAIIDImageFolder": RogoAIIDImageFolder,
    "RogoAIIDEvalImage": RogoAIIDEvalImage,
    "RogoAIIDEvalVideo": RogoAIIDEvalVideo,
    "RogoAIColorEvalVideo": RogoAIColorEvalVideo,
    "RogoAIVideoQualityReport": RogoAIVideoQualityReport,
    "RogoAIIDBatchReport": RogoAIIDBatchReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RogoAIIDReference": "RogoAI ID Reference",
    "RogoAIIDRuler": "RogoAI ID Ruler",
    "RogoAIIDImageFolder": "RogoAI ID Image Folder",
    "RogoAIIDEvalImage": "RogoAI ID Eval Image",
    "RogoAIIDEvalVideo": "RogoAI ID Eval Video",
    "RogoAIColorEvalVideo": "RogoAI Color Eval Video",
    "RogoAIVideoQualityReport": "RogoAI Video Quality Report",
    "RogoAIIDBatchReport": "RogoAI ID Batch Report",
}






