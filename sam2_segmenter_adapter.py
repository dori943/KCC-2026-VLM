"""
sam2_segmenter_adapter.py
Optional SAM2-based mask refinement for affordance grasp candidates.

Model:
- Hugging Face: facebook/sam2-hiera-large

Design goals:
- Lazy import heavy deps (torch/transformers/Pillow/numpy)
- Keep main simulation path stable even when SAM2 is unavailable
- Accept R1-style candidate dict and return refined candidate dict
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


def _clamp_int(value: float, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(round(float(value)))))


class SAM2SegmentationAdapter:
    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-large",
        device: Optional[str] = None,
        local_model_dir: Optional[str] = None,
        local_files_only: bool = False,
        trust_remote_code: bool = True,
    ):
        self.model_id = model_id
        self.requested_device = device
        self.device = device or "cpu"
        self.local_model_dir = local_model_dir
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code

        self._torch = None
        self._np = None
        self._PILImage = None
        self._processor_cls = None
        self._model_cls = None

        self._model = None
        self._processor = None
        self._loaded_source = None
        self._last_error = None

    def is_available(self) -> bool:
        return self._model is not None and self._processor is not None

    def load(self) -> None:
        if self.is_available():
            return

        self._last_error = None
        try:
            self._lazy_import_runtime_deps()
        except Exception as exc:
            self._last_error = f"SAM2 dependency import failed: {exc}"
            return

        sources = []
        local_root = self._resolve_local_root()
        if local_root is not None:
            sources.append(("local", str(local_root)))
        sources.append(("hf", self.model_id))

        errors = []
        for source_label, model_ref in sources:
            try:
                kwargs = {"trust_remote_code": self.trust_remote_code}
                if self.local_files_only or source_label == "local":
                    kwargs["local_files_only"] = True

                if self.device.startswith("cuda"):
                    kwargs["torch_dtype"] = self._torch.bfloat16

                model = self._model_cls.from_pretrained(model_ref, **kwargs)
                processor = self._processor_cls.from_pretrained(model_ref, **kwargs)
                if hasattr(model, "eval"):
                    model.eval()
                if hasattr(model, "to"):
                    model = model.to(self.device)

                self._model = model
                self._processor = processor
                self._loaded_source = {
                    "source": source_label,
                    "model_ref": model_ref,
                }
                return
            except Exception as exc:
                errors.append(f"{source_label}:{model_ref} -> {exc}")

        self._last_error = "Failed to load SAM2 model from all candidates.\n" + "\n".join(
            f"- {msg}" for msg in errors
        )

    def refine_candidate(
        self,
        image: Any,
        candidate: dict,
    ) -> dict[str, Any]:
        if not isinstance(candidate, dict):
            return self._failure("candidate must be a dict.")

        if not self.is_available():
            self.load()
        if not self.is_available():
            return self._failure(self._last_error or "SAM2 model is unavailable.")

        try:
            image_pil = self._prepare_image(image)
            image_w, image_h = image_pil.size
            box_xyxy = self._candidate_bbox_to_pixel(candidate, image_size=(image_w, image_h))
            point_xy = self._candidate_point_to_pixel(candidate, box_xyxy, image_size=(image_w, image_h))

            mask, iou_score = self._predict_mask(image_pil=image_pil, box_xyxy=box_xyxy, point_xy=point_xy)
            if mask is None:
                return self._failure("SAM2 returned no valid mask.")

            refined_bbox = self._bbox_from_mask(mask)
            if refined_bbox is None:
                return self._failure("Mask area is empty.")

            refined_center = [
                (refined_bbox[0] + refined_bbox[2]) / 2.0,
                (refined_bbox[1] + refined_bbox[3]) / 2.0,
            ]
            refined_bbox_norm = [
                round(refined_bbox[0] / image_w * 1000.0, 3),
                round(refined_bbox[1] / image_h * 1000.0, 3),
                round(refined_bbox[2] / image_w * 1000.0, 3),
                round(refined_bbox[3] / image_h * 1000.0, 3),
            ]
            refined_center_norm = [
                round(refined_center[0] / image_w * 1000.0, 3),
                round(refined_center[1] / image_h * 1000.0, 3),
            ]

            refined = candidate.copy()
            refined.update(
                {
                    "bbox_pixel": refined_bbox,
                    "bbox_norm": refined_bbox_norm,
                    "center_pixel": refined_center,
                    "center_norm": refined_center_norm,
                    "mask_area_px": int(mask.sum()),
                    "sam2_iou_score": None if iou_score is None else float(iou_score),
                    "refined_by": "sam2-hiera-large",
                }
            )
            if iou_score is not None:
                refined["score"] = max(float(candidate.get("score", 0.0)), float(iou_score))

            return {
                "success": True,
                "model_loaded": True,
                "source": "sam2",
                "model_source": self._loaded_source,
                "candidate": refined,
                "error": None,
            }
        except Exception as exc:
            return self._failure(str(exc))

    def refine_result(
        self,
        image: Any,
        affordance_result: dict,
        top_k: int = 1,
    ) -> dict[str, Any]:
        if not isinstance(affordance_result, dict):
            return self._failure("affordance_result must be a dict.")

        candidates = affordance_result.get("grasp_candidates")
        if not isinstance(candidates, list) or not candidates:
            return self._failure("No grasp candidates to refine.")

        refined_result = dict(affordance_result)
        refined_candidates = []
        refined_count = 0
        target_count = max(1, int(top_k))

        for idx, candidate in enumerate(candidates):
            if idx < target_count:
                refine_output = self.refine_candidate(image=image, candidate=candidate)
                if refine_output.get("success"):
                    refined_candidates.append(refine_output["candidate"])
                    refined_count += 1
                else:
                    refined_candidates.append(candidate)
            else:
                refined_candidates.append(candidate)

        refined_result["grasp_candidates"] = refined_candidates
        base_summary = str(refined_result.get("affordance_summary") or "").strip()
        sam2_summary = f"SAM2 refined {refined_count}/{min(target_count, len(candidates))} candidate(s)."
        refined_result["affordance_summary"] = (
            f"{base_summary} | {sam2_summary}" if base_summary else sam2_summary
        )
        refined_result["sam2_refined"] = refined_count > 0

        return {
            "success": True,
            "model_loaded": self.is_available(),
            "source": "sam2",
            "result": refined_result,
            "error": None,
        }

    def _lazy_import_runtime_deps(self) -> None:
        if self._torch is not None:
            return

        import numpy as np  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
        from PIL import Image  # pylint: disable=import-outside-toplevel

        processor_cls = None
        model_cls = None
        try:
            from transformers import Sam2Model, Sam2Processor  # pylint: disable=import-outside-toplevel

            model_cls = Sam2Model
            processor_cls = Sam2Processor
        except Exception:
            from transformers import AutoModelForMaskGeneration, AutoProcessor  # pylint: disable=import-outside-toplevel

            model_cls = AutoModelForMaskGeneration
            processor_cls = AutoProcessor

        self._torch = torch
        self._np = np
        self._PILImage = Image
        self._model_cls = model_cls
        self._processor_cls = processor_cls
        self.device = self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_local_root(self) -> Optional[Path]:
        candidates = []
        if self.local_model_dir:
            candidates.append(Path(self.local_model_dir).expanduser())
        env_dir = os.getenv("SAM2_LOCAL_DIR")
        if env_dir:
            candidates.append(Path(env_dir).expanduser())
        model_path = Path(self.model_id).expanduser()
        if model_path.exists():
            candidates.append(model_path)

        for path in candidates:
            if path.is_dir():
                return path
        return None

    def _prepare_image(self, image: Any):
        if isinstance(image, self._PILImage.Image):
            return image.convert("RGB")
        if isinstance(image, str):
            return self._PILImage.open(image).convert("RGB")
        if hasattr(image, "shape"):
            arr = self._np.asarray(image)
            if arr.ndim != 3:
                raise ValueError("image array must have shape [H, W, C].")
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != self._np.uint8:
                arr = arr.astype(self._np.uint8)
            return self._PILImage.fromarray(arr)
        raise TypeError("Unsupported image type for SAM2 adapter.")

    def _candidate_bbox_to_pixel(self, candidate: dict, image_size: tuple[int, int]) -> list[int]:
        width, height = image_size

        bbox_pixel = candidate.get("bbox_pixel")
        if (
            isinstance(bbox_pixel, list)
            and len(bbox_pixel) == 4
            and all(isinstance(v, (int, float)) for v in bbox_pixel)
        ):
            x1, y1, x2, y2 = bbox_pixel
        else:
            bbox_norm = candidate.get("bbox_norm")
            if not (
                isinstance(bbox_norm, list)
                and len(bbox_norm) == 4
                and all(isinstance(v, (int, float)) for v in bbox_norm)
            ):
                raise ValueError("candidate has neither valid bbox_pixel nor bbox_norm.")
            x1 = bbox_norm[0] / 1000.0 * width
            y1 = bbox_norm[1] / 1000.0 * height
            x2 = bbox_norm[2] / 1000.0 * width
            y2 = bbox_norm[3] / 1000.0 * height

        x1 = _clamp_int(x1, 0, width - 1)
        y1 = _clamp_int(y1, 0, height - 1)
        x2 = _clamp_int(x2, 0, width - 1)
        y2 = _clamp_int(y2, 0, height - 1)
        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)
        return [x1, y1, x2, y2]

    def _candidate_point_to_pixel(
        self,
        candidate: dict,
        bbox_xyxy: list[int],
        image_size: tuple[int, int],
    ) -> list[int]:
        width, height = image_size
        center_pixel = candidate.get("center_pixel")
        if (
            isinstance(center_pixel, list)
            and len(center_pixel) >= 2
            and all(isinstance(v, (int, float)) for v in center_pixel[:2])
        ):
            cx, cy = center_pixel[0], center_pixel[1]
        else:
            center_norm = candidate.get("center_norm")
            if (
                isinstance(center_norm, list)
                and len(center_norm) >= 2
                and all(isinstance(v, (int, float)) for v in center_norm[:2])
            ):
                cx = center_norm[0] / 1000.0 * width
                cy = center_norm[1] / 1000.0 * height
            else:
                cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
                cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0

        return [_clamp_int(cx, 0, width - 1), _clamp_int(cy, 0, height - 1)]

    def _predict_mask(
        self,
        image_pil,
        box_xyxy: list[int],
        point_xy: list[int],
    ):
        torch = self._torch
        prompt_variants = [
            {
                "input_boxes": [[[float(v) for v in box_xyxy]]],
                "input_points": [[[float(point_xy[0]), float(point_xy[1])]]],
                "input_labels": [[1]],
            },
            {
                "input_boxes": [[[float(v) for v in box_xyxy]]],
            },
        ]

        last_exc = None
        for prompt_kwargs in prompt_variants:
            try:
                inputs = self._processor(
                    images=image_pil,
                    return_tensors="pt",
                    **prompt_kwargs,
                )
                if hasattr(inputs, "to"):
                    inputs = inputs.to(self.device)

                with torch.no_grad():
                    outputs = self._model(**inputs)

                pred_masks = getattr(outputs, "pred_masks", None)
                if pred_masks is None:
                    raise RuntimeError("SAM2 outputs do not contain pred_masks.")

                post_masks = None
                if hasattr(self._processor, "post_process_masks"):
                    post_masks = self._processor.post_process_masks(
                        pred_masks,
                        inputs.get("original_sizes"),
                        inputs.get("reshaped_input_sizes"),
                    )
                elif hasattr(self._processor, "image_processor") and hasattr(
                    self._processor.image_processor, "post_process_masks"
                ):
                    post_masks = self._processor.image_processor.post_process_masks(
                        pred_masks,
                        inputs.get("original_sizes"),
                        inputs.get("reshaped_input_sizes"),
                    )

                if post_masks is None:
                    post_masks = pred_masks

                iou_scores = getattr(outputs, "iou_scores", None)
                mask_np, score = self._extract_best_mask(post_masks, iou_scores)
                if mask_np is None:
                    continue
                return mask_np.astype(bool), score
            except Exception as exc:
                last_exc = exc

        if last_exc is not None:
            raise RuntimeError(f"SAM2 mask prediction failed: {last_exc}")
        return None, None

    def _extract_best_mask(self, masks_obj, iou_scores):
        np = self._np
        torch = self._torch

        if isinstance(masks_obj, (list, tuple)):
            if not masks_obj:
                return None, None
            masks_item = masks_obj[0]
        else:
            masks_item = masks_obj

        if isinstance(masks_item, torch.Tensor):
            arr = masks_item.detach().cpu().numpy()
        else:
            arr = np.asarray(masks_item)

        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3:
            pass
        elif arr.ndim >= 4:
            arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        else:
            return None, None

        scores_flat = None
        if iou_scores is not None:
            if isinstance(iou_scores, torch.Tensor):
                scores_flat = iou_scores.detach().cpu().numpy().reshape(-1)
            else:
                scores_flat = np.asarray(iou_scores).reshape(-1)

        if scores_flat is not None and scores_flat.size == arr.shape[0]:
            best_idx = int(np.argmax(scores_flat))
            best_score = float(scores_flat[best_idx])
        else:
            areas = arr.reshape(arr.shape[0], -1).sum(axis=1)
            best_idx = int(np.argmax(areas))
            best_score = None

        best_mask = arr[best_idx] > 0
        return best_mask, best_score

    def _bbox_from_mask(self, mask) -> Optional[list[int]]:
        np = self._np
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            return None
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max())
        y2 = int(ys.max())
        return [x1, y1, x2, y2]

    def _failure(self, error_message: str) -> dict[str, Any]:
        return {
            "success": False,
            "model_loaded": self.is_available(),
            "source": "sam2",
            "error": error_message,
        }
