"""
affordancegrasp_r1_adapter.py
Optional adapter for Hugging Face hqking/affordance-r1 inference.

Goals:
- Keep module import lightweight.
- Lazy-load heavy dependencies.
- Support repo-root and subfolder model layouts.
- Return structured dict output with graceful fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Optional


class AffordanceGraspR1Adapter:
    """Thin adapter around AffordanceGrasp-R1 style VLM inference."""

    ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
    BBOX_PATTERN = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")

    def __init__(
        self,
        model_id: str = "hqking/affordance-r1",
        device: Optional[str] = None,
        local_model_dir: Optional[str] = None,
        local_files_only: bool = False,
        subfolder_candidates: Optional[tuple[str, ...]] = None,
        allow_snapshot_download: bool = True,
        trust_remote_code: bool = True,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        default_max_new_tokens: int = 256,
    ):
        self.model_id = model_id
        self.requested_device = device
        self.device = device or "cpu"
        self.local_model_dir = local_model_dir
        self.local_files_only = local_files_only
        self.subfolder_candidates = subfolder_candidates or ("huggingface",)
        self.allow_snapshot_download = allow_snapshot_download
        self.trust_remote_code = trust_remote_code
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.default_max_new_tokens = default_max_new_tokens

        self._torch = None
        self._np = None
        self._PILImage = None
        self._processor_cls = None
        self._model_cls = None
        self._vision_info_fn = None

        self._model = None
        self._processor = None
        self._loaded_source: Optional[dict[str, Any]] = None
        self._last_error: Optional[str] = None

    def is_available(self) -> bool:
        return self._model is not None and self._processor is not None

    def load(self) -> None:
        if self.is_available():
            return

        self._last_error = None
        try:
            self._lazy_import_runtime_deps()
        except Exception as exc:
            self._last_error = f"Dependency import failed: {exc}"
            return

        candidate_sources = self._build_source_candidates()
        if not candidate_sources:
            self._last_error = "No valid model source candidate was found."
            return

        errors: list[str] = []
        for source in candidate_sources:
            model_ref = source["model_ref"]
            subfolder = source["subfolder"]
            label = source["label"]
            try:
                model = self._load_model_once(model_ref=model_ref, subfolder=subfolder)
                processor = self._load_processor_once(model_ref=model_ref, subfolder=subfolder)
                self._model = model
                self._processor = processor
                self._loaded_source = source
                return
            except Exception as exc:
                errors.append(f"{label}: {exc}")

                snapshot_dir = self._try_prepare_snapshot_dir(model_ref=model_ref, subfolder=subfolder)
                if snapshot_dir is None:
                    continue
                try:
                    model = self._load_model_once(
                        model_ref=str(snapshot_dir),
                        subfolder=None,
                        force_local_files_only=True,
                    )
                    processor = self._load_processor_once(
                        model_ref=str(snapshot_dir),
                        subfolder=None,
                        force_local_files_only=True,
                    )
                    self._model = model
                    self._processor = processor
                    self._loaded_source = {
                        "model_ref": str(snapshot_dir),
                        "subfolder": None,
                        "label": f"snapshot:{snapshot_dir}",
                    }
                    return
                except Exception as snapshot_exc:
                    errors.append(f"snapshot:{model_ref}/{subfolder or ''}: {snapshot_exc}")

        self._last_error = "Failed to load affordance-r1 from all candidates.\n" + "\n".join(
            f"- {msg}" for msg in errors
        )

    def predict(
        self,
        image: Any,
        prompt: Optional[str] = None,
        extra_context: Optional[dict[str, Any]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        if not self.is_available():
            self.load()

        if not self.is_available():
            return self._failure_result(self._last_error or "Model is not available.")

        try:
            image_pil = self._prepare_image(image)
            output_text = self._run_generation(
                image_pil=image_pil,
                prompt=prompt,
                extra_context=extra_context,
                max_new_tokens=max_new_tokens or self.default_max_new_tokens,
            )
            parsed = self._parse_output(output_text=output_text, image_size=image_pil.size)
            return {
                "success": True,
                "model_loaded": True,
                "source": "affordance-r1",
                "model_source": self._loaded_source,
                "raw_output": output_text,
                "grasp_candidates": parsed["grasp_candidates"],
                "affordance_summary": parsed["summary"],
                "error": None,
            }
        except Exception as exc:
            return self._failure_result(str(exc))

    def predict_from_path(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        extra_context: Optional[dict[str, Any]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        path = Path(image_path).expanduser()
        if not path.is_file():
            return self._failure_result(f"Image path does not exist: {path}")
        return self.predict(
            image=str(path),
            prompt=prompt,
            extra_context=extra_context,
            max_new_tokens=max_new_tokens,
        )

    def _lazy_import_runtime_deps(self) -> None:
        if self._torch is not None:
            return

        import numpy as np  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
        from PIL import Image  # pylint: disable=import-outside-toplevel
        from transformers import AutoProcessor  # pylint: disable=import-outside-toplevel

        model_cls = None
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # pylint: disable=import-outside-toplevel

            model_cls = Qwen2_5_VLForConditionalGeneration
        except Exception:
            try:
                from transformers import AutoModelForVision2Seq  # pylint: disable=import-outside-toplevel

                model_cls = AutoModelForVision2Seq
            except Exception:
                from transformers import AutoModelForCausalLM  # pylint: disable=import-outside-toplevel

                model_cls = AutoModelForCausalLM

        try:
            from qwen_vl_utils import process_vision_info  # pylint: disable=import-outside-toplevel
        except Exception:
            process_vision_info = None

        self._torch = torch
        self._np = np
        self._PILImage = Image
        self._processor_cls = AutoProcessor
        self._model_cls = model_cls
        self._vision_info_fn = process_vision_info
        self.device = self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _build_source_candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, Optional[str]]] = set()

        def add(model_ref: str, subfolder: Optional[str], label: str) -> None:
            key = (model_ref, subfolder)
            if key in seen:
                return
            seen.add(key)
            candidates.append({"model_ref": model_ref, "subfolder": subfolder, "label": label})

        local_root = self._resolve_local_root()
        if local_root is not None:
            root_str = str(local_root)
            add(root_str, None, f"local-root:{root_str}")
            for subfolder in self.subfolder_candidates:
                sub_path = local_root / subfolder
                if sub_path.is_dir():
                    add(root_str, subfolder, f"local-subfolder:{sub_path}")
            if local_root.name.lower() == "huggingface":
                add(root_str, None, f"local-huggingface-root:{root_str}")

        add(self.model_id, None, f"hf-repo-root:{self.model_id}")
        for subfolder in self.subfolder_candidates:
            normalized = subfolder.replace("\\", "/")
            add(self.model_id, normalized, f"hf-repo-subfolder:{self.model_id}/{normalized}")
        return candidates

    def _resolve_local_root(self) -> Optional[Path]:
        values: list[str] = []
        if self.local_model_dir:
            values.append(self.local_model_dir)

        env_path = os.getenv("AFFORDANCE_R1_LOCAL_DIR")
        if env_path:
            values.append(env_path)

        model_id_path = Path(self.model_id).expanduser()
        if model_id_path.exists():
            values.append(str(model_id_path))

        for value in values:
            path = Path(value).expanduser()
            if path.is_dir():
                return path
        return None

    def _load_model_once(
        self,
        model_ref: str,
        subfolder: Optional[str],
        force_local_files_only: bool = False,
    ) -> Any:
        torch = self._torch
        model_cls = self._model_cls

        base_kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.local_files_only or force_local_files_only:
            base_kwargs["local_files_only"] = True
        if subfolder:
            base_kwargs["subfolder"] = subfolder

        if self.device.startswith("cuda"):
            base_kwargs["torch_dtype"] = torch.bfloat16
            base_kwargs["device_map"] = "auto"
            base_kwargs["attn_implementation"] = "eager"
        else:
            base_kwargs["torch_dtype"] = torch.float32
            base_kwargs["device_map"] = "cpu"

        variants = [
            base_kwargs,
            {k: v for k, v in base_kwargs.items() if k != "attn_implementation"},
            {k: v for k, v in base_kwargs.items() if k not in {"attn_implementation", "device_map"}},
        ]

        last_exc = None
        for kwargs in variants:
            try:
                model = model_cls.from_pretrained(model_ref, **kwargs)
                if hasattr(model, "eval"):
                    model.eval()
                return model
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(last_exc)

    def _load_processor_once(
        self,
        model_ref: str,
        subfolder: Optional[str],
        force_local_files_only: bool = False,
    ) -> Any:
        processor_cls = self._processor_cls

        base_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        if self.local_files_only or force_local_files_only:
            base_kwargs["local_files_only"] = True
        if subfolder:
            base_kwargs["subfolder"] = subfolder

        variants = [
            base_kwargs,
            {k: v for k, v in base_kwargs.items() if k not in {"min_pixels", "max_pixels"}},
        ]

        last_exc = None
        for kwargs in variants:
            try:
                return processor_cls.from_pretrained(model_ref, **kwargs)
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(last_exc)

    def _prepare_image(self, image: Any):
        Image = self._PILImage
        np = self._np

        if isinstance(image, (str, Path)):
            path = Path(image).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Image path does not exist: {path}")
            return Image.open(path).convert("RGB")

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            if image.ndim != 3 or image.shape[2] not in (3, 4):
                raise ValueError(f"Unsupported ndarray shape: {image.shape}")
            rgb = image[:, :, :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            return Image.fromarray(rgb).convert("RGB")

        raise TypeError(f"Unsupported image type: {type(image)}")

    def _build_prompt(self, prompt: Optional[str], extra_context: Optional[dict[str, Any]]) -> str:
        base_prompt = prompt or (
            "Find the best robot grasp affordance region. "
            "Return [x1, y1, x2, y2] and a part name."
        )
        if not extra_context:
            return base_prompt
        return f"{base_prompt}\n\nAdditional context: {json.dumps(extra_context, ensure_ascii=False)}"

    def _run_generation(
        self,
        image_pil,
        prompt: Optional[str],
        extra_context: Optional[dict[str, Any]],
        max_new_tokens: int,
    ) -> str:
        prompt_text = self._build_prompt(prompt=prompt, extra_context=extra_context)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        if hasattr(self._processor, "apply_chat_template"):
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt_text

        if self._vision_info_fn is not None:
            image_inputs, video_inputs = self._vision_info_fn(messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self._processor(
                text=[text],
                images=[image_pil],
                padding=True,
                return_tensors="pt",
            )

        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)

        with self._torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
            )

        input_ids = getattr(inputs, "input_ids", None)
        if input_ids is not None:
            trimmed_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
        else:
            trimmed_ids = generated_ids

        return self._processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def _parse_output(self, output_text: str, image_size: tuple[int, int]) -> dict[str, Any]:
        img_w, img_h = image_size
        answer_match = self.ANSWER_PATTERN.search(output_text)
        answer_text = answer_match.group(1).strip() if answer_match else output_text.strip()

        bbox_match = self.BBOX_PATTERN.search(answer_text)
        candidates: list[dict[str, Any]] = []
        summary = "No grasp candidate parsed from model output."

        if bbox_match:
            x1, y1, x2, y2 = [int(v) for v in bbox_match.groups()]
            part_text = answer_text[bbox_match.end() :].strip()
            part = part_text.split()[0].strip(".,:;").lower() if part_text else "object"

            bbox_pixel = [
                int(x1 / 1000.0 * img_w),
                int(y1 / 1000.0 * img_h),
                int(x2 / 1000.0 * img_w),
                int(y2 / 1000.0 * img_h),
            ]
            candidate = {
                "part": part,
                "bbox_norm": [x1, y1, x2, y2],
                "bbox_pixel": bbox_pixel,
                "center_norm": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                "center_pixel": [
                    (bbox_pixel[0] + bbox_pixel[2]) / 2.0,
                    (bbox_pixel[1] + bbox_pixel[3]) / 2.0,
                ],
                "score": 1.0,
            }
            candidates.append(candidate)
            summary = f"Top candidate: part={part}, bbox_norm={[x1, y1, x2, y2]}"

        return {"grasp_candidates": candidates, "summary": summary}

    def _failure_result(self, error_message: str) -> dict[str, Any]:
        return {
            "success": False,
            "model_loaded": self.is_available(),
            "source": "affordance-r1",
            "raw_output": None,
            "grasp_candidates": [],
            "affordance_summary": None,
            "error": error_message,
        }

    def _try_prepare_snapshot_dir(self, model_ref: str, subfolder: Optional[str]) -> Optional[Path]:
        """Fallback: materialize remote repo to a local cache directory."""
        if self.local_files_only or not self.allow_snapshot_download:
            return None
        if not self._looks_like_hf_repo_id(model_ref):
            return None

        try:
            from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel
        except Exception:
            return None

        try:
            allow_patterns = None
            if subfolder:
                normalized = subfolder.replace("\\", "/").strip("/")
                allow_patterns = [f"{normalized}/*"]
            snapshot_root = Path(
                snapshot_download(
                    repo_id=model_ref,
                    allow_patterns=allow_patterns,
                    resume_download=True,
                )
            )
            if subfolder:
                resolved = snapshot_root / subfolder
            else:
                resolved = snapshot_root
            if resolved.is_dir():
                return resolved
        except Exception:
            return None
        return None

    @staticmethod
    def _looks_like_hf_repo_id(model_ref: str) -> bool:
        # Treat only "owner/repo" strings as remote Hugging Face repo IDs.
        return isinstance(model_ref, str) and model_ref.count("/") == 1 and not Path(model_ref).exists()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _main() -> None:
    parser = argparse.ArgumentParser(description="AffordanceGrasp-R1 adapter smoke test")
    parser.add_argument("--model-id", default="hqking/affordance-r1")
    parser.add_argument("--model-dir", default=os.getenv("AFFORDANCE_R1_LOCAL_DIR"))
    parser.add_argument("--image", default=None, help="Optional image path for inference smoke test")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    adapter = AffordanceGraspR1Adapter(
        model_id=args.model_id,
        local_model_dir=args.model_dir,
        local_files_only=args.local_only or _env_flag("AFFORDANCE_R1_LOCAL_ONLY", False),
    )
    adapter.load()
    print(
        json.dumps(
            {"available": adapter.is_available(), "error": adapter._last_error},
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.image:
        result = adapter.predict_from_path(
            image_path=args.image,
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    _main()
