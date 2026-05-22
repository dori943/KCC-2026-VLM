"""Pipeline entrypoints."""

from app.pipelines.module2b_pipeline import (
    compare_module2b_outputs,
    export_module2b_normalized_context,
    run_module2b_pipeline,
)

__all__ = [
    "run_module2b_pipeline",
    "export_module2b_normalized_context",
    "compare_module2b_outputs",
]
