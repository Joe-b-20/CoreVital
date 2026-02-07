# ============================================================================
# CoreVital - Performance Monitor
#
# Purpose: Lightweight timing for CoreVital operations and overhead measurement
# Inputs: Context manager calls around operations
# Outputs: Summary (parent_operations, unaccounted_time) and optional detailed breakdown
# Dependencies: time, contextlib
# Usage: with monitor.operation("name"): ...
#
# Changelog:
#   2026-02-04: Initial performance monitor for Phase-0.75
#   2026-02-05: Removed origin labeling, simplified strict mode output
#               baseline_ms is now the single source for raw inference timing
#   2026-02-06: sink_write removed from parent_operations (perf data must be
#               finalized before the write that carries it)
# ============================================================================

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class OperationTiming:
    """Timing for a single operation (can have children)."""
    operation_name: str
    duration_ms: float = 0.0
    children: List["OperationTiming"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # For repeated ops (per-step, per-layer): list of ms per occurrence
    _samples: List[float] = field(default_factory=list, repr=False)

    def add_sample(self, ms: float) -> None:
        self._samples.append(ms)


class PerformanceMonitor:
    """
    Lightweight performance monitor. Tracks wall time per operation via context managers.
    Sequential execution only; no overlapping.
    """

    def __init__(self, mode: str = "summary"):
        self.mode = mode  # "summary" | "detailed" | "strict"
        self.stack: List[OperationTiming] = []
        self.root_timings: List[OperationTiming] = []
        self._run_start_time: Optional[float] = None  # Actual wall clock start
        self._run_end_time: Optional[float] = None    # Actual wall clock end
        self.total_wall_time_ms: float = 0.0
        self.instrumented_inference_ms: Optional[float] = None
        self.detailed_file_path: Optional[str] = None
        # Strict mode: separate tracking for warmup/baseline (not counted in total)
        self.warmup_ms: Optional[float] = None
        self.baseline_ms: Optional[float] = None  # Also serves as raw_inference_ms
        self.original_model_load_ms: Optional[float] = None  # Original model load time (before caching)

    def mark_run_start(self) -> None:
        """Mark the actual start of the instrumented run (for total_wall_time_ms)."""
        self._run_start_time = time.perf_counter()

    def mark_run_end(self) -> None:
        """Mark the actual end of the instrumented run (for total_wall_time_ms)."""
        self._run_end_time = time.perf_counter()
        if self._run_start_time is not None:
            self.total_wall_time_ms = (self._run_end_time - self._run_start_time) * 1000

    @contextmanager
    def operation(self, name: str, **metadata: Any):
        """Time an operation. Nested calls form a hierarchy.
        
        Args:
            name: Operation name
            **metadata: Additional metadata to store with the timing
        """
        start = time.perf_counter()
        timing = OperationTiming(operation_name=name, metadata=metadata)
        if self.stack:
            self.stack[-1].children.append(timing)
        else:
            self.root_timings.append(timing)
        self.stack.append(timing)
        try:
            yield timing  # Yield timing object so caller can add children/metadata
        finally:
            self.stack.pop()
            timing.duration_ms = (time.perf_counter() - start) * 1000

    def set_total_wall_time_ms(self, ms: float) -> None:
        """Override total wall time (use mark_run_start/end instead when possible)."""
        self.total_wall_time_ms = ms

    def set_instrumented_inference_ms(self, ms: float) -> None:
        self.instrumented_inference_ms = ms

    def set_detailed_file(self, path: str) -> None:
        self.detailed_file_path = path

    def set_warmup_ms(self, ms: float) -> None:
        """Set warmup time (strict mode, not counted in total)."""
        self.warmup_ms = ms

    def set_baseline_ms(self, ms: float) -> None:
        """Set baseline/raw inference time (strict mode, not counted in total)."""
        self.baseline_ms = ms

    def set_original_model_load_ms(self, ms: float) -> None:
        """Set original model load time (strict mode, before caching)."""
        self.original_model_load_ms = ms

    def _parent_operations(self) -> List[OperationTiming]:
        """Root-level operations in order (these are the 'parents')."""
        return self.root_timings

    def _sum_parent_ms(self) -> float:
        """Sum of all parent operation durations (for sanity checks)."""
        return sum(p.duration_ms for p in self._parent_operations())

    def build_summary_dict(self) -> Dict[str, Any]:
        """Build the object for Report.extensions['performance'].
        
        Note: Origin is NOT included for parent operations in summary.
        Origin is only meaningful for child operations in detailed breakdown.
        
        In strict mode:
        - model_load is replaced with original_model_load_ms (cold load time)
        - total_wall_time_ms already includes pre-run time (model load + warmup + baseline)
        - warmup_ms and baseline_ms are shown separately (they're "intentional" unaccounted time)
        """
        total = self.total_wall_time_ms or self._sum_parent_ms()
        parents = self._parent_operations()
        
        # In strict mode, we need to replace model_load with original_model_load_ms
        # total_wall_time_ms already includes the pre-run time (no adjustment needed)
        if self.mode == "strict" and self.original_model_load_ms is not None:
            # Build parent_operations with model_load replaced by original value
            parent_ops = []
            sum_parent_ms = 0.0
            for p in parents:
                if p.operation_name == "model_load":
                    ms = self.original_model_load_ms
                else:
                    ms = p.duration_ms
                sum_parent_ms += ms
                parent_ops.append({
                    "name": p.operation_name,
                    "ms": round(ms, 2),
                    "pct": round(100.0 * ms / total, 2) if total else 0.0,
                })
            
        else:
            # Non-strict mode: use actual durations
            sum_parent_ms = sum(p.duration_ms for p in parents)
            parent_ops = [
                {
                    "name": p.operation_name,
                    "ms": round(p.duration_ms, 2),
                    "pct": round(100.0 * p.duration_ms / total, 2) if total else 0.0,
                }
                for p in parents
            ]
        
        # Calculate unaccounted time
        unaccounted_ms = total - sum_parent_ms
        # In strict mode, warmup + baseline are intentional "dead time" not tracked as parents
        # They should be subtracted from unaccounted (shown separately)
        if self.mode == "strict":
            unaccounted_ms -= (self.warmup_ms or 0.0) + (self.baseline_ms or 0.0)
        unaccounted_pct = 100.0 * unaccounted_ms / total if total else 0.0

        out: Dict[str, Any] = {
            "total_wall_time_ms": round(total, 2),
            "parent_operations": parent_ops,
            "unaccounted_time": {"ms": round(unaccounted_ms, 2), "pct": round(unaccounted_pct, 2)},
            "detailed_file": self.detailed_file_path,
        }
        
        # Strict mode: include extra metrics
        if self.mode == "strict":
            if self.original_model_load_ms is not None:
                out["original_model_load_ms"] = round(self.original_model_load_ms, 2)
            if self.warmup_ms is not None:
                out["warmup_ms"] = round(self.warmup_ms, 2)
            if self.baseline_ms is not None:
                out["baseline_ms"] = round(self.baseline_ms, 2)
                out["instrumented_inference_ms"] = round(self.instrumented_inference_ms, 2) if self.instrumented_inference_ms is not None else None
                if self.instrumented_inference_ms is not None:
                    inf_overhead_ms = self.instrumented_inference_ms - self.baseline_ms
                    out["inference_overhead_ms"] = round(inf_overhead_ms, 2)
                    out["inference_overhead_pct"] = round(100.0 * inf_overhead_ms / self.baseline_ms, 2) if self.baseline_ms else 0.0
                # corevital_overhead = sum(non-model_load, non-tokenize parents) - baseline_ms
                # Use the adjusted parent values from above
                corevital_ops_ms = sum_parent_ms
                for op in parent_ops:
                    if op["name"] in ("model_load", "tokenize"):
                        corevital_ops_ms -= op["ms"]
                corevital_overhead_ms = corevital_ops_ms - self.baseline_ms
                out["corevital_overhead_ms"] = round(corevital_overhead_ms, 2)
                out["corevital_overhead_pct"] = round(100.0 * corevital_overhead_ms / self.baseline_ms, 2) if self.baseline_ms else 0.0
        return out

    def build_detailed_breakdown(self) -> Dict[str, Any]:
        """Build the nested breakdown for the detailed JSON file.
        
        In strict mode, model_load is replaced with original_model_load_ms.
        """
        total = self.total_wall_time_ms or self._sum_parent_ms()

        def node_to_dict(node: OperationTiming, override_ms: Optional[float] = None) -> Dict[str, Any]:
            """Convert an operation node to dict."""
            ms = override_ms if override_ms is not None else node.duration_ms
            d: Dict[str, Any] = {
                "ms": round(ms, 2),
                "pct": round(100.0 * ms / total, 2) if total else 0.0,
            }
            # Include per_step stats if present (from _samples or metadata)
            if node._samples:
                d["per_step"] = {
                    "count": len(node._samples),
                    "min_ms": round(min(node._samples), 2),
                    "max_ms": round(max(node._samples), 2),
                    "avg_ms": round(sum(node._samples) / len(node._samples), 2),
                }
            elif "per_step" in node.metadata:
                # Support per_step stored in metadata
                ps = node.metadata["per_step"]
                d["per_step"] = {
                    "count": ps["count"],
                    "min_ms": round(ps["min_ms"], 2),
                    "max_ms": round(ps["max_ms"], 2),
                    "avg_ms": round(ps["avg_ms"], 2),
                }
            # Recursively add children
            if node.children:
                d["children"] = {c.operation_name: node_to_dict(c) for c in node.children}
            return d

        breakdown: Dict[str, Any] = {}
        for p in self._parent_operations():
            # In strict mode, replace model_load with original_model_load_ms
            if self.mode == "strict" and p.operation_name == "model_load" and self.original_model_load_ms is not None:
                breakdown[p.operation_name] = node_to_dict(p, override_ms=self.original_model_load_ms)
            else:
                breakdown[p.operation_name] = node_to_dict(p)
        return {
            "total_wall_time_ms": round(total, 2),
            "breakdown": breakdown,
        }
