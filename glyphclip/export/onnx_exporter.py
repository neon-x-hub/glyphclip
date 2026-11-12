"""Model export and serialization utilities for edge and production deployment."""

from pathlib import Path
from typing import Optional, Tuple, Union
import time
import numpy as np
import torch
import torch.nn as nn


class TorchScriptExporter:
    """Exports and validates optimized TorchScript serialized models."""

    @staticmethod
    def export(
        model: nn.Module,
        output_path: Union[str, Path],
        device: str = "cpu",
        in_channels: Optional[int] = None,
    ) -> Path:
        """Export model using torch.jit.trace (supports 1-ch raster or 2-ch Vector-SDF)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        model.to(device)

        ch = in_channels if in_channels is not None else int(getattr(model, "in_channels", 1))
        dummy_input = torch.randn(1, ch, 224, 224, device=device)
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(output_path))
        return output_path

    @staticmethod
    def verify(
        pytorch_model: nn.Module,
        script_path: Union[str, Path],
        device: str = "cpu",
        atol: float = 1e-4,
        in_channels: Optional[int] = None,
    ) -> bool:
        """Verify numerical consistency between PyTorch and TorchScript runtime."""
        pytorch_model.eval().to(device)
        loaded = torch.jit.load(str(script_path), map_location=device)
        loaded.eval()

        ch = in_channels if in_channels is not None else int(getattr(pytorch_model, "in_channels", 1))
        dummy = torch.randn(2, ch, 224, 224, device=device)
        with torch.no_grad():
            out_pt = pytorch_model(dummy)
            out_ts = loaded(dummy)

        is_close = torch.allclose(out_pt, out_ts, atol=atol)
        if not is_close:
            max_diff = (out_pt - out_ts).abs().max().item()
            raise ValueError(f"TorchScript verification failed: max diff {max_diff} > {atol}")
        return True

    @staticmethod
    def benchmark_cpu(
        script_path: Union[str, Path],
        num_warmup: int = 5,
        num_runs: int = 20,
        in_channels: int = 1,
    ) -> float:
        """Benchmark average CPU latency in milliseconds per glyph canvas."""
        loaded = torch.jit.load(str(script_path), map_location="cpu")
        loaded.eval()
        dummy = torch.randn(1, in_channels, 224, 224)

        with torch.no_grad():
            for _ in range(num_warmup):
                _ = loaded(dummy)

            t0 = time.perf_counter()
            for _ in range(num_runs):
                _ = loaded(dummy)
            elapsed = time.perf_counter() - t0

        return (elapsed / num_runs) * 1000.0


class ONNXExporter:
    """Handles exporting and benchmark validation for ONNX runtimes."""

    @staticmethod
    def is_available() -> bool:
        """Check if onnx serialization package is installed."""
        try:
            import onnx
            return True
        except ImportError:
            return False

    @staticmethod
    def export(
        model: nn.Module,
        output_path: Union[str, Path],
        opset_version: int = 17,
        device: str = "cpu",
        in_channels: Optional[int] = None,
    ) -> Path:
        """Export model to ONNX with dynamic batch size (supports 1-ch or 2-ch inputs)."""
        if not ONNXExporter.is_available():
            raise RuntimeError("Package 'onnx' is required for ONNX export.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        model.to(device)
        ch = in_channels if in_channels is not None else int(getattr(model, "in_channels", 1))
        dummy_input = torch.randn(1, ch, 224, 224, device=device)

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_glyphs"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_glyphs": {0: "batch_size"},
                "embeddings": {0: "batch_size"},
            },
        )
        return output_path
