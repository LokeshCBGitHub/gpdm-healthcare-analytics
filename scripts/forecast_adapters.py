from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class _Adapter:
    name: str = "base"
    available: bool = False

    def forecast(self, values: np.ndarray, periods: int) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class ClassicalAdapter(_Adapter):
    name = "classical_hw"
    available = True

    def forecast(self, values: np.ndarray, periods: int) -> Dict[str, Any]:
        values = np.asarray(values, dtype=float)
        n = len(values)
        alpha, beta = 0.3, 0.1
        level = values[0]
        trend = (values[1] - values[0]) if n > 1 else 0.0
        for v in values[1:]:
            prev_level = level
            level = alpha * v + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        point = np.array([level + trend * (i + 1) for i in range(periods)], dtype=float)
        base_std = values.std() * 0.3 if n > 3 else max(values.std(), 1e-6)
        h = np.sqrt(np.arange(1, periods + 1))
        band = 1.96 * base_std * h
        return {
            "point": np.maximum(point, 0.0),
            "lower": np.maximum(point - band, 0.0),
            "upper": point + band,
            "model": self.name,
            "zero_shot": False,
        }


class ChronosAdapter(_Adapter):
    name = "chronos"

    def __init__(self, model_name: str = "amazon/chronos-t5-tiny",
                 num_samples: int = 20):
        self._model_name = model_name
        self._num_samples = num_samples
        self._pipeline = None
        self._lock = threading.Lock()
        self.available = self._probe()

    def _probe(self) -> bool:
        try:
            import chronos
            return True
        except Exception as e:
            logger.debug("[Chronos] not available: %s", e)
            return False

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline
            try:
                import torch
                from chronos import ChronosPipeline
                device = "cuda" if _has_cuda() else "cpu"
                dtype_name = "bfloat16" if device == "cuda" else "float32"
                import torch as _t
                dtype = getattr(_t, dtype_name)
                self._pipeline = ChronosPipeline.from_pretrained(
                    self._model_name, device_map=device, torch_dtype=dtype,
                )
            except Exception as e:
                logger.warning("[Chronos] load failed: %s", e)
                self._pipeline = None
        return self._pipeline

    def forecast(self, values: np.ndarray, periods: int) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        pipe = self._load()
        if pipe is None:
            return None
        try:
            import torch
            ctx = torch.tensor(np.asarray(values, dtype=float))
            samples = pipe.predict(
                context=ctx, prediction_length=int(periods),
                num_samples=int(self._num_samples),
            )
            arr = samples[0].cpu().numpy()
            point = np.median(arr, axis=0)
            lower = np.quantile(arr, 0.05, axis=0)
            upper = np.quantile(arr, 0.95, axis=0)
            return {
                "point": np.maximum(point, 0.0),
                "lower": np.maximum(lower, 0.0),
                "upper": upper,
                "model": self.name,
                "zero_shot": True,
                "model_id": self._model_name,
            }
        except Exception as e:
            logger.warning("[Chronos] forecast failed: %s", e)
            return None


class TimesFMAdapter(_Adapter):
    name = "timesfm"

    def __init__(self, repo_id: str = "google/timesfm-1.0-200m"):
        self._repo = repo_id
        self._model = None
        self._lock = threading.Lock()
        self.available = self._probe()

    def _probe(self) -> bool:
        try:
            import timesfm
            return True
        except Exception as e:
            logger.debug("[TimesFM] not available: %s", e)
            return False

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                import timesfm
                backend = "gpu" if _has_cuda() else "cpu"
                m = timesfm.TimesFm(
                    context_len=512, horizon_len=128, input_patch_len=32,
                    output_patch_len=128, num_layers=20, model_dims=1280,
                    backend=backend,
                )
                m.load_from_checkpoint(repo_id=self._repo)
                self._model = m
            except Exception as e:
                logger.warning("[TimesFM] load failed: %s", e)
                self._model = None
        return self._model

    def forecast(self, values: np.ndarray, periods: int) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        model = self._load()
        if model is None:
            return None
        try:
            vals = np.asarray(values, dtype=np.float32).reshape(1, -1)
            freq = [0]
            point_forecast, quantile_forecast = model.forecast(
                inputs=vals, freq=freq,
            )
            pt = point_forecast[0, :periods]
            q = quantile_forecast[0, :periods, :]
            lower = q[:, 0]
            upper = q[:, -1]
            return {
                "point": np.maximum(pt, 0.0),
                "lower": np.maximum(lower, 0.0),
                "upper": upper,
                "model": self.name,
                "zero_shot": True,
                "model_id": self._repo,
            }
        except Exception as e:
            logger.warning("[TimesFM] forecast failed: %s", e)
            return None


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class ForecastBackend:

    def __init__(self, priority: List[str]):
        self._adapters: List[_Adapter] = []
        for name in priority:
            a = _build(name)
            if a is not None:
                self._adapters.append(a)
        if not any(isinstance(a, ClassicalAdapter) for a in self._adapters):
            self._adapters.append(ClassicalAdapter())

    def forecast(self, values, periods: int) -> Dict[str, Any]:
        for a in self._adapters:
            out = a.forecast(np.asarray(values, dtype=float), int(periods))
            if out is not None:
                out.setdefault("attempted", [x.name for x in self._adapters])
                out.setdefault("chose", a.name)
                return out
        return ClassicalAdapter().forecast(values, periods)

    def status(self) -> Dict[str, Any]:
        return {
            "priority": [a.name for a in self._adapters],
            "available": {a.name: bool(a.available) for a in self._adapters},
            "cuda": _has_cuda(),
        }


def _build(name: str) -> Optional[_Adapter]:
    n = (name or "").strip().lower()
    if n in ("classical", "classical_hw", "hw"):
        return ClassicalAdapter()
    if n == "chronos":
        return ChronosAdapter()
    if n == "timesfm":
        return TimesFMAdapter()
    logger.warning("[forecast_adapters] unknown backend %r", name)
    return None


_BACKEND: Optional[ForecastBackend] = None
_BACKEND_LOCK = threading.Lock()


def get_backend() -> ForecastBackend:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is None:
            env = os.environ.get("GPDM_FORECAST_BACKEND", "classical")
            priority = [p.strip() for p in env.split(",") if p.strip()]
            _BACKEND = ForecastBackend(priority)
            logger.info("[forecast_adapters] status=%s", _BACKEND.status())
    return _BACKEND


def reset_backend() -> None:
    global _BACKEND
    with _BACKEND_LOCK:
        _BACKEND = None
