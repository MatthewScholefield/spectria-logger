"""Keras adapter that logs training metrics to Spectria JSONL format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .writer import RunWriter


class SpectriaCallback:
    """Keras callback that writes epoch-level (and optionally batch-level) metrics to JSONL.

    Usage::

        from spectria.logger import SpectriaCallback

        model.fit(x, y, callbacks=[
            SpectriaCallback(project="mnist", run="baseline",
                             config={"lr": 0.01, "optimizer": "adam"})
        ])
    """

    def __init__(
        self,
        project: str = "default",
        run: str | None = None,
        baseline: str | None = None,
        config: dict[str, Any] | None = None,
        logdir: str | Path = "./spectria_logs",
        include_batch_metrics: bool = False,
    ) -> None:
        self.project = project
        self.run = run or _auto_run_name()
        self.baseline = baseline
        self.config = config or {}
        self.logdir = Path(logdir)
        self.include_batch_metrics = include_batch_metrics
        self._writer = RunWriter(
            logdir=self.logdir,
            project=self.project,
            run=self.run,
            baseline=self.baseline,
            config=self.config,
        )
        self._current_epoch = 0

    def _get_keras_callback_cls(self):
        """Lazily import keras to avoid hard dependency at import time."""
        try:
            import keras

            if hasattr(keras, "callbacks") and hasattr(keras.callbacks, "Callback"):
                return keras.callbacks.Callback
        except ImportError:
            pass
        raise ImportError(
            "keras is required to use SpectriaCallback. "
            "Install it with: pip install spectria-logger[keras]"
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None):
        logs = logs or {}
        row = {"epoch": epoch}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                row[key] = value
        self._writer.write_row(row)
        self._current_epoch = epoch + 1

    def on_batch_end(self, batch: int, logs: dict[str, Any] | None = None):
        if not self.include_batch_metrics:
            return
        logs = logs or {}
        row = {"epoch": self._current_epoch, "batch": batch}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                row[key] = value
        self._writer.write_row(row)

    def on_train_end(self, logs=None):
        pass

    def get_keras_callback(self):
        """Return a dict-based callback compatible with keras.Callback interface.

        This allows using SpectriaCallback with model.fit() without subclassing.
        """
        return {
            "on_train_begin": self.on_train_begin,
            "on_epoch_end": self.on_epoch_end,
            "on_batch_end": self.on_batch_end,
            "on_train_end": self.on_train_end,
        }


def _auto_run_name() -> str:
    """Generate a run name from timestamp."""
    from datetime import datetime

    return datetime.now().strftime("run-%Y%m%d-%H%M%S")


def as_keras_callback(callback: SpectriaCallback):
    """Wrap a SpectriaCallback as a keras.Callback subclass instance.

    This is needed for keras >= 3 which requires actual Callback subclass objects.

    Usage::

        from spectria.logger import SpectriaCallback, as_keras_callback

        model.fit(x, y, callbacks=[
            as_keras_callback(SpectriaCallback(project="mnist", run="baseline"))
        ])
    """
    import keras

    class _SpectriaKerasCallback(keras.callbacks.Callback):
        def __init__(self, spectria_cb: SpectriaCallback):
            super().__init__()
            self._cb = spectria_cb

        def on_epoch_end(self, epoch, logs=None):
            self._cb.on_epoch_end(epoch, logs)

        def on_batch_end(self, batch, logs=None):
            self._cb.on_batch_end(batch, logs)

    return _SpectriaKerasCallback(callback)
