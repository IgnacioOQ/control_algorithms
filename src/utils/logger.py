"""
Logging utilities for training metrics.

Provides CSV-based logging with optional TensorBoard support.
"""

import csv
import os
from datetime import datetime
from typing import Any, Dict, Optional

# Try to import tensorboard; if not available, disable
try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class Logger:
    """Training metrics logger with CSV and optional TensorBoard support.

    Logs episode rewards, losses, and other metrics to CSV files.
    Optionally writes to TensorBoard for visualization.

    Attributes:
        log_dir: Directory where logs are stored.
        use_tensorboard: Whether TensorBoard logging is enabled.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = False,
    ):
        """Initialize the logger.

        Args:
            log_dir: Base directory for logs.
            experiment_name: Name for this experiment. If None, uses timestamp.
            use_tensorboard: Whether to also log to TensorBoard.
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # CSV logging
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        self._csv_file = None
        self._csv_writer = None
        self._csv_headers_written = False

        # TensorBoard
        self.use_tensorboard = use_tensorboard and HAS_TENSORBOARD
        self._tb_writer = None
        if self.use_tensorboard:
            self._tb_writer = SummaryWriter(log_dir=self.log_dir)

        # Tracking
        self._step = 0

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics.

        Args:
            metrics: Dictionary of metric names to values.
            step: Optional step/episode number. If None, uses internal counter.
        """
        if step is None:
            step = self._step
            self._step += 1

        # Add step to metrics
        metrics_with_step = {"step": step, **metrics}

        # CSV logging
        self._log_csv(metrics_with_step)

        # TensorBoard logging
        if self.use_tensorboard and self._tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._tb_writer.add_scalar(key, value, step)

    def _log_csv(self, metrics: Dict[str, Any]) -> None:
        """Write metrics to CSV file."""
        if self._csv_file is None:
            self._csv_file = open(self.csv_path, "w", newline="")

        if not self._csv_headers_written:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(metrics.keys())
            )
            self._csv_writer.writeheader()
            self._csv_headers_written = True

        self._csv_writer.writerow(metrics)
        self._csv_file.flush()

    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        extra: Optional[Dict[str, float]] = None,
    ) -> None:
        """Convenience method to log episode summary.

        Args:
            episode: Episode number.
            reward: Total episode reward.
            length: Episode length (steps).
            extra: Additional metrics to log.
        """
        metrics = {
            "episode": episode,
            "episode_reward": reward,
            "episode_length": length,
        }
        if extra:
            metrics.update(extra)

        self.log(metrics, step=episode)

    def close(self) -> None:
        """Close all file handles."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()
