from __future__ import annotations

from .config import SimulationConfig
from .engine import SaliencyScanpathEngine
from .saliency_inference import SaliencyInference
from .strategies import PanelGuidedStrategy, SaliencyOnlyStrategy
from .types import SimulationResult


class ReadingSimulator:
    """Dispatch reading simulation requests to the configured strategy implementation."""

    def __init__(self, inference: SaliencyInference, config: SimulationConfig | None = None) -> None:
        """Create the simulator facade and prepare shared saliency infrastructure."""
        self.config = config or SimulationConfig()
        self.engine = SaliencyScanpathEngine(inference, self.config)
        self._strategies = {
            "saliency_only": SaliencyOnlyStrategy(self.engine, self.config),
            "panel_guided": PanelGuidedStrategy(self.engine, self.config),
        }

    def simulate(
        self,
        page_path: str,
        initial_fixation: tuple[float, float] | None = None,
        initial_roi_size: int | None = None,
    ) -> SimulationResult:
        """Run the active strategy and return a unified simulation result."""
        strategy = self._strategies.get(self.config.strategy)
        if strategy is None:
            raise ValueError(f"Unsupported simulation strategy: {self.config.strategy}")
        return strategy.simulate(page_path, initial_fixation=initial_fixation, initial_roi_size=initial_roi_size)
