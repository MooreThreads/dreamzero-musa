from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BasePolicy(ABC):
    @abstractmethod
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def reset(self, reset_info: Optional[dict[str, Any]] = None) -> None:
        """Reset policy state while remaining compatible with openpi-client."""
        return None
