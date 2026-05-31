"""
Loop safeguard for Strands ReAct agent.

Implements iteration budget control via hook system. Registers on BeforeModelCallEvent,
increments counter per LLM call, and cancels agent when budget is exceeded.
Counter resets per user turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from strands.hooks.events import BeforeModelCallEvent

logger = logging.getLogger(__name__)


@dataclass
class IterationLimiter:
    """Caps LLM calls per user turn and cancels agent on breach."""

    max_iterations: int = 25
    _count: int = field(default=0, init=False)
    _tripped: bool = field(default=False, init=False)

    def reset(self) -> None:
        """Reset between user turns."""
        self._count = 0
        self._tripped = False

    @property
    def count(self) -> int:
        return self._count

    def on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Hook callback invoked before every LLM request."""
        self._count += 1
        logger.debug("ReAct cycle #%d (budget: %d)", self._count, self.max_iterations)

        if self._count > self.max_iterations and not self._tripped:
            self._tripped = True
            logger.warning(
                "Iteration budget exhausted (%d > %d). Cancelling agent.",
                self._count,
                self.max_iterations,
            )
            event.agent.cancel()
