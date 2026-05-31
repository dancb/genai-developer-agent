"""
Loop safeguard for the Strands single-agent ReAct loop.

WHY THIS EXISTS
---------------
The current Strands SDK (Agent class) does NOT expose a `max_iterations`
parameter the way LangChain's AgentExecutor does. Iteration budget control
is intentionally pushed out to user code via the hook system. This module
provides a clean, drop-in implementation.

HOW IT WORKS
------------
We register a hook on `BeforeModelCallEvent`. Each time the agent is about
to call the LLM, we increment a counter. If the counter exceeds the budget,
we call `agent.cancel()` — which is the thread-safe, idempotent termination
mechanism documented in the Strands Agent Loop reference. The agent then
returns with `stop_reason="cancelled"` and the conversation history is left
in a consistent state.

The counter resets at the start of each user turn (call .reset() in the CLI).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from strands.hooks.events import BeforeModelCallEvent

logger = logging.getLogger(__name__)


@dataclass
class IterationLimiter:
    """Caps the number of LLM calls per user turn and cancels the agent on breach.

    Attributes:
        max_iterations: Hard upper bound on model calls per user turn.
        _count: Internal model-call counter (zero-based per turn).
        _tripped: Whether the limiter has already fired cancel() this turn.
    """

    max_iterations: int = 25
    _count: int = field(default=0, init=False)
    _tripped: bool = field(default=False, init=False)

    def reset(self) -> None:
        """Reset between user turns. Call this in the REPL loop before each prompt."""
        self._count = 0
        self._tripped = False

    @property
    def count(self) -> int:
        return self._count

    def on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Hook callback. Strands invokes this before every LLM request.

        Args:
            event: The Strands hook event. Carries an .agent reference we can
                   call .cancel() on without storing the agent ourselves.
        """
        self._count += 1
        logger.debug("ReAct cycle #%d (budget: %d)", self._count, self.max_iterations)

        if self._count > self.max_iterations and not self._tripped:
            self._tripped = True
            logger.warning(
                "Iteration budget exhausted (%d > %d). Cancelling agent.",
                self._count,
                self.max_iterations,
            )
            # event.agent.cancel() is thread-safe & idempotent per the SDK docs.
            # The agent will stop at the next checkpoint and return
            # stop_reason="cancelled".
            event.agent.cancel()
