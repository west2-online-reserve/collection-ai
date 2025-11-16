"""Demo plugin step providers for pkmninsh."""

from __future__ import annotations

from typing import List

from pkmninsh.engine.pipeline import ActionContext
from pkmninsh.engine.steps.catalog import Step, StepProvider


class DemoLogStepProvider(StepProvider):
    """Simple provider that records execution for debugging demos."""

    namespace = "plugins.demo.log_step"

    def priority(self) -> int:
        # Run after target resolution but before the default pre-effect hooks.
        return 15

    def match(self, _: ActionContext) -> bool:
        return True

    def steps(self, _: ActionContext) -> List[Step]:
        def _record(ctx: ActionContext) -> None:
            ctx.setdefault("plugin_steps", []).append("plugins.demo.log_step")

        return [_record]


def build_demo_providers() -> List[StepProvider]:
    """Return the step providers exported by this demo plugin."""

    return [DemoLogStepProvider()]
