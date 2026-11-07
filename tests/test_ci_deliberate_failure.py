"""Deliberate negative control; this branch must never be merged."""

import unittest


class DeliberateFailure(unittest.TestCase):
    """Prove required CI rejects a real failing unit test."""

    def test_failure_blocks_merge(self) -> None:
        """Fail deterministically without touching application behavior."""
        self.fail('Intentional CI validation failure; do not merge')
