"""Temporary main-targeted failure control; never merge this test."""

import unittest


class MainNegativeControl(unittest.TestCase):
    """Require the real main checks to reject a deterministic failure."""

    def test_required_checks_block_merge(self) -> None:
        """Fail without importing models or changing application behavior."""
        self.fail('Intentional main protection validation; do not merge')
