"""Gemeinsame Pfade für die Tests (DEMO-PLAYBOOK §1b); die Mini-Instanzen stehen in den Testdateien selbst, dort wo sie gerechnet werden."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
