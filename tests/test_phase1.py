"""
tests/test_config.py
Phase 1 — Config loader unit tests.
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_config_loads():
    """Config should load without errors."""
    from config.settings import get_config
    cfg = get_config()
    assert cfg is not None


def test_config_app_name():
    """Config should have the correct project name."""
    from config.settings import get_config
    cfg = get_config()
    assert cfg.app["name"] == "Context-Aware Workspace Organizer"


def test_config_vision_keys():
    """Vision config should contain required keys."""
    from config.settings import get_config
    cfg = get_config()
    vision = cfg.vision
    assert "yolo" in vision
    assert "bytetrack" in vision
    assert "trocr" in vision
    assert "segformer" in vision


def test_config_desk_dimensions():
    """Desk dimensions should be physically reasonable."""
    from config.settings import get_config
    cfg = get_config()
    desk = cfg.desk
    assert desk["width_m"] > 0
    assert desk["depth_m"] > 0


def test_database_manager_init():
    """DatabaseManager should initialise and create tables."""
    import tempfile
    import os
    from database.db_manager import DatabaseManager

    # Use ignore_cleanup_errors so Windows file-lock on SQLite does not
    # cause a false failure — the DB file is created and tables exist,
    # which is the behaviour under test.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db_path = os.path.join(tmp, "test.db")
        db = DatabaseManager(db_path)
        db.init_db()
        assert os.path.exists(db_path)
        # Explicitly release the SQLite connection before tmp dir cleanup
        db.engine.dispose()


def test_system_info():
    """system_info should return expected keys."""
    from utils.system_info import get_system_info
    info = get_system_info()
    for key in ["os", "python_version", "cpu_cores", "ram_total_gb"]:
        assert key in info
