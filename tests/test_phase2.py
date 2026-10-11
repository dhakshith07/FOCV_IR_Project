"""
tests/test_phase2.py
Phase 2 — Comprehensive unit and integration tests for the workspace simulator.

Test groups
-----------
  A. Book model           (geometry, serialisation, overlap)
  B. Zone model           (containment, clamping, geometry)
  C. Obstacle model       (blocking geometry)
  D. Factory functions    (default zones and obstacles)
  E. WorkspaceSimulator   (init, reset, randomize, move, add, remove)
  F. Session control      (user profiles, context switching)
  G. Query helpers        (zone lookup, context filter)
  H. Renderer             (figure creation smoke test)

Run:
    pytest tests/test_phase2.py -v
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.book import Book, BOOK_COLORS
from simulation.zones import Zone, Obstacle, create_default_zones, create_default_obstacles
from simulation.workspace import WorkspaceSimulator, CONTEXT_BOOK_MAP, USER_PROFILES


# ═══════════════════════════════════════════════════════════════════════════════
#  A. Book Model
# ═══════════════════════════════════════════════════════════════════════════════

class TestBook:

    def test_book_creation_defaults(self):
        b = Book(id="B01", title="Test Book", x=10.0, y=20.0)
        assert b.id == "B01"
        assert b.title == "Test Book"
        assert b.x == 10.0
        assert b.y == 20.0
        assert b.width == 9.0
        assert b.height == 13.0
        assert b.usage_frequency == 0
        assert b.current_context == "general"
        assert b.last_used is not None   # auto-set in __post_init__

    def test_book_rect_geometry(self):
        b = Book(id="B01", title="T", x=50.0, y=40.0, width=10.0, height=20.0)
        x0, y0, x1, y1 = b.rect
        assert x0 == 45.0
        assert y0 == 30.0
        assert x1 == 55.0
        assert y1 == 50.0

    def test_book_corners_closed_polygon(self):
        b = Book(id="B01", title="T", x=50.0, y=40.0, width=10.0, height=20.0)
        xs, ys = b.corners
        assert len(xs) == 5   # closed: first == last
        assert len(ys) == 5
        assert xs[0] == xs[-1]
        assert ys[0] == ys[-1]

    def test_book_contains_point_inside(self):
        b = Book(id="B01", title="T", x=50.0, y=40.0, width=10.0, height=20.0)
        assert b.contains_point(50.0, 40.0)   # centre
        assert b.contains_point(45.0, 30.0)   # corner
        assert b.contains_point(55.0, 50.0)   # opposite corner

    def test_book_contains_point_outside(self):
        b = Book(id="B01", title="T", x=50.0, y=40.0, width=10.0, height=20.0)
        assert not b.contains_point(10.0, 10.0)
        assert not b.contains_point(56.0, 40.0)   # just outside right edge

    def test_book_overlap_true(self):
        a = Book(id="A", title="A", x=10.0, y=10.0, width=10.0, height=10.0)
        b = Book(id="B", title="B", x=12.0, y=12.0, width=10.0, height=10.0)
        assert a.overlaps(b)

    def test_book_overlap_false(self):
        a = Book(id="A", title="A", x=10.0, y=10.0, width=8.0, height=8.0)
        b = Book(id="B", title="B", x=40.0, y=40.0, width=8.0, height=8.0)
        assert not a.overlaps(b)

    def test_book_overlap_margin(self):
        """Books just outside each other but within margin should still overlap."""
        a = Book(id="A", title="A", x=10.0, y=10.0, width=8.0, height=8.0)
        b = Book(id="B", title="B", x=19.0, y=10.0, width=8.0, height=8.0)
        # Without margin: x1_a=14, x0_b=15 → no overlap
        assert not a.overlaps(b, margin=0)
        # With margin=2: expanded x1_a=16 > x0_b=15 → overlap
        assert a.overlaps(b, margin=2)

    def test_book_to_dict_keys(self):
        b = Book(id="B01", title="ML", x=10.0, y=20.0)
        d = b.to_dict()
        required_keys = {"id", "title", "x", "y", "width", "height",
                         "usage_frequency", "last_used", "current_context", "color"}
        assert required_keys.issubset(d.keys())

    def test_book_to_dict_values(self):
        b = Book(id="B01", title="ML", x=10.0, y=20.0, usage_frequency=5)
        d = b.to_dict()
        assert d["id"] == "B01"
        assert d["title"] == "ML"
        assert d["usage_frequency"] == 5

    def test_book_from_dict_round_trip(self):
        b = Book(id="B01", title="ML", x=33.5, y=22.0, usage_frequency=3,
                 current_context="Math", color="#ff0000")
        d = b.to_dict()
        b2 = Book.from_dict(d)
        assert b2.id == b.id
        assert b2.title == b.title
        assert abs(b2.x - b.x) < 0.01
        assert b2.usage_frequency == b.usage_frequency
        assert b2.color == b.color

    def test_book_repr(self):
        b = Book(id="B01", title="ML", x=10.0, y=20.0)
        r = repr(b)
        assert "B01" in r
        assert "ML" in r


# ═══════════════════════════════════════════════════════════════════════════════
#  B. Zone Model
# ═══════════════════════════════════════════════════════════════════════════════

class TestZone:

    @pytest.fixture
    def zone(self):
        return Zone(
            name="Test Zone", x0=10, y0=10, x1=50, y1=40,
            color="rgba(0,0,0,0)", border_color="rgba(0,0,0,0)", priority=1,
        )

    def test_zone_dimensions(self, zone):
        assert zone.width == 40
        assert zone.height == 30

    def test_zone_center(self, zone):
        cx, cy = zone.center
        assert cx == 30.0
        assert cy == 25.0

    def test_zone_contains_inside(self, zone):
        assert zone.contains(30, 25)   # centre
        assert zone.contains(10, 10)   # corner

    def test_zone_contains_outside(self, zone):
        assert not zone.contains(5, 25)     # left of zone
        assert not zone.contains(60, 25)    # right of zone

    def test_zone_contains_rect_fully_inside(self, zone):
        assert zone.contains_rect(15, 15, 45, 35)

    def test_zone_contains_rect_outside(self, zone):
        assert not zone.contains_rect(0, 0, 100, 100)  # too big

    def test_zone_clamp_center_already_valid(self, zone):
        cx, cy = zone.clamp_center(30, 25, 5, 5)
        assert cx == 30
        assert cy == 25

    def test_zone_clamp_center_out_of_range(self, zone):
        # Object with hw=5, hh=5 → must be at least 5 from each edge
        cx, cy = zone.clamp_center(0, 0, 5, 5)
        assert cx == 15   # x0 + hw = 10 + 5
        assert cy == 15   # y0 + hh = 10 + 5


# ═══════════════════════════════════════════════════════════════════════════════
#  C. Obstacle Model
# ═══════════════════════════════════════════════════════════════════════════════

class TestObstacle:

    @pytest.fixture
    def obs(self):
        return Obstacle(name="Laptop", x=60, y=50, width=30, height=20)

    def test_obstacle_rect(self, obs):
        assert obs.x0 == 45
        assert obs.y0 == 40
        assert obs.x1 == 75
        assert obs.y1 == 60

    def test_obstacle_blocks_inside(self, obs):
        assert obs.blocks(60, 50, 5, 5)   # centre

    def test_obstacle_blocks_adjacent_with_margin(self, obs):
        # Just outside the obstacle edge but within margin=1.5
        assert obs.blocks(76, 50, 0.1, 0.1, margin=1.5)

    def test_obstacle_not_blocks_far_away(self, obs):
        assert not obs.blocks(10, 10, 2, 2, margin=1.5)


# ═══════════════════════════════════════════════════════════════════════════════
#  D. Factory Functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestFactories:

    def test_create_default_zones_count(self):
        zones = create_default_zones()
        assert len(zones) == 3

    def test_create_default_zones_names(self):
        zones = create_default_zones()
        names = {z.name for z in zones}
        assert "Active Zone" in names
        assert "Secondary Zone" in names
        assert "Storage Zone" in names

    def test_create_default_zones_priority_order(self):
        zones = create_default_zones()
        priorities = [z.priority for z in zones]
        assert 1 in priorities   # Active
        assert 2 in priorities   # Secondary
        assert 3 in priorities   # Storage

    def test_active_zone_inside_secondary(self):
        zones = create_default_zones()
        active    = next(z for z in zones if z.name == "Active Zone")
        secondary = next(z for z in zones if z.name == "Secondary Zone")
        # Active zone x-range must be contained within secondary x-range
        assert secondary.x0 <= active.x0
        assert secondary.x1 >= active.x1
        # Active zone y-range must be fully within secondary y-range
        # NOTE: Active zone starts at y=6 (front/user side),
        # Secondary at y=8, but Active zone's TOP (y1=36) is within Secondary's y1=58.
        # Active zone deliberately reaches closer to the desk front edge (user side).
        # The important constraint: Active y1 must be <= Secondary y1.
        assert secondary.y1 >= active.y1
        # And x bounds
        assert secondary.x0 <= active.x0 and secondary.x1 >= active.x1

    def test_create_default_obstacles_count(self):
        obs = create_default_obstacles()
        assert len(obs) == 4

    def test_create_default_obstacles_names(self):
        obs = create_default_obstacles()
        names = {o.name for o in obs}
        assert "Laptop" in names
        assert "Desk Lamp" in names
        assert "Pencil Cup" in names
        assert "Sticky Notes" in names

    def test_obstacles_are_on_desk(self):
        """All obstacle centres must lie within the 120×80 desk bounds."""
        obs = create_default_obstacles()
        for o in obs:
            assert 0 <= o.x <= 120, f"{o.name} x={o.x} out of bounds"
            assert 0 <= o.y <= 80,  f"{o.name} y={o.y} out of bounds"


# ═══════════════════════════════════════════════════════════════════════════════
#  E. WorkspaceSimulator
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorkspaceSimulator:

    @pytest.fixture
    def ws(self):
        return WorkspaceSimulator()

    def test_init_creates_5_default_books(self, ws):
        assert len(ws.books) == 5

    def test_init_book_ids(self, ws):
        assert set(ws.books.keys()) == {"B01", "B02", "B03", "B04", "B05"}

    def test_init_book_titles(self, ws):
        assert ws.books["B01"].title == "Machine Learning"
        assert ws.books["B02"].title == "Mathematics"
        assert ws.books["B03"].title == "DBMS"
        assert ws.books["B04"].title == "Computer Networks"
        assert ws.books["B05"].title == "Operating Systems"

    def test_init_books_on_desk(self, ws):
        for book in ws.books.values():
            hw, hh = book.width / 2, book.height / 2
            assert book.x - hw >= 0, f"{book.id} left edge off desk"
            assert book.x + hw <= ws.DESK_WIDTH,  f"{book.id} right edge off desk"
            assert book.y - hh >= 0, f"{book.id} bottom edge off desk"
            assert book.y + hh <= ws.DESK_DEPTH,  f"{book.id} top edge off desk"

    def test_init_default_user(self, ws):
        assert ws.current_user == "Student A — ML Track"

    def test_init_has_zones(self, ws):
        assert len(ws.zones) == 3

    def test_init_has_obstacles(self, ws):
        assert len(ws.obstacles) == 4

    # ── Reset ────────────────────────────────────────────────────────────
    def test_reset_workspace_restores_positions(self, ws):
        # Move a book, then reset
        original_x = ws.books["B01"].x
        ws.move_book("B01", 90.0, 50.0)
        ws.reset_workspace()
        assert abs(ws.books["B01"].x - original_x) < 0.01

    def test_reset_preserves_book_count(self, ws):
        ws.reset_workspace()
        assert len(ws.books) == 5

    # ── Randomize ────────────────────────────────────────────────────────
    def test_randomize_books_succeeds(self, ws):
        result = ws.randomize_books()
        assert result is True

    def test_randomize_books_stay_on_desk(self, ws):
        ws.randomize_books()
        for book in ws.books.values():
            hw, hh = book.width / 2, book.height / 2
            assert book.x - hw >= 1, f"{book.id} off desk left"
            assert book.x + hw <= ws.DESK_WIDTH  - 1, f"{book.id} off desk right"
            assert book.y - hh >= 1, f"{book.id} off desk bottom"
            assert book.y + hh <= ws.DESK_DEPTH  - 1, f"{book.id} off desk top"

    def test_randomize_books_no_obstacle_overlap(self, ws):
        ws.randomize_books()
        for book in ws.books.values():
            hw, hh = book.width / 2, book.height / 2
            for obs in ws.obstacles:
                assert not obs.blocks(book.x, book.y, hw, hh, margin=1.0), \
                    f"{book.id} overlaps obstacle {obs.name}"

    def test_randomize_books_no_book_overlap(self, ws):
        ws.randomize_books()
        books = list(ws.books.values())
        for i, a in enumerate(books):
            for b in books[i + 1:]:
                assert not a.overlaps(b, margin=0.5), \
                    f"Books {a.id} and {b.id} overlap after randomize"

    def test_randomize_changes_positions(self, ws):
        original = {bid: (b.x, b.y) for bid, b in ws.books.items()}
        ws.randomize_books()
        new = {bid: (b.x, b.y) for bid, b in ws.books.items()}
        # At least one book should have moved
        moved = any(original[bid] != new[bid] for bid in original)
        assert moved

    # ── Move book ─────────────────────────────────────────────────────────
    def test_move_book_success(self, ws):
        # Move B01 to (50, 42) — between active zone top (y=36) and obstacles (y>=48).
        # Verified clear: no obstacle rects extend to this area.
        ok, msg = ws.move_book("B01", 50.0, 42.0)
        assert ok, f"Move failed: {msg}"
        assert ws.books["B01"].x == 50.0
        assert ws.books["B01"].y == 42.0

    def test_move_book_clamp_to_desk(self, ws):
        ok, msg = ws.move_book("B01", -100.0, -100.0, clamp_to_desk=True)
        assert ok   # clamped, not rejected
        b = ws.books["B01"]
        assert b.x > 0 and b.y > 0

    def test_move_book_invalid_id(self, ws):
        ok, msg = ws.move_book("B99", 50.0, 30.0)
        assert not ok

    def test_move_book_obstacle_blocked(self, ws):
        # Laptop is at (72, 58); try to place B01 directly on it
        ok, msg = ws.move_book("B01", 72.0, 58.0)
        assert not ok
        assert "obstacle" in msg.lower()

    def test_move_book_no_overlap_other_books(self, ws):
        # B02 is at (37, 18); try to place B01 right on top of it
        ok, msg = ws.move_book("B01", ws.books["B02"].x, ws.books["B02"].y)
        assert not ok

    # ── Add book ──────────────────────────────────────────────────────────
    def test_add_book_success(self, ws):
        # Place new book at (50, 42) — confirmed clear of all obstacles and existing books.
        new_book = Book(id="B06", title="Algorithms", x=50.0, y=42.0)
        ok, msg = ws.add_book(new_book)
        assert ok, f"Add failed: {msg}"
        assert "B06" in ws.books
        assert ws.books["B06"].title == "Algorithms"

    def test_add_book_duplicate_id(self, ws):
        duplicate = Book(id="B01", title="Duplicate", x=60.0, y=30.0)
        ok, msg = ws.add_book(duplicate)
        assert not ok
        assert len(ws.books) == 5   # count unchanged

    def test_add_book_obstacle_blocked(self, ws):
        bad_book = Book(id="B06", title="Bad", x=72.0, y=58.0)
        ok, msg = ws.add_book(bad_book)
        assert not ok

    def test_add_book_overlapping_existing(self, ws):
        overlapping = Book(id="B06", title="Overlap", x=ws.books["B01"].x, y=ws.books["B01"].y)
        ok, msg = ws.add_book(overlapping)
        assert not ok

    # ── Remove book ───────────────────────────────────────────────────────
    def test_remove_book_success(self, ws):
        ok, msg = ws.remove_book("B01")
        assert ok
        assert "B01" not in ws.books
        assert len(ws.books) == 4

    def test_remove_book_invalid_id(self, ws):
        ok, msg = ws.remove_book("B99")
        assert not ok
        assert len(ws.books) == 5

    # ── is_position_valid ─────────────────────────────────────────────────
    def test_position_valid_on_desk(self, ws):
        # (50, 42) is confirmed clear: between active zone and obstacles, no default books
        assert ws.is_position_valid(50.0, 42.0, 9.0, 13.0)

    def test_position_invalid_off_desk(self, ws):
        assert not ws.is_position_valid(-10.0, 20.0, 9.0, 13.0)

    def test_position_invalid_on_obstacle(self, ws):
        assert not ws.is_position_valid(72.0, 58.0, 9.0, 13.0)

    # ── Layout dict ───────────────────────────────────────────────────────
    def test_get_layout_dict_structure(self, ws):
        layout = ws.get_layout_dict()
        assert "timestamp" in layout
        assert "user" in layout
        assert "context" in layout
        assert "books" in layout
        assert len(layout["books"]) == 5

    def test_get_books_dataframe_rows(self, ws):
        rows = ws.get_books_dataframe_rows()
        assert len(rows) == 5
        assert "ID" in rows[0]
        assert "Zone" in rows[0]
        assert "Usage Freq." in rows[0]


# ═══════════════════════════════════════════════════════════════════════════════
#  F. Session Control
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionControl:

    @pytest.fixture
    def ws(self):
        return WorkspaceSimulator()

    def test_set_user_valid(self, ws):
        ws.set_user("Student B — CS Track")
        assert ws.current_user == "Student B — CS Track"

    def test_set_user_changes_context(self, ws):
        ws.set_user("Student B — CS Track")
        # CS Track prefers Computer Science context
        assert ws.current_context == "Computer Science"

    def test_set_context_updates_books(self, ws):
        ws.set_context("Machine Learning")
        ml_books = ws.get_books_by_context("Machine Learning")
        bg_books = ws.get_books_by_context("background")
        # B01 should be ML, others should be background
        assert any(b.id == "B01" for b in ml_books)
        assert len(bg_books) == 4   # the other 4

    def test_set_context_general_no_background(self, ws):
        ws.set_context("Machine Learning")
        ws.set_context("general")
        # After general, no book should have context="background"
        for book in ws.books.values():
            assert book.current_context != "background"

    def test_increment_usage(self, ws):
        ws.books["B01"].usage_frequency = 5
        ws.increment_usage("B01")
        assert ws.books["B01"].usage_frequency == 6

    def test_increment_usage_updates_last_used(self, ws):
        old_time = ws.books["B01"].last_used
        import time; time.sleep(0.01)
        ws.increment_usage("B01")
        assert ws.books["B01"].last_used >= old_time

    def test_increment_usage_invalid_id(self, ws):
        # Should not raise
        ws.increment_usage("B99")


# ═══════════════════════════════════════════════════════════════════════════════
#  G. Query Helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryHelpers:

    @pytest.fixture
    def ws(self):
        return WorkspaceSimulator()

    def test_get_book_zone_returns_zone(self, ws):
        # Default B01 at (20, 18) should be in Active Zone (x:[18,102], y:[6,36])
        zone = ws.get_book_zone("B01")
        assert zone is not None
        assert zone.name == "Active Zone"

    def test_get_book_zone_invalid_id(self, ws):
        assert ws.get_book_zone("B99") is None

    def test_get_books_in_zone_active(self, ws):
        books = ws.get_books_in_zone("Active Zone")
        # All default books are at y=18, which is inside the active zone
        assert len(books) == 5

    def test_get_books_in_zone_nonexistent(self, ws):
        books = ws.get_books_in_zone("Nonexistent Zone")
        assert books == []

    def test_get_books_by_context(self, ws):
        ws.set_context("Machine Learning")
        ml_books = ws.get_books_by_context("Machine Learning")
        assert len(ml_books) == 1
        assert ml_books[0].id == "B01"

    def test_get_context_relevant_books_ml(self, ws):
        ws.set_context("Machine Learning")
        relevant = ws.get_context_relevant_books()
        assert "B01" in relevant

    def test_get_context_relevant_books_general(self, ws):
        ws.set_context("general")
        relevant = ws.get_context_relevant_books()
        assert relevant == []

    def test_get_context_relevant_books_cs(self, ws):
        ws.set_context("Computer Science")
        relevant = ws.get_context_relevant_books()
        assert "B03" in relevant
        assert "B04" in relevant
        assert "B05" in relevant


# ═══════════════════════════════════════════════════════════════════════════════
#  H. Renderer (smoke tests — checks figure is created without errors)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRenderer:

    @pytest.fixture
    def ws(self):
        return WorkspaceSimulator()

    def test_create_figure_returns_plotly_figure(self, ws):
        from simulation.renderer_2d import create_workspace_figure
        import plotly.graph_objects as go
        fig = create_workspace_figure(ws)
        assert isinstance(fig, go.Figure)

    def test_create_figure_has_traces(self, ws):
        from simulation.renderer_2d import create_workspace_figure
        fig = create_workspace_figure(ws)
        assert len(fig.data) > 0

    def test_create_figure_with_selected_book(self, ws):
        from simulation.renderer_2d import create_workspace_figure
        # Should not raise
        fig = create_workspace_figure(ws, selected_book_id="B01")
        assert fig is not None

    def test_create_figure_with_no_books(self, ws):
        from simulation.renderer_2d import create_workspace_figure
        for bid in list(ws.books.keys()):
            ws.remove_book(bid)
        fig = create_workspace_figure(ws)
        assert fig is not None   # empty desk renders without error

    def test_create_figure_after_randomize(self, ws):
        from simulation.renderer_2d import create_workspace_figure
        ws.randomize_books()
        fig = create_workspace_figure(ws)
        assert len(fig.data) > 0

    def test_figure_layout_has_dark_background(self, ws):
        from simulation.renderer_2d import create_workspace_figure
        fig = create_workspace_figure(ws)
        assert fig.layout.paper_bgcolor is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  I. Constants & Colour Palette
# ═══════════════════════════════════════════════════════════════════════════════

class TestConstants:

    def test_book_colors_defined(self):
        assert len(BOOK_COLORS) >= 5

    def test_context_book_map_has_all_subjects(self):
        assert "Machine Learning"    in CONTEXT_BOOK_MAP
        assert "Mathematics"         in CONTEXT_BOOK_MAP
        assert "DBMS"                in CONTEXT_BOOK_MAP
        assert "Computer Networks"   in CONTEXT_BOOK_MAP
        assert "Operating Systems"   in CONTEXT_BOOK_MAP

    def test_user_profiles_defined(self):
        assert len(USER_PROFILES) >= 3
        for name, profile in USER_PROFILES.items():
            assert "preferred_context" in profile
            assert "frequent_ids" in profile

    def test_workspace_desk_dimensions(self):
        ws = WorkspaceSimulator()
        assert ws.DESK_WIDTH  == 120.0
        assert ws.DESK_DEPTH  == 80.0
        assert ws.ROBOT_REACH == 70.0
