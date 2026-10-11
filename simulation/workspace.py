"""
simulation/workspace.py
WorkspaceSimulator — the core state engine for the virtual desk.

Responsibilities
----------------
• Maintain the complete desk state: books, zones, obstacles, robot, user
• Provide mutation operations: move_book, add_book, remove_book, randomize, reset
• Validate all changes against desk boundaries, zone rules, and obstacle geometry
• Expose query helpers used by the visualiser and future AI modules

No AI algorithms are implemented here (Phase 2 scope).
"""

from __future__ import annotations

import copy
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from simulation.book import Book, BOOK_COLORS
from simulation.zones import (
    Zone, Obstacle,
    create_default_zones, create_default_obstacles,
)


# ── Context → subject mapping ─────────────────────────────────────────────────
CONTEXT_BOOK_MAP: Dict[str, List[str]] = {
    "general":          [],                        # no specific affinity
    "Machine Learning": ["B01"],
    "Mathematics":      ["B02"],
    "DBMS":             ["B03"],
    "Computer Networks":["B04"],
    "Operating Systems":["B05"],
    "Computer Science": ["B03", "B04", "B05"],
    "AI & ML":          ["B01", "B02"],
}

# ── Default user profiles ─────────────────────────────────────────────────────
USER_PROFILES: Dict[str, Dict] = {
    "Student A — ML Track": {
        "preferred_context": "Machine Learning",
        "frequent_ids": ["B01", "B02"],
    },
    "Student B — CS Track": {
        "preferred_context": "Computer Science",
        "frequent_ids": ["B03", "B04", "B05"],
    },
    "Student C — Generalist": {
        "preferred_context": "general",
        "frequent_ids": ["B01", "B02", "B03", "B04", "B05"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  WorkspaceSimulator
# ═══════════════════════════════════════════════════════════════════════════════

class WorkspaceSimulator:
    """
    Full state of the virtual study desk.

    Coordinate system
    -----------------
    All positions are in centimetres.
    Origin: bottom-left corner of the desk surface.
    X : left (0) → right (DESK_WIDTH)
    Y : front/user-edge (0) → back (DESK_DEPTH)

    Usage
    -----
        ws = WorkspaceSimulator()
        ws.reset_workspace()
        ws.randomize_books()
        ws.move_book("B01", 45.0, 20.0)
        layout = ws.get_layout_dict()
    """

    DESK_WIDTH:  float = 120.0   # cm
    DESK_DEPTH:  float = 80.0    # cm

    # Robot arm base position (right-back quadrant, on desk edge)
    ROBOT_BASE:  Tuple[float, float] = (60.0, 84.0)   # above desk boundary
    ROBOT_REACH: float = 70.0    # cm (max reach radius)

    # User position (in front of desk)
    USER_POS:    Tuple[float, float] = (60.0, -8.0)

    # ── Default book specifications ────────────────────────────────────────
    _DEFAULT_BOOKS: List[Dict] = [
        dict(id="B01", title="Machine Learning",      x=20.0, y=18.0,
             color=BOOK_COLORS[0], current_context="Machine Learning",   usage_frequency=12),
        dict(id="B02", title="Mathematics",           x=37.0, y=18.0,
             color=BOOK_COLORS[1], current_context="Mathematics",        usage_frequency=18),
        dict(id="B03", title="DBMS",                  x=60.0, y=18.0,
             color=BOOK_COLORS[2], current_context="DBMS",               usage_frequency=7),
        dict(id="B04", title="Computer Networks",     x=83.0, y=18.0,
             color=BOOK_COLORS[3], current_context="Computer Networks",  usage_frequency=5),
        dict(id="B05", title="Operating Systems",     x=100.0, y=18.0,
             color=BOOK_COLORS[4], current_context="Operating Systems",  usage_frequency=9),
    ]

    # ── Constructor ───────────────────────────────────────────────────────
    def __init__(self) -> None:
        self.zones:        List[Zone]        = create_default_zones()
        self.obstacles:    List[Obstacle]    = create_default_obstacles()
        self.books:        Dict[str, Book]   = {}
        self.current_user: str               = "Student A — ML Track"
        self.current_context: str            = "general"
        self._next_color_idx: int            = len(self._DEFAULT_BOOKS)

        self._initialize_books()

    # ── Initialisation ────────────────────────────────────────────────────
    def _initialize_books(self) -> None:
        """Load the five default books at their canonical positions."""
        for spec in self._DEFAULT_BOOKS:
            book = Book(
                id=spec["id"],
                title=spec["title"],
                x=spec["x"],
                y=spec["y"],
                color=spec["color"],
                current_context=spec["current_context"],
                usage_frequency=spec["usage_frequency"],
                last_used=datetime.now(),
            )
            self.books[book.id] = book

    # ── Reset / Randomise ─────────────────────────────────────────────────
    def reset_workspace(self) -> None:
        """Restore every book to its canonical default position."""
        for spec in self._DEFAULT_BOOKS:
            if spec["id"] in self.books:
                book = self.books[spec["id"]]
                book.x = spec["x"]
                book.y = spec["y"]

    def randomize_books(self, max_attempts: int = 500) -> bool:
        """
        Place all books at random valid positions on the desk.

        Validity rules
        --------------
        1. Book bounding box stays within desk boundary (with 2 cm border).
        2. Book does not overlap any fixed obstacle (with 1.5 cm margin).
        3. Books do not overlap each other (with 1.5 cm margin).

        Returns True if all books were successfully placed, False on timeout.
        """
        books_list = list(self.books.values())
        placed: List[Book] = []

        for book in books_list:
            hw, hh = book.width / 2, book.height / 2
            border = 2.0

            placed_successfully = False
            for _ in range(max_attempts):
                cx = random.uniform(border + hw, self.DESK_WIDTH  - border - hw)
                cy = random.uniform(border + hh, self.DESK_DEPTH  - border - hh)

                # Check obstacle collision
                if any(obs.blocks(cx, cy, hw, hh) for obs in self.obstacles):
                    continue

                # Check collision with already-placed books
                temp = Book(id="_tmp", title="", x=cx, y=cy,
                            width=book.width, height=book.height)
                if any(temp.overlaps(p) for p in placed):
                    continue

                book.x, book.y = cx, cy
                placed.append(book)
                placed_successfully = True
                break

            if not placed_successfully:
                return False   # placement timed out

        return True

    # ── Book mutations ────────────────────────────────────────────────────
    def move_book(
        self, book_id: str, new_x: float, new_y: float,
        clamp_to_desk: bool = True
    ) -> Tuple[bool, str]:
        """
        Move a book to (new_x, new_y).

        Returns (success: bool, message: str).
        If clamp_to_desk is True, the position is clamped to keep the book
        fully on the desk surface; otherwise out-of-bounds moves are rejected.
        """
        if book_id not in self.books:
            return False, f"Book '{book_id}' not found."

        book = self.books[book_id]
        hw, hh = book.width / 2, book.height / 2
        border = 1.0

        if clamp_to_desk:
            new_x = max(hw + border, min(self.DESK_WIDTH  - hw - border, new_x))
            new_y = max(hh + border, min(self.DESK_DEPTH  - hh - border, new_y))
        else:
            if (new_x - hw < 0 or new_x + hw > self.DESK_WIDTH
                    or new_y - hh < 0 or new_y + hw > self.DESK_DEPTH):
                return False, "Position out of desk bounds."

        # Obstacle check
        if any(obs.blocks(new_x, new_y, hw, hh) for obs in self.obstacles):
            return False, "Position conflicts with a desk obstacle."

        # Overlap check (exclude self)
        temp = Book(id="_tmp", title="", x=new_x, y=new_y,
                    width=book.width, height=book.height)
        for bid, other in self.books.items():
            if bid != book_id and temp.overlaps(other):
                return False, f"Position overlaps with book '{bid}'."

        book.x, book.y = new_x, new_y
        return True, "Book moved successfully."

    def add_book(self, book: Book) -> Tuple[bool, str]:
        """
        Add a new book to the workspace.
        Returns (success, message).
        """
        if book.id in self.books:
            return False, f"Book ID '{book.id}' already exists."

        # Assign next colour if none specified
        if not book.color or book.color == "#4299e1":
            book.color = BOOK_COLORS[self._next_color_idx % len(BOOK_COLORS)]
            self._next_color_idx += 1

        # Validate position
        hw, hh = book.width / 2, book.height / 2
        if any(obs.blocks(book.x, book.y, hw, hh) for obs in self.obstacles):
            return False, "Requested position conflicts with an obstacle."

        temp = Book(id="_tmp", title="", x=book.x, y=book.y,
                    width=book.width, height=book.height)
        for other in self.books.values():
            if temp.overlaps(other):
                return False, f"Requested position overlaps book '{other.id}'."

        self.books[book.id] = book
        return True, f"Book '{book.id}' added."

    def remove_book(self, book_id: str) -> Tuple[bool, str]:
        """Remove a book from the workspace. Returns (success, message)."""
        if book_id not in self.books:
            return False, f"Book '{book_id}' not found."
        del self.books[book_id]
        return True, f"Book '{book_id}' removed."

    # ── Session / context ─────────────────────────────────────────────────
    def set_user(self, user_name: str) -> None:
        """Set the active user profile and apply default context."""
        if user_name in USER_PROFILES:
            self.current_user = user_name
            pref = USER_PROFILES[user_name]["preferred_context"]
            self.set_context(pref)

    def set_context(self, context: str) -> None:
        """
        Set the current study context.
        Updates each book's `current_context` to indicate which are
        *currently relevant* to the session.
        """
        self.current_context = context
        relevant_ids = CONTEXT_BOOK_MAP.get(context, [])

        for book in self.books.values():
            if relevant_ids:
                book.current_context = context if book.id in relevant_ids else "background"
            else:
                # 'general' — each book keeps its own subject context
                book.current_context = book.current_context if book.current_context != "background" else "general"

    def increment_usage(self, book_id: str) -> None:
        """Record one interaction with a book."""
        if book_id in self.books:
            self.books[book_id].usage_frequency += 1
            self.books[book_id].last_used = datetime.now()

    # ── Queries ───────────────────────────────────────────────────────────
    def get_book_zone(self, book_id: str) -> Optional[Zone]:
        """Return the highest-priority zone that contains the book's centre."""
        if book_id not in self.books:
            return None
        book = self.books[book_id]
        # Zones are ordered storage → secondary → active (priority 3→1)
        # We want the *highest priority* (smallest number) that contains the book
        matched = [z for z in self.zones if z.contains(book.x, book.y)]
        if not matched:
            return None
        return min(matched, key=lambda z: z.priority)

    def get_books_in_zone(self, zone_name: str) -> List[Book]:
        """Return all books whose centre lies in the named zone."""
        zone = next((z for z in self.zones if z.name == zone_name), None)
        if zone is None:
            return []
        return [b for b in self.books.values() if zone.contains(b.x, b.y)]

    def get_books_by_context(self, context: str) -> List[Book]:
        """Return books that belong to a given subject context."""
        return [b for b in self.books.values() if b.current_context == context]

    def get_context_relevant_books(self) -> List[str]:
        """Return IDs of books relevant to the current session context."""
        return CONTEXT_BOOK_MAP.get(self.current_context, [])

    def is_position_valid(
        self, cx: float, cy: float, width: float, height: float,
        exclude_id: Optional[str] = None
    ) -> bool:
        """
        Return True if a rectangle centred at (cx, cy) with given dimensions
        can be placed without violating any constraint.
        """
        hw, hh = width / 2, height / 2
        border = 1.0
        # Desk boundary
        if (cx - hw < border or cx + hw > self.DESK_WIDTH - border
                or cy - hh < border or cy + hh > self.DESK_DEPTH - border):
            return False
        # Obstacles
        if any(obs.blocks(cx, cy, hw, hh) for obs in self.obstacles):
            return False
        # Books
        temp = Book(id="_tmp", title="", x=cx, y=cy, width=width, height=height)
        for bid, book in self.books.items():
            if bid != exclude_id and temp.overlaps(book):
                return False
        return True

    def get_layout_dict(self) -> Dict:
        """Return the current workspace state as a plain dictionary."""
        return {
            "timestamp": datetime.now().isoformat(),
            "user":      self.current_user,
            "context":   self.current_context,
            "books":     {bid: b.to_dict() for bid, b in self.books.items()},
        }

    def get_books_dataframe_rows(self) -> List[Dict]:
        """Return book data as rows suitable for a pandas/Streamlit dataframe."""
        rows = []
        for book in sorted(self.books.values(), key=lambda b: b.id):
            zone = self.get_book_zone(book.id)
            rows.append({
                "ID":              book.id,
                "Title":           book.title,
                "X (cm)":          round(book.x, 1),
                "Y (cm)":          round(book.y, 1),
                "Width":           book.width,
                "Height":          book.height,
                "Usage Freq.":     book.usage_frequency,
                "Last Used":       book.last_used.strftime("%H:%M:%S") if book.last_used else "—",
                "Context":         book.current_context,
                "Zone":            zone.name if zone else "Off Desk",
            })
        return rows

    # ── Robot helpers (stubs for future phases) ───────────────────────────
    @property
    def robot_base(self) -> Tuple[float, float]:
        return self.ROBOT_BASE

    @property
    def user_position(self) -> Tuple[float, float]:
        return self.USER_POS

    def __repr__(self) -> str:
        return (
            f"WorkspaceSimulator(books={len(self.books)}, "
            f"user={self.current_user!r}, context={self.current_context!r})"
        )
