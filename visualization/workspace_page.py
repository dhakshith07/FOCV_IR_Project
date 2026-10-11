"""
visualization/workspace_page.py
Streamlit page — Virtual Workspace Simulator.

Renders the complete Phase 2 workspace UI:
  • Interactive 2D Plotly desk figure
  • Book control panel (move, details, usage)
  • Workspace control panel (randomize, reset, add, remove)
  • Session control panel (user, context)
  • Full book properties dataframe
"""

from __future__ import annotations

import random
from datetime import datetime

import pandas as pd
import streamlit as st

from simulation.book import Book, BOOK_COLORS
from simulation.workspace import WorkspaceSimulator, USER_PROFILES, CONTEXT_BOOK_MAP
from simulation.renderer_2d import create_workspace_figure


# ── Shared CSS injected once ─────────────────────────────────────────────────
_CSS = """
<style>
.ws-header {
    background: linear-gradient(135deg, #1a2744 0%, #1e3a5f 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1rem;
}
.ws-title {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #90cdf4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.ws-meta {
    font-size: 0.8rem;
    color: #4a5568;
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}
.info-row {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.6rem;
    flex-wrap: wrap;
}
.info-pill {
    background: rgba(99,179,237,0.12);
    border: 1px solid rgba(99,179,237,0.25);
    color: #63b3ed;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.stat-val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #63b3ed;
    font-family: 'JetBrains Mono', monospace;
}
.stat-lbl {
    font-size: 0.7rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.ctrl-section {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.ctrl-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #a0aec0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.7rem;
}
.book-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    margin-right: 6px;
    vertical-align: middle;
}
.msg-ok   { color: #48bb78; font-size: 0.8rem; padding: 0.3rem 0; }
.msg-err  { color: #fc8181; font-size: 0.8rem; padding: 0.3rem 0; }
</style>
"""


# ── Session-state initialisation ─────────────────────────────────────────────
def _init_state() -> None:
    if "workspace" not in st.session_state:
        st.session_state.workspace = WorkspaceSimulator()
    if "selected_book_id" not in st.session_state:
        st.session_state.selected_book_id = "B01"
    if "ws_message" not in st.session_state:
        st.session_state.ws_message = ("", "")   # (text, "ok"|"err")
    if "show_grid" not in st.session_state:
        st.session_state.show_grid = True
    if "show_zone_labels" not in st.session_state:
        st.session_state.show_zone_labels = True


def _ws() -> WorkspaceSimulator:
    return st.session_state.workspace


def _set_msg(text: str, kind: str = "ok") -> None:
    st.session_state.ws_message = (text, kind)


# ── Stat card HTML ────────────────────────────────────────────────────────────
def _stat(val: str, lbl: str) -> str:
    return f"""
    <div class="stat-card">
        <div class="stat-val">{val}</div>
        <div class="stat-lbl">{lbl}</div>
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Main render entry point
# ═══════════════════════════════════════════════════════════════════════════════

def render_workspace_page() -> None:
    """Called from app.py when the user selects the Workspace page."""
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state()

    ws = _ws()

    # ── Header ────────────────────────────────────────────────────────────
    ctx_pill  = f'<span class="info-pill">📚 {ws.current_context}</span>'
    user_pill = f'<span class="info-pill">👤 {ws.current_user}</span>'
    book_pill = f'<span class="info-pill">📖 {len(ws.books)} books</span>'
    zone_pill = f'<span class="info-pill">🗺️ {len(ws.zones)} zones</span>'
    obs_pill  = f'<span class="info-pill">🚧 {len(ws.obstacles)} obstacles</span>'

    st.markdown(f"""
    <div class="ws-header">
        <p class="ws-title">🖥️ Virtual Workspace Simulator</p>
        <p class="ws-meta">Phase 2 — 2D Desk Environment (120 × 80 cm)</p>
        <div class="info-row">
            {user_pill}{ctx_pill}{book_pill}{zone_pill}{obs_pill}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stat bar ──────────────────────────────────────────────────────────
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    total_usage = sum(b.usage_frequency for b in ws.books.values())
    most_used   = max(ws.books.values(), key=lambda b: b.usage_frequency, default=None)
    active_bks  = len(ws.get_books_in_zone("Active Zone"))

    sc1.markdown(_stat(str(len(ws.books)), "Books on Desk"), unsafe_allow_html=True)
    sc2.markdown(_stat(str(len(ws.zones)), "Zones"), unsafe_allow_html=True)
    sc3.markdown(_stat(str(len(ws.obstacles)), "Obstacles"), unsafe_allow_html=True)
    sc4.markdown(_stat(str(total_usage), "Total Usage"), unsafe_allow_html=True)
    sc5.markdown(_stat(most_used.id if most_used else "—", "Top Book"), unsafe_allow_html=True)

    st.markdown("")

    # ── Main layout: figure (left) + controls (right) ─────────────────────
    fig_col, ctrl_col = st.columns([3, 2], gap="medium")

    # ── Workspace figure ──────────────────────────────────────────────────
    with fig_col:
        # Render options row
        ro1, ro2, ro3 = st.columns([1, 1, 2])
        with ro1:
            st.session_state.show_grid = st.toggle("Grid", value=st.session_state.show_grid, key="tgl_grid")
        with ro2:
            st.session_state.show_zone_labels = st.toggle("Labels", value=st.session_state.show_zone_labels, key="tgl_labels")
        with ro3:
            # Global message bar
            msg_text, msg_kind = st.session_state.ws_message
            if msg_text:
                css_class = "msg-ok" if msg_kind == "ok" else "msg-err"
                icon = "✅" if msg_kind == "ok" else "❌"
                st.markdown(f'<div class="{css_class}">{icon} {msg_text}</div>',
                            unsafe_allow_html=True)

        fig = create_workspace_figure(
            ws,
            selected_book_id=st.session_state.selected_book_id,
            show_zone_labels=st.session_state.show_zone_labels,
            show_grid=st.session_state.show_grid,
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True, config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False,
        })

    # ── Control panel ─────────────────────────────────────────────────────
    with ctrl_col:
        tab_books, tab_workspace, tab_session = st.tabs([
            "📚 Books", "⚙️ Workspace", "👤 Session"
        ])

        # ── TAB 1: Book Controls ──────────────────────────────────────────
        with tab_books:
            _render_book_tab(ws)

        # ── TAB 2: Workspace Controls ─────────────────────────────────────
        with tab_workspace:
            _render_workspace_tab(ws)

        # ── TAB 3: Session Controls ───────────────────────────────────────
        with tab_session:
            _render_session_tab(ws)

    # ── Book properties table ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Book Properties")

    rows = ws.get_books_dataframe_rows()
    if rows:
        df = pd.DataFrame(rows)
        # Add colour indicator column
        df.insert(0, "🎨", [ws.books[r["ID"]].color for r in rows])

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=220,
            column_config={
                "🎨": st.column_config.TextColumn("Color", width="small"),
                "ID": st.column_config.TextColumn("ID", width="small"),
                "Usage Freq.": st.column_config.ProgressColumn(
                    "Usage", min_value=0, max_value=30, format="%d×"
                ),
                "X (cm)": st.column_config.NumberColumn(format="%.1f cm"),
                "Y (cm)": st.column_config.NumberColumn(format="%.1f cm"),
            },
        )

    # ── Zone summary ──────────────────────────────────────────────────────
    st.markdown("#### 🗺️ Zone Distribution")
    zcols = st.columns(len(ws.zones))
    for zi, zone in enumerate(sorted(ws.zones, key=lambda z: z.priority)):
        books_in_zone = ws.get_books_in_zone(zone.name)
        with zcols[zi]:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                        border-radius:8px;padding:0.7rem;text-align:center;">
                <div style="font-size:0.75rem;font-weight:700;color:#a0aec0;
                            text-transform:uppercase;letter-spacing:0.05em;">{zone.name}</div>
                <div style="font-size:1.4rem;font-weight:700;color:#63b3ed;">{len(books_in_zone)}</div>
                <div style="font-size:0.7rem;color:#4a5568;">books</div>
                <div style="font-size:0.7rem;color:#718096;margin-top:0.3rem;">
                    {', '.join(b.id for b in books_in_zone) if books_in_zone else '—'}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tab renderers
# ═══════════════════════════════════════════════════════════════════════════════

def _render_book_tab(ws: WorkspaceSimulator) -> None:
    """📚 Books tab — select, inspect, move, and record usage."""

    st.markdown('<div class="ctrl-title">📌 Select Book</div>', unsafe_allow_html=True)

    book_ids = list(ws.books.keys())
    if not book_ids:
        st.info("No books on the desk.")
        return

    # Book selector with colour dot
    def book_label(bid: str) -> str:
        b = ws.books[bid]
        return f"{b.id} — {b.title}"

    sel = st.selectbox(
        "Book",
        options=book_ids,
        format_func=book_label,
        index=book_ids.index(st.session_state.selected_book_id)
               if st.session_state.selected_book_id in book_ids else 0,
        key="book_selector",
        label_visibility="collapsed",
    )
    st.session_state.selected_book_id = sel
    book = ws.books[sel]

    # Book detail card
    st.markdown(f"""
    <div class="ctrl-section">
        <span class="book-dot" style="background:{book.color}"></span>
        <b style="color:#e2e8f0">{book.title}</b><br>
        <small style="color:#4a5568;font-family:JetBrains Mono,monospace">
            ID: {book.id} &nbsp;|&nbsp;
            Pos: ({book.x:.1f}, {book.y:.1f}) cm &nbsp;|&nbsp;
            {book.width:.0f}×{book.height:.0f} cm
        </small><br>
        <small style="color:#718096">
            Context: {book.current_context} &nbsp;|&nbsp;
            Used: {book.usage_frequency}×
        </small>
    </div>
    """, unsafe_allow_html=True)

    # Zone indicator
    zone = ws.get_book_zone(sel)
    if zone:
        zone_colors = {"Active Zone": "#48bb78", "Secondary Zone": "#63b3ed", "Storage Zone": "#a0aec0"}
        zc = zone_colors.get(zone.name, "#718096")
        st.markdown(
            f'<span style="font-size:0.8rem;color:{zc};font-weight:600;">📍 Currently in: {zone.name}</span>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<div class="ctrl-title">📐 Move Book</div>', unsafe_allow_html=True)

    new_x = st.slider(
        "X Position (cm)", 0.0, ws.DESK_WIDTH,
        float(book.x), step=0.5, key=f"slide_x_{sel}"
    )
    new_y = st.slider(
        "Y Position (cm)", 0.0, ws.DESK_DEPTH,
        float(book.y), step=0.5, key=f"slide_y_{sel}"
    )

    mc1, mc2 = st.columns(2)
    with mc1:
        if st.button("📌 Move Book", use_container_width=True, key="btn_move"):
            ok, msg = ws.move_book(sel, new_x, new_y)
            _set_msg(msg, "ok" if ok else "err")
            st.rerun()
    with mc2:
        if st.button("🎯 Snap to Zone", use_container_width=True, key="btn_snap"):
            # Place in active zone centre
            az = next((z for z in ws.zones if z.name == "Active Zone"), None)
            if az:
                zx, zy = az.center
                ok, msg = ws.move_book(sel, zx, zy)
                _set_msg(msg, "ok" if ok else "err")
                st.rerun()

    st.markdown("---")
    st.markdown('<div class="ctrl-title">📊 Usage Tracking</div>', unsafe_allow_html=True)
    if st.button(f"➕ Record Usage for {sel}", use_container_width=True, key="btn_usage"):
        ws.increment_usage(sel)
        _set_msg(f"Usage recorded for {sel} ({ws.books[sel].usage_frequency}×)", "ok")
        st.rerun()

    # Usage bar
    max_u = max((b.usage_frequency for b in ws.books.values()), default=1) or 1
    st.progress(book.usage_frequency / max_u,
                text=f"Usage: {book.usage_frequency}× (relative to most-used)")


def _render_workspace_tab(ws: WorkspaceSimulator) -> None:
    """⚙️ Workspace tab — randomize, reset, add, remove."""

    # Quick actions
    st.markdown('<div class="ctrl-title">⚡ Quick Actions</div>', unsafe_allow_html=True)
    qa1, qa2 = st.columns(2)
    with qa1:
        if st.button("🎲 Randomize", use_container_width=True, key="btn_rand",
                     help="Place all books at random valid positions"):
            success = ws.randomize_books()
            _set_msg(
                "Books randomized successfully!" if success
                else "Randomization timed out — desk may be too crowded.",
                "ok" if success else "err"
            )
            st.rerun()
    with qa2:
        if st.button("🔄 Reset", use_container_width=True, key="btn_reset",
                     help="Return all books to their default positions"):
            ws.reset_workspace()
            _set_msg("Workspace reset to default positions.", "ok")
            st.rerun()

    st.markdown("---")

    # Add book form
    st.markdown('<div class="ctrl-title">➕ Add New Book</div>', unsafe_allow_html=True)
    with st.form("form_add_book", clear_on_submit=True):
        fi1, fi2 = st.columns(2)
        with fi1:
            new_id    = st.text_input("ID", placeholder="B06")
        with fi2:
            new_title = st.text_input("Title", placeholder="Graph Theory")

        fi3, fi4 = st.columns(2)
        with fi3:
            new_x = st.number_input("X (cm)", 5.0, 115.0, 60.0, step=5.0)
        with fi4:
            new_y = st.number_input("Y (cm)", 5.0,  75.0, 25.0, step=5.0)

        new_ctx = st.selectbox(
            "Context",
            options=list(CONTEXT_BOOK_MAP.keys()),
            index=0,
        )
        add_btn = st.form_submit_button("➕ Add Book", use_container_width=True)

    if add_btn:
        if not new_id or not new_title:
            _set_msg("Book ID and Title are required.", "err")
        else:
            colour_idx = len(ws.books)
            new_book = Book(
                id=new_id.strip().upper(),
                title=new_title.strip(),
                x=float(new_x),
                y=float(new_y),
                color=BOOK_COLORS[colour_idx % len(BOOK_COLORS)],
                current_context=new_ctx,
                usage_frequency=0,
                last_used=datetime.now(),
            )
            ok, msg = ws.add_book(new_book)
            _set_msg(msg, "ok" if ok else "err")
        st.rerun()

    st.markdown("---")

    # Remove book
    st.markdown('<div class="ctrl-title">🗑️ Remove Book</div>', unsafe_allow_html=True)
    if ws.books:
        rem_id = st.selectbox(
            "Select to remove",
            options=list(ws.books.keys()),
            format_func=lambda bid: f"{bid} — {ws.books[bid].title}",
            key="remove_select",
            label_visibility="collapsed",
        )
        if st.button(f"🗑️ Remove {rem_id}", use_container_width=True,
                     key="btn_remove", type="secondary"):
            ok, msg = ws.remove_book(rem_id)
            if ok and st.session_state.selected_book_id == rem_id:
                remaining = list(ws.books.keys())
                st.session_state.selected_book_id = remaining[0] if remaining else ""
            _set_msg(msg, "ok" if ok else "err")
            st.rerun()
    else:
        st.caption("No books to remove.")


def _render_session_tab(ws: WorkspaceSimulator) -> None:
    """👤 Session tab — select user profile and study context."""

    st.markdown('<div class="ctrl-title">👤 User Profile</div>', unsafe_allow_html=True)
    user_names = list(USER_PROFILES.keys())
    selected_user = st.radio(
        "User",
        options=user_names,
        index=user_names.index(ws.current_user) if ws.current_user in user_names else 0,
        key="user_radio",
        label_visibility="collapsed",
    )
    if selected_user != ws.current_user:
        ws.set_user(selected_user)
        _set_msg(f"User switched to: {selected_user}", "ok")
        st.rerun()

    # User profile details
    profile = USER_PROFILES.get(selected_user, {})
    st.markdown(f"""
    <div class="ctrl-section">
        <small style="color:#718096">
            Preferred Context: <b style="color:#63b3ed">{profile.get('preferred_context','—')}</b><br>
            Frequently Used: <b style="color:#63b3ed">{', '.join(profile.get('frequent_ids',[]))}</b>
        </small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="ctrl-title">📚 Study Context</div>', unsafe_allow_html=True)

    ctx_options = list(CONTEXT_BOOK_MAP.keys())
    sel_ctx = st.selectbox(
        "Context",
        options=ctx_options,
        index=ctx_options.index(ws.current_context) if ws.current_context in ctx_options else 0,
        key="ctx_select",
        label_visibility="collapsed",
    )
    if st.button("🔄 Apply Context", use_container_width=True, key="btn_ctx"):
        ws.set_context(sel_ctx)
        _set_msg(f"Context updated: {sel_ctx}", "ok")
        st.rerun()

    # Context → books mapping
    relevant = CONTEXT_BOOK_MAP.get(sel_ctx, [])
    st.markdown(f"""
    <div class="ctrl-section">
        <div class="ctrl-title">Relevant Books for Context</div>
        <small style="color:#718096">
            {"<b style='color:#63b3ed'>" + ', '.join(relevant) + "</b>"
             if relevant else "All books (general context)"}
        </small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="ctrl-title">📊 Session Statistics</div>', unsafe_allow_html=True)

    # Book usage ranking
    sorted_books = sorted(ws.books.values(), key=lambda b: b.usage_frequency, reverse=True)
    for i, book in enumerate(sorted_books[:5], 1):
        max_u = sorted_books[0].usage_frequency if sorted_books else 1
        frac  = book.usage_frequency / max(max_u, 1)
        bar_w = int(frac * 100)
        dot_c = book.color
        st.markdown(f"""
        <div style="margin-bottom:0.4rem;">
            <span class="book-dot" style="background:{dot_c}"></span>
            <span style="color:#a0aec0;font-size:0.8rem;">{i}. {book.title[:22]}</span>
            <span style="float:right;color:#63b3ed;font-size:0.8rem;
                         font-family:JetBrains Mono,monospace">{book.usage_frequency}×</span>
            <div style="height:3px;background:rgba(255,255,255,0.05);border-radius:2px;margin-top:2px;">
                <div style="height:3px;width:{bar_w}%;background:{dot_c};border-radius:2px;
                             opacity:0.7;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
