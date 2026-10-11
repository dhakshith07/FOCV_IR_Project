"""
simulation/renderer_2d.py
Interactive 2D top-down workspace renderer using Plotly.

Produces a go.Figure that renders the complete desk state:
  • Desk surface (warm parchment background)
  • Named zones (colour-coded transparent overlays with dashed borders)
  • Fixed obstacles (dark rectangles with emoji labels)
  • Books (coloured rectangles with usage-frequency heat tint, rich hover)
  • Robot arm base indicator
  • User position indicator
  • Zone legend annotations

No Streamlit imports — this module is UI-framework-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from simulation.workspace import WorkspaceSimulator


# ── Visual constants ─────────────────────────────────────────────────────────
DESK_BG       = "#f5efe0"          # warm parchment
DESK_BORDER   = "#8b7355"
GRID_COLOR    = "rgba(139,115,85,0.12)"
PLOT_BG       = "#1a1f2e"          # dark outer background
FONT_FAMILY   = "Inter, sans-serif"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #rrggbb to rgba(r,g,b,alpha)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _make_rect_trace(
    x0: float, y0: float, x1: float, y1: float,
    fill: str, line_color: str, line_width: float,
    name: str, hover: str = "", show_legend: bool = False,
    opacity: float = 1.0,
) -> go.Scatter:
    """Return a closed-polygon Scatter trace that renders as a filled rectangle."""
    xs = [x0, x1, x1, x0, x0]
    ys = [y0, y0, y1, y1, y0]
    return go.Scatter(
        x=xs, y=ys,
        fill="toself",
        fillcolor=fill,
        line=dict(color=line_color, width=line_width),
        mode="lines",
        name=name,
        hovertemplate=hover if hover else None,
        hoverinfo="skip" if not hover else "text",
        showlegend=show_legend,
        opacity=opacity,
    )


def create_workspace_figure(
    workspace: "WorkspaceSimulator",
    selected_book_id: str | None = None,
    show_zone_labels: bool = True,
    show_grid: bool = True,
    height: int = 620,
) -> go.Figure:
    """
    Build and return a Plotly Figure of the current workspace state.

    Parameters
    ----------
    workspace        : The WorkspaceSimulator whose state is rendered.
    selected_book_id : Optional book ID to highlight with a brighter border.
    show_zone_labels : Whether to render zone name annotations.
    show_grid        : Whether to show desk grid lines.
    height           : Figure height in pixels.
    """

    fig = go.Figure()
    W, D = workspace.DESK_WIDTH, workspace.DESK_DEPTH
    relevant_ids = workspace.get_context_relevant_books()

    # ── 1. Desk surface ────────────────────────────────────────────────────
    fig.add_trace(_make_rect_trace(
        0, 0, W, D,
        fill=DESK_BG, line_color=DESK_BORDER, line_width=2.5,
        name="Desk", hover="",
    ))

    # ── 2. Grid lines ─────────────────────────────────────────────────────
    if show_grid:
        for gx in range(0, int(W) + 1, 10):
            fig.add_shape(type="line", x0=gx, y0=0, x1=gx, y1=D,
                          line=dict(color=GRID_COLOR, width=0.5))
        for gy in range(0, int(D) + 1, 10):
            fig.add_shape(type="line", x0=0, y0=gy, x1=W, y1=gy,
                          line=dict(color=GRID_COLOR, width=0.5))

    # ── 3. Zones (drawn lowest-priority first so active zone is on top) ───
    for zone in sorted(workspace.zones, key=lambda z: z.priority, reverse=True):
        fig.add_trace(_make_rect_trace(
            zone.x0, zone.y0, zone.x1, zone.y1,
            fill=zone.color,
            line_color=zone.border_color,
            line_width=1.2,
            name=zone.name,
            hover=(
                f"<b>{zone.name}</b><br>"
                f"Priority: {zone.priority}<br>"
                f"{zone.description}<extra></extra>"
            ),
        ))
        if show_zone_labels:
            lx, ly = zone.label_pos
            fig.add_annotation(
                x=lx, y=ly,
                text=f"<b>{zone.name}</b>",
                showarrow=False,
                font=dict(size=9, color=zone.border_color.replace("0.35", "0.85").replace("0.45", "0.95").replace("0.25", "0.75")),
                xanchor="left",
                bgcolor="rgba(26,31,46,0.55)",
                borderpad=2,
            )

    # ── 4. Obstacles ──────────────────────────────────────────────────────
    for obs in workspace.obstacles:
        # Filled rectangle
        fig.add_trace(_make_rect_trace(
            obs.x0, obs.y0, obs.x1, obs.y1,
            fill=obs.color,
            line_color="#1a202c",
            line_width=1.5,
            name=obs.name,
            hover=(
                f"<b>{obs.icon} {obs.name}</b><br>"
                f"Position: ({obs.x:.0f}, {obs.y:.0f}) cm<br>"
                f"Size: {obs.width:.0f}×{obs.height:.0f} cm<br>"
                f"<i>Fixed obstacle — unmovable</i><extra></extra>"
            ),
        ))
        # Icon + label
        fig.add_trace(go.Scatter(
            x=[obs.x], y=[obs.y],
            mode="text",
            text=[obs.icon],
            textfont=dict(size=14),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_annotation(
            x=obs.x, y=obs.y0 - 1.2,
            text=f"<b>{obs.name}</b>",
            showarrow=False,
            font=dict(size=8, color="#a0aec0"),
            yanchor="top",
        )

    # ── 5. Books ──────────────────────────────────────────────────────────
    for book in workspace.books.values():
        is_selected  = (book.id == selected_book_id)
        is_relevant  = (book.id in relevant_ids) if relevant_ids else True
        is_background = (book.current_context == "background")

        # Compute fill opacity based on relevance
        fill_alpha = 0.90 if is_relevant else 0.35
        fill_color = _hex_to_rgba(book.color, fill_alpha)

        # Border style
        if is_selected:
            border_color, border_width = "#ffffff", 2.5
        elif is_relevant and relevant_ids:
            border_color, border_width = "#ffffff", 1.8
        else:
            border_color, border_width = "rgba(255,255,255,0.4)", 1.0

        xs, ys = book.corners

        # Usage frequency heat strip (bottom of book)
        x0, y0, x1, y1 = book.rect
        heat_intensity = min(book.usage_frequency / 25.0, 1.0)
        heat_color = f"rgba(246,173,85,{heat_intensity * 0.7:.2f})"

        # Main book body
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=border_color, width=border_width),
            mode="lines",
            name=book.id,
            customdata=[[
                book.id, book.title, book.x, book.y,
                book.usage_frequency, book.current_context,
                book.last_used.strftime("%H:%M") if book.last_used else "—",
            ]],
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "ID: %{customdata[0]}<br>"
                "Position: (%{customdata[2]:.1f}, %{customdata[3]:.1f}) cm<br>"
                "Usage Frequency: <b>%{customdata[4]}×</b><br>"
                "Context: %{customdata[5]}<br>"
                "Last Used: %{customdata[6]}<extra></extra>"
            ),
            showlegend=False,
        ))

        # Usage heat strip (bottom band)
        fig.add_trace(_make_rect_trace(
            x0 + 0.5, y0 + 0.5, x1 - 0.5, y0 + 1.8,
            fill=heat_color, line_color="rgba(0,0,0,0)", line_width=0,
            name="_heat", hover="",
        ))

        # Book label — ID
        fig.add_annotation(
            x=book.x, y=book.y + 1.5,
            text=f"<b>{book.id}</b>",
            showarrow=False,
            font=dict(size=9, color="#ffffff", family=FONT_FAMILY),
        )

        # Short title (max 3 words → 2 lines)
        words = book.title.split()
        short = " ".join(words[:2]) + ("<br>" + " ".join(words[2:]) if len(words) > 2 else "")
        fig.add_annotation(
            x=book.x, y=book.y - 0.8,
            text=f"<span style='font-size:7px'>{short}</span>",
            showarrow=False,
            font=dict(size=7, color="rgba(255,255,255,0.85)"),
        )

        # Highlight ring for selected book
        if is_selected:
            hw, hh = book.width / 2 + 1.2, book.height / 2 + 1.2
            fig.add_shape(
                type="rect",
                x0=book.x - hw, y0=book.y - hh, x1=book.x + hw, y1=book.y + hh,
                line=dict(color="#f6e05e", width=2, dash="dot"),
                fillcolor="rgba(0,0,0,0)",
            )

    # ── 6. Robot arm base ─────────────────────────────────────────────────
    rx, ry = workspace.ROBOT_BASE
    # Reach circle (faint)
    theta_steps = 60
    import math
    reach_r = 55.0   # visual reach radius (cm)
    cxs = [rx + reach_r * math.cos(2 * math.pi * i / theta_steps) for i in range(theta_steps + 1)]
    cys = [ry + reach_r * math.sin(2 * math.pi * i / theta_steps) for i in range(theta_steps + 1)]
    fig.add_trace(go.Scatter(
        x=cxs, y=cys,
        mode="lines",
        line=dict(color="rgba(237,137,54,0.15)", width=1, dash="dot"),
        fill="toself",
        fillcolor="rgba(237,137,54,0.04)",
        hoverinfo="skip",
        showlegend=False,
    ))

    # Robot base marker
    fig.add_trace(go.Scatter(
        x=[rx], y=[ry],
        mode="markers+text",
        marker=dict(size=18, color="#ed8936",
                    symbol="diamond", line=dict(color="#fff3e0", width=2)),
        text=["🤖"],
        textfont=dict(size=14),
        textposition="middle center",
        name="Robot Base",
        hovertemplate=(
            "<b>🤖 Virtual Robot Arm</b><br>"
            f"Base: ({rx:.0f}, {ry:.0f}) cm<br>"
            f"Max Reach: {workspace.ROBOT_REACH:.0f} cm<br>"
            "<i>6-DOF — Phase 6 implementation</i><extra></extra>"
        ),
        showlegend=False,
    ))
    fig.add_annotation(
        x=rx, y=ry - 3,
        text="<b>ROBOT</b>",
        showarrow=False,
        font=dict(size=8, color="#ed8936"),
    )

    # ── 7. User position ──────────────────────────────────────────────────
    ux, uy = workspace.USER_POS
    fig.add_trace(go.Scatter(
        x=[ux], y=[uy],
        mode="markers+text",
        marker=dict(size=22, color="#4299e1",
                    symbol="circle", line=dict(color="#bee3f8", width=2)),
        text=["👤"],
        textfont=dict(size=16),
        textposition="middle center",
        name=workspace.current_user,
        hovertemplate=(
            f"<b>👤 {workspace.current_user}</b><br>"
            f"Position: ({ux:.0f}, {uy:.0f}) cm<br>"
            f"Context: {workspace.current_context}<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.add_annotation(
        x=ux, y=uy - 4,
        text=f"<b>USER</b>",
        showarrow=False,
        font=dict(size=8, color="#4299e1"),
    )

    # ── 8. Usage frequency legend bar ────────────────────────────────────
    fig.add_annotation(
        x=1, y=-12,
        xref="x", yref="y",
        text="<b>■</b> Usage Heat: <span style='color:#f6ad55'>█████</span>"
             " low → high",
        showarrow=False,
        font=dict(size=8, color="#718096"),
        xanchor="left",
    )

    # ── 9. Layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            range=[-8, W + 8],
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            ticksuffix=" cm",
            tickfont=dict(color="#4a5568", size=9),
            color="#4a5568",
        ),
        yaxis=dict(
            range=[-18, D + 12],
            showgrid=False,
            zeroline=False,
            showticklabels=True,
            ticksuffix=" cm",
            tickfont=dict(color="#4a5568", size=9),
            color="#4a5568",
            scaleanchor="x",
            scaleratio=1,
        ),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#1a202c",
            font_color="#e2e8f0",
            font_family=FONT_FAMILY,
            bordercolor="#4a5568",
        ),
        font=dict(family=FONT_FAMILY, color="#a0aec0"),
        dragmode="pan",
    )

    return fig
