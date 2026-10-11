from __future__ import annotations
import sys, time
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _get_pipeline():
    from robotics.pipeline import RobotPipeline
    return RobotPipeline

def _get_scene():
    from robotics.robot_model import SceneConfig
    return SceneConfig

def _get_sim():
    from simulation.robot_simulator import RobotSimulator
    return RobotSimulator


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');
.pipe-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);
  border-radius:12px;padding:.9rem 1.1rem;margin-bottom:.6rem;}
.pipe-card:hover{border-color:rgba(99,179,237,.3);}
.stage-badge{display:inline-block;padding:.2rem .7rem;border-radius:20px;
  font-size:.72rem;font-weight:700;font-family:'JetBrains Mono',monospace;}
.badge-ok{background:rgba(72,187,120,.18);border:1px solid #48bb78;color:#48bb78;}
.badge-fail{background:rgba(252,129,74,.18);border:1px solid #fc8181;color:#fc8181;}
.info-row{display:flex;justify-content:space-between;padding:.16rem 0;
  border-bottom:1px solid rgba(255,255,255,.04);font-size:.79rem;}
.info-key{color:#718096;} .info-val{color:#e2e8f0;font-family:'JetBrains Mono',monospace;}
.ph13-hdr{font-size:1.05rem;font-weight:700;color:#e2e8f0;margin:1.4rem 0 .7rem 0;
  padding-bottom:.45rem;border-bottom:1px solid rgba(99,179,237,.15);}
.p13m{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);
  border-radius:10px;padding:.65rem .8rem;text-align:center;}
.p13mv{font-size:1.25rem;font-weight:700;font-family:'JetBrains Mono',monospace;}
.p13ml{font-size:.66rem;color:#718096;margin-top:.15rem;text-transform:uppercase;letter-spacing:.04em;}
</style>
"""


def _rgba(c):
    r,g,b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
    a = c[3] if len(c) > 3 else 1.0
    return f"rgba({r},{g},{b},{a:.2f})"

def _box(cx, cy, cz, hw, hd, hh, col):
    x = [cx-hw,cx+hw,cx+hw,cx-hw,cx-hw,cx+hw,cx+hw,cx-hw]
    y = [cy-hd,cy-hd,cy+hd,cy+hd,cy-hd,cy-hd,cy+hd,cy+hd]
    z = [cz-hh,cz-hh,cz-hh,cz-hh,cz+hh,cz+hh,cz+hh,cz+hh]
    i=[0,0,1,1,2,2,4,4,5,5,6,6]; j=[1,2,2,3,3,0,5,6,6,7,7,4]; k=[2,3,3,0,0,1,6,7,7,4,4,5]
    return go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k,color=_rgba(col),
                     opacity=col[3] if len(col)>3 else 1.0,
                     flatshading=True,showscale=False,hoverinfo="skip")

def _cyl(cx, cy, cz, r, h, col, n=18):
    th = np.linspace(0,2*np.pi,n,endpoint=False)
    xs,ys,zs = [],[],[]
    for t in th:
        xs += [cx+r*np.cos(t), cx+r*np.cos(t), None]
        ys += [cy+r*np.sin(t), cy+r*np.sin(t), None]
        zs += [cz, cz+h, None]
    return go.Scatter3d(x=xs,y=ys,z=zs,mode="lines",
                        line=dict(color=_rgba(col),width=5),
                        hoverinfo="skip",showlegend=False)

def fig3d(snap, pr=None):
    tr = []
    t = snap["table"]
    tr.append(_box(t["position"][0],t["position"][1],t["position"][2],
                   t["width"]/2,t["depth"]/2,t["height"]/2,t["color"]))
    for obs in snap["obstacles"]:
        tr.append(_cyl(obs["position"][0],obs["position"][1],obs["position"][2],
                       obs["radius"],obs["height"],obs["color"]))
    aid = snap.get("active_book")
    for bk in snap["books"]:
        c,s,p = bk["color"],bk["size"],bk["position"]
        op = 0.55 if bk["is_held"] else (1.0 if bk["id"]==aid else 0.85)
        tr.append(_box(p[0],p[1],p[2],s[0]/2,s[1]/2,s[2]/2,(c[0],c[1],c[2],op)))
        tr.append(go.Scatter3d(x=[p[0]],y=[p[1]],z=[p[2]+s[2]/2+.025],
                               mode="text",text=[bk["title"][:14]],
                               textfont=dict(color="white",size=9),
                               hoverinfo="skip",showlegend=False))
    if pr:
        cart = pr.get("rrt",{}).get("path_cartesian",[])
        if len(cart) >= 2:
            tz = t["position"][2]+t["height"]/2+.015
            label = "RRT path (" + str(len(cart)) + " wpts)"
            tr.append(go.Scatter3d(x=[p[0] for p in cart],y=[p[1] for p in cart],
                                   z=[tz]*len(cart),mode="lines+markers",
                                   line=dict(color="rgba(159,122,234,.85)",width=4,dash="dot"),
                                   marker=dict(size=4,color="rgba(159,122,234,1)"),
                                   name=label,showlegend=True))
        tg = pr.get("target_pos",[])
        if len(tg) == 3:
            tr.append(go.Scatter3d(x=[tg[0]],y=[tg[1]],z=[tg[2]],mode="markers",
                                   marker=dict(size=14,color="#f6e05e",symbol="diamond",
                                               line=dict(color="#ecc94b",width=2)),
                                   name="IK Target",showlegend=True))
    jp = snap["joint_positions"]
    jx=[p[0] for p in jp]; jy=[p[1] for p in jp]; jz=[p[2] for p in jp]
    tr.append(go.Scatter3d(x=jx,y=jy,z=jz,mode="lines+markers",
                           line=dict(color="#63b3ed",width=9),
                           marker=dict(size=6,color="#90cdf4",line=dict(color="#4299e1",width=1)),
                           name="Arm",showlegend=True))
    ee = snap["ee_pos"]
    tr.append(go.Scatter3d(x=[ee[0]],y=[ee[1]],z=[ee[2]],mode="markers",
                           marker=dict(size=12,color="#f6ad55",symbol="diamond",
                                       line=dict(color="#ecc94b",width=2)),
                           name="EE",showlegend=True))
    hee = snap.get("ee_history",[])
    if len(hee) > 1:
        tr.append(go.Scatter3d(x=[p[0] for p in hee],y=[p[1] for p in hee],
                               z=[p[2] for p in hee],mode="lines",
                               line=dict(color="rgba(159,122,234,.4)",width=2,dash="dot"),
                               name="EE traj",showlegend=True,hoverinfo="skip"))
    fp = snap["finger_positions"]; gs = snap["gripper"]["state"]
    gc = "#48bb78" if gs=="closed" else ("#f6ad55" if gs=="moving" else "#a0aec0")
    for bk2,tk in [("left_base","left_tip"),("right_base","right_tip")]:
        bp,tp2 = fp[bk2],fp[tk]
        tr.append(go.Scatter3d(x=[bp[0],tp2[0]],y=[bp[1],tp2[1]],z=[bp[2],tp2[2]],
                               mode="lines",line=dict(color=gc,width=7),
                               hoverinfo="skip",showlegend=False))
    fig = go.Figure(data=tr)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(bgcolor="rgba(10,14,26,1)",
            xaxis=dict(title="X(m)",range=[-.75,.75],gridcolor="rgba(255,255,255,.05)",color="#718096"),
            yaxis=dict(title="Y(m)",range=[-.15,.95],gridcolor="rgba(255,255,255,.05)",color="#718096"),
            zaxis=dict(title="Z(m)",range=[-.05,.70],gridcolor="rgba(255,255,255,.05)",color="#718096"),
            camera=dict(eye=dict(x=1.6,y=-1.1,z=1.0)),
            aspectmode="manual",aspectratio=dict(x=1.4,y=1.0,z=0.8)),
        margin=dict(l=0,r=0,t=0,b=0),font=dict(color="#a0aec0",family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,.3)",bordercolor="rgba(255,255,255,.1)",
                    font=dict(color="#a0aec0",size=10)),height=520)
    return fig

def fig2d(pr, snap):
    tr = []; t = snap["table"]; hw,hd = t["width"]/2,t["depth"]/2; cy = t["position"][1]
    tr.append(go.Scatter(x=[-hw,hw,hw,-hw,-hw],y=[cy-hd,cy-hd,cy+hd,cy+hd,cy-hd],
                         mode="lines",fill="toself",fillcolor="rgba(139,100,64,.2)",
                         line=dict(color="rgba(139,100,64,.5)",width=1.5),showlegend=False))
    for obs in snap["obstacles"]:
        th = np.linspace(0,2*np.pi,30); ox,oy = obs["position"][0],obs["position"][1]
        tr.append(go.Scatter(x=(ox+obs["radius"]*np.cos(th)).tolist(),
                             y=(oy+obs["radius"]*np.sin(th)).tolist(),
                             mode="lines",fill="toself",fillcolor="rgba(220,60,60,.35)",
                             line=dict(color="rgba(220,60,60,.8)",width=2),showlegend=False))
    for bk in snap["books"]:
        p,s,c = bk["position"],bk["size"],bk["color"]
        fc = "rgba({},{},{},.55)".format(int(c[0]*255),int(c[1]*255),int(c[2]*255))
        tr.append(go.Scatter(x=[p[0]-s[0]/2,p[0]+s[0]/2,p[0]+s[0]/2,p[0]-s[0]/2,p[0]-s[0]/2],
                             y=[p[1]-s[1]/2,p[1]-s[1]/2,p[1]+s[1]/2,p[1]+s[1]/2,p[1]-s[1]/2],
                             mode="lines",fill="toself",fillcolor=fc,
                             line=dict(color=_rgba(c),width=2),
                             name=bk["title"][:12],showlegend=True))
    if pr:
        cart = pr.get("rrt",{}).get("path_cartesian",[])
        if len(cart) >= 2:
            px2=[p[0] for p in cart]; py2=[p[1] for p in cart]
            label = "RRT (" + str(len(cart)) + " wpts)"
            tr.append(go.Scatter(x=px2,y=py2,mode="lines+markers",
                                 line=dict(color="rgba(159,122,234,.9)",width=2.5),
                                 marker=dict(size=6,color="rgba(159,122,234,1)"),name=label))
            tr.append(go.Scatter(x=[px2[0]],y=[py2[0]],mode="markers",
                                 marker=dict(size=12,color="#48bb78",symbol="circle"),name="Start"))
            tr.append(go.Scatter(x=[px2[-1]],y=[py2[-1]],mode="markers",
                                 marker=dict(size=14,color="#f6e05e",symbol="star"),name="Goal"))
        tg = pr.get("target_pos",[])
        if len(tg) >= 2:
            tr.append(go.Scatter(x=[tg[0]],y=[tg[1]],mode="markers",
                                 marker=dict(size=16,color="#f6e05e",symbol="diamond"),name="IK Target"))
    ee = snap["ee_pos"]
    tr.append(go.Scatter(x=[ee[0]],y=[ee[1]],mode="markers",
                         marker=dict(size=14,color="#f6ad55",symbol="star"),name="EE"))
    fig = go.Figure(data=tr)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,14,26,1)",
        xaxis=dict(title="X(m)",range=[-.75,.75],gridcolor="rgba(255,255,255,.05)",color="#718096",
                   zeroline=True,zerolinecolor="rgba(255,255,255,.1)"),
        yaxis=dict(title="Y(m)",range=[-.15,.95],scaleanchor="x",
                   gridcolor="rgba(255,255,255,.05)",color="#718096"),
        margin=dict(l=40,r=10,t=20,b=40),font=dict(color="#a0aec0",family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,.3)",bordercolor="rgba(255,255,255,.1)",
                    font=dict(color="#a0aec0",size=9)),height=340)
    return fig

def fig_joints(angles):
    names=["J1","J2","J3","J4","J5","J6"]
    cols=["#63b3ed","#90cdf4","#48bb78","#f6ad55","#9f7aea","#fc8181"]
    fig = go.Figure(go.Bar(x=names,y=angles,marker=dict(color=cols),
                            text=["{:.1f}deg".format(a) for a in angles],
                            textposition="outside",textfont=dict(color="#a0aec0",size=10)))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,14,26,1)",
        yaxis=dict(title="Angle(deg)",range=[-200,200],gridcolor="rgba(255,255,255,.05)",
                   color="#718096",zeroline=True,zerolinecolor="rgba(255,255,255,.15)"),
        xaxis=dict(color="#a0aec0",tickfont=dict(size=9)),
        margin=dict(l=40,r=10,t=20,b=40),font=dict(color="#a0aec0",family="Inter"),height=240)
    return fig

def fig_ik_conv(errs):
    if not errs:
        f = go.Figure()
        f.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,14,26,1)",height=200,
            annotations=[dict(text="No IK data",showarrow=False,font=dict(color="#4a5568",size=13))])
        return f
    f = go.Figure()
    f.add_trace(go.Scatter(x=list(range(1,len(errs)+1)),y=[e*1000 for e in errs],
        mode="lines",line=dict(color="#63b3ed",width=2),fill="tozeroy",
        fillcolor="rgba(99,179,237,.08)",name="Error(mm)"))
    f.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,14,26,1)",
        xaxis=dict(title="Iteration",gridcolor="rgba(255,255,255,.05)",color="#718096"),
        yaxis=dict(title="Error(mm)",gridcolor="rgba(255,255,255,.05)",color="#718096"),
        margin=dict(l=45,r=10,t=20,b=40),font=dict(color="#a0aec0",family="Inter"),height=200)
    return f


def _ir(k, v):
    return '<div class="info-row"><span class="info-key">{}</span><span class="info-val">{}</span></div>'.format(k,v)

def _bdg(ok, ol="OK", fl="FAIL"):
    cls = "badge-ok" if ok else "badge-fail"
    lbl = ol if ok else fl
    return '<span class="stage-badge {}">{}</span>'.format(cls,lbl)

def _mm(col, val, lbl, c="#63b3ed"):
    col.markdown('<div class="p13m"><div class="p13mv" style="color:{}">{}</div>'
                 '<div class="p13ml">{}</div></div>'.format(c,val,lbl),
                 unsafe_allow_html=True)


def _init():
    if "p13_inited" in st.session_state:
        return
    scene = _get_scene().default_scene()
    sim   = _get_sim()(scene=scene, headless=True)
    pipe  = _get_pipeline()(arm=scene.arm, scene=scene)
    st.session_state.update({
        "p13_inited": True, "p13_scene": scene, "p13_sim": sim,
        "p13_pipe": pipe, "p13_book_sel": "B1", "p13_log": [],
        "p13_history": [], "p13_latest": None, "p13_ik_per_iter": [],
    })


def _run(bid):
    """Run FULL pipeline: IK -> RRT -> Robot for every movement waypoint."""
    scene = st.session_state["p13_scene"]
    sim   = st.session_state["p13_sim"]
    pipe  = st.session_state["p13_pipe"]
    log   = st.session_state["p13_log"]

    def L(m):
        log.append("[{}] {}".format(time.strftime("%H:%M:%S"), m))

    book = scene.get_book(bid)
    if book is None:
        L("ERROR: {} not found".format(bid))
        return

    scene.arm.set_home(); sim.reset(); sim.start(book_id=bid)
    st.session_state["p13_history"].clear()
    st.session_state["p13_ik_per_iter"] = []
    L("=== Phase-13: {} ({}) ===".format(book.title, bid))

    g  = book.grasp_position(); ab = g.copy(); ab[2] += .12
    pp = book.place_position()
    if pp is None:
        pp = g.copy(); pp[0] += .30
    ap = pp.copy(); ap[2] += .12
    ret = ap.copy(); ret[2] += .08

    # Every movement: mandatory IK -> RRT -> execute
    mvs = [
        ("Approach",    ab,  None,    None),
        ("Lower-grasp", g,   "close", book),
        ("Lift",        ab,  None,    book),
        ("Carry",       ap,  None,    book),
        ("Place",       pp,  "open",  None),
        ("Retreat",     ret, None,    None),
    ]

    for sn, tg, gcmd, abk in mvs:
        L("--- {} target=({:.3f},{:.3f},{:.3f}) ---".format(sn,tg[0],tg[1],tg[2]))
        pr = pipe.plan_and_execute(target_pos=tg, gripper_action=gcmd, active_book=abk)
        rd = pipe.latest_result_dict()
        if rd is not None:
            rd["step_name"] = sn; rd["target_pos"] = tg.tolist()
            pi = pr.ik_result.per_iter_error if pr.ik_result else []
            rd["ik_per_iter"] = pi
            st.session_state["p13_ik_per_iter"] = pi
            st.session_state["p13_history"].append(rd)
            st.session_state["p13_latest"] = rd
        if gcmd == "close":
            sim.gripper.close()
            if abk: sim.gripper.force_grasp(abk)
        elif gcmd == "open":
            sim.gripper.open()
        sim.arm.set_angles(scene.arm.angles)
        sim.ee_history.append(sim.arm.end_effector_position().copy())
        if rd:
            L("  IK:{} {:.2f}mm | RRT:{}wpts {:.0f}ms | Col:{} | {}".format(
                "conv" if rd["ik"]["converged"] else "div",
                rd["ik"]["error_mm"],
                rd["rrt"]["n_waypoints"], rd["rrt"]["planning_ms"],
                rd["rrt"]["collision_status"],
                "OK" if rd.get("success") else "FAIL"))
    L("=== done ===")


def render_pipeline_page():
    st.markdown(_CSS, unsafe_allow_html=True)
    _init()

    scene  = st.session_state["p13_scene"]
    sim    = st.session_state["p13_sim"]
    pipe   = st.session_state["p13_pipe"]
    latest = st.session_state.get("p13_latest")
    hist   = st.session_state.get("p13_history", [])

    # ---- header ----------------------------------------------------------
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a202c,#2d3748,#1a365d);
                border:1px solid rgba(159,122,234,.25);border-radius:16px;
                padding:1.5rem 2rem;margin-bottom:1.2rem;">
      <p style="font-size:1.65rem;font-weight:800;margin:0;
                background:linear-gradient(135deg,#9f7aea,#b794f4,#e9d8fd);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;">
        &#128279; Phase 13 &mdash; Integrated Planning Pipeline</p>
      <p style="color:#718096;margin-top:.4rem;font-size:.9rem;">
        Target Book &rarr; Jacobian IK &rarr; Joint-Space RRT &rarr; Virtual Robot &rarr; Verify</p>
      <div style="margin-top:.7rem;display:flex;gap:.5rem;flex-wrap:wrap;">
        <span style="background:rgba(99,179,237,.12);border:1px solid rgba(99,179,237,.3);
                     color:#63b3ed;padding:.2rem .65rem;border-radius:20px;font-size:.72rem;">
          Phase 11: Jacobian IK</span>
        <span style="background:rgba(72,187,120,.12);border:1px solid rgba(72,187,120,.3);
                     color:#48bb78;padding:.2rem .65rem;border-radius:20px;font-size:.72rem;">
          Phase 10: Joint-Space RRT</span>
        <span style="background:rgba(246,173,85,.12);border:1px solid rgba(246,173,85,.3);
                     color:#f6ad55;padding:.2rem .65rem;border-radius:20px;font-size:.72rem;">
          Phase 12: Virtual Robot</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ---- sidebar ---------------------------------------------------------
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Phase 13 Controls")
        book_opts = {b.book_id: b.title for b in scene.books}
        st.selectbox("Book target", options=list(book_opts.keys()),
                     format_func=lambda k: "{} - {}".format(k, book_opts[k]),
                     key="p13_book_sel")
        st.markdown("---")
        st.markdown(
            '<div style="font-size:.75rem;color:#718096;line-height:1.7;">'
            '<b>Pipeline (strict):</b><br>'
            '1. Target Pose<br>2. Jacobian IK<br>3. Joint-Space RRT<br>'
            '4. Execute Robot<br>5. Verify</div>', unsafe_allow_html=True)
        if hist:
            n_ok  = sum(1 for r in hist if r.get("success"))
            n_ik  = sum(1 for r in hist if r.get("ik",{}).get("converged"))
            n_rrt = sum(1 for r in hist if r.get("rrt",{}).get("success"))
            st.markdown("---")
            st.markdown(
                '<div style="font-size:.78rem;color:#a0aec0;">'
                '<b>Session Stats</b><br>'
                'Motions:{}<br>IK:{}/{}<br>RRT:{}/{}<br>OK:{}/{}</div>'.format(
                    len(hist),n_ik,len(hist),n_rrt,len(hist),n_ok,len(hist)),
                unsafe_allow_html=True)

    # ---- controls --------------------------------------------------------
    st.markdown('<div class="ph13-hdr">&#9881; Controls</div>', unsafe_allow_html=True)
    bc1,bc2,bc3 = st.columns(3)
    run_c   = bc1.button("&#128640; Run Full Pipeline", use_container_width=True,
                          key="p13_run", help="IK + RRT + Robot for EVERY waypoint")
    rst_c   = bc2.button("&#128260; Reset Scene", use_container_width=True, key="p13_rst")
    stp_c   = bc3.button("&#9197; Step (1 motion)", use_container_width=True,
                          key="p13_stp", help="Pipeline for next single movement")

    if rst_c:
        sim.reset(); scene.arm.set_home()
        st.session_state["p13_history"].clear()
        st.session_state["p13_latest"] = None
        st.session_state["p13_log"] = []
        st.session_state["p13_ik_per_iter"] = []
        st.rerun()

    if run_c:
        with st.spinner("IK -> RRT -> Robot..."):
            _run(st.session_state["p13_book_sel"])
        latest = st.session_state.get("p13_latest")
        hist   = st.session_state.get("p13_history", [])
        st.rerun()

    if stp_c and not run_c:
        si = len(st.session_state.get("p13_history", []))
        bk = scene.get_book(st.session_state["p13_book_sel"])
        if bk:
            g  = bk.grasp_position(); ab = g.copy(); ab[2] += .12
            pp = bk.place_position()
            if pp is None: pp = g.copy(); pp[0] += .30
            ap = pp.copy(); ap[2] += .12
            ret = ap.copy(); ret[2] += .08
            all_m = [
                ("Approach",    ab,  None,    None),
                ("Lower-grasp", g,   "close", bk),
                ("Lift",        ab,  None,    bk),
                ("Carry",       ap,  None,    bk),
                ("Place",       pp,  "open",  None),
                ("Retreat",     ret, None,    None),
            ]
            if si < len(all_m):
                sn,tg,gcmd,abk = all_m[si]
                with st.spinner("Step: {}".format(sn)):
                    pr = pipe.plan_and_execute(tg, gripper_action=gcmd, active_book=abk)
                    rd = pipe.latest_result_dict()
                    if rd:
                        rd["step_name"] = sn; rd["target_pos"] = tg.tolist()
                        pi = pr.ik_result.per_iter_error if pr.ik_result else []
                        rd["ik_per_iter"] = pi
                        st.session_state["p13_ik_per_iter"] = pi
                        st.session_state["p13_history"].append(rd)
                        st.session_state["p13_latest"] = rd
                    if gcmd == "close" and abk:
                        sim.gripper.close(); sim.gripper.force_grasp(abk)
                    elif gcmd == "open":
                        sim.gripper.open()
                    sim.arm.set_angles(scene.arm.angles)
                    sim.ee_history.append(sim.arm.end_effector_position().copy())
                latest = st.session_state.get("p13_latest")
                hist   = st.session_state.get("p13_history", [])
                st.rerun()

    snap = sim.snapshot()

    # ---- metrics ---------------------------------------------------------
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    if latest:
        ie=latest["ik"].get("error_mm",0); io=latest["ik"].get("converged",False)
        rw=latest["rrt"].get("n_waypoints",0); rm=latest["rrt"].get("planning_ms",0)
        co=latest.get("rrt",{}).get("collision_free",True); lo=latest.get("success",False)
    else:
        ie=rw=rm=0; io=co=lo=False
    _mm(m1, str(len(hist)), "Motions", "#63b3ed")
    _mm(m2, ("OK " if io else "X ") + "{:.1f}mm".format(ie), "Last IK err",
        "#48bb78" if io else "#fc8181")
    _mm(m3, str(rw), "RRT wpts", "#9f7aea")
    _mm(m4, "{:.0f}ms".format(rm), "RRT time", "#f6ad55")
    _mm(m5, "Clear" if co else "COLLISION!", "Collision", "#48bb78" if co else "#fc8181")
    _mm(m6, ("OK" if lo else ("-" if not hist else "FAIL")), "Last result",
        "#48bb78" if lo else ("#718096" if not hist else "#fc8181"))
    st.markdown("")

    # ---- 3-D scene -------------------------------------------------------
    st.markdown('<div class="ph13-hdr">&#127916; 3-D Scene (Arm + RRT Path)</div>',
                unsafe_allow_html=True)
    st.plotly_chart(fig3d(snap, latest), use_container_width=True, key="p13_3d")

    c2d,cj = st.columns(2)
    with c2d:
        st.markdown("##### &#128506; Top-Down View - RRT Path")
        st.plotly_chart(fig2d(latest, snap), use_container_width=True, key="p13_2d")
    with cj:
        st.markdown("##### &#128208; Joint Angles After IK+RRT")
        st.plotly_chart(fig_joints(snap["joint_angles_deg"]),
                        use_container_width=True, key="p13_j")

    # ---- pipeline breakdown ----------------------------------------------
    st.markdown('<div class="ph13-hdr">&#128202; Latest Motion - Pipeline Breakdown</div>',
                unsafe_allow_html=True)
    if latest is None:
        st.info("Click Run Full Pipeline or Step to start.")
    else:
        r = latest; tp = r["target_pos"]; ik = r["ik"]; rrt = r["rrt"]; ex = r["execution"]
        cok = rrt.get("collision_free", True); sn = r.get("step_name", "")
        bk_key = "p13_book_sel"
        st.markdown(
            '<div style="display:flex;align-items:center;gap:.4rem;flex-wrap:wrap;'
            'margin-bottom:.75rem;font-size:.82rem;font-family:monospace;">'
            '<span style="background:rgba(99,179,237,.15);border:1px solid rgba(99,179,237,.4);'
            'color:#63b3ed;padding:.25rem .6rem;border-radius:6px;">'
            'Book: {}</span>'
            '<span style="color:#4a5568;">&rarr;</span>'
            '<span style="background:rgba(99,179,237,.1);border:1px solid rgba(99,179,237,.3);'
            'color:#63b3ed;padding:.25rem .6rem;border-radius:6px;">'
            '({:.3f},{:.3f},{:.3f})m</span>'
            '<span style="color:#4a5568;">&rarr;</span>'
            '<span style="background:rgba(99,179,237,.1);border:1px solid rgba(99,179,237,.3);'
            'color:#63b3ed;padding:.25rem .6rem;border-radius:6px;">'
            'IK {}</span>'
            '<span style="color:#4a5568;">&rarr;</span>'
            '<span style="background:rgba(159,122,234,.1);border:1px solid rgba(159,122,234,.3);'
            'color:#9f7aea;padding:.25rem .6rem;border-radius:6px;">'
            'RRT {}wpts</span>'
            '<span style="color:#4a5568;">&rarr;</span>'
            '<span style="background:rgba(72,187,120,.1);border:1px solid rgba(72,187,120,.3);'
            'color:#48bb78;padding:.25rem .6rem;border-radius:6px;">'
            'Exec {}steps</span>'
            '<span style="color:#4a5568;">&rarr;</span>'
            '<span style="background:{};border:1px solid {};color:{};'
            'padding:.25rem .6rem;border-radius:6px;font-weight:700;">'
            '{}</span></div>'.format(
                st.session_state.get(bk_key,"?"),
                tp[0],tp[1],tp[2],
                "OK" if ik["converged"] else "FAIL",
                rrt["n_waypoints"], ex["steps"],
                "rgba(72,187,120,.15)" if r["success"] else "rgba(252,129,74,.15)",
                "#48bb78" if r["success"] else "#fc8181",
                "#48bb78" if r["success"] else "#fc8181",
                "SUCCESS" if r["success"] else "FAILED"),
            unsafe_allow_html=True)

        cc1,cc2,cc3 = st.columns(3)
        with cc1:
            fp2 = ik.get("final_pos",[])
            st.markdown(
                '<div class="pipe-card">'
                '<b style="color:#63b3ed;">&#128205; Target Pose</b>'
                '&nbsp;<span style="color:#718096;font-size:.76rem;">{}</span><br><br>'
                '{}{}{}{}{}{}</div>'.format(
                    sn,
                    _ir("X","{:+.4f}m".format(tp[0])),
                    _ir("Y","{:+.4f}m".format(tp[1])),
                    _ir("Z","{:+.4f}m".format(tp[2])),
                    _ir("EE X","{:+.4f}m".format(fp2[0]) if fp2 else "-"),
                    _ir("EE Y","{:+.4f}m".format(fp2[1]) if fp2 else "-"),
                    _ir("EE Z","{:+.4f}m".format(fp2[2]) if fp2 else "-")),
                unsafe_allow_html=True)
        with cc2:
            astr = " | ".join("{:.1f}deg".format(a) for a in ik.get("joint_angles_deg",[]))
            st.markdown(
                '<div class="pipe-card">'
                '<b style="color:#63b3ed;">&#129504; Jacobian IK</b>'
                '&nbsp;{}<br><br>'
                '{}{}{}</div>'.format(
                    _bdg(ik["converged"],"Converged","Diverged"),
                    _ir("Iterations",str(ik["iterations"])),
                    _ir("Error","{:.3f}mm".format(ik["error_mm"])),
                    _ir("Joints",astr or "-")),
                unsafe_allow_html=True)
        with cc3:
            cs = rrt.get("collision_status","clear")
            st.markdown(
                '<div class="pipe-card">'
                '<b style="color:#9f7aea;">&#127807; Joint-Space RRT</b>'
                '&nbsp;{}<br><br>'
                '{}{}{}{}{}</div>'.format(
                    _bdg(rrt["success"],"Found","Failed"),
                    _ir("Nodes",str(rrt["n_nodes"])),
                    _ir("Waypoints",str(rrt["n_waypoints"])),
                    _ir("Path(rad)","{:.4f}".format(rrt["path_length_rad"])),
                    _ir("Plan time","{:.1f}ms".format(rrt["planning_ms"])),
                    _ir("Collision",("OK " if cok else "X ")+cs)),
                unsafe_allow_html=True)

        fe = ex.get("final_ee",[])
        st.markdown(
            '<div class="pipe-card" style="border-color:rgba(72,187,120,.22);">'
            '<b style="color:#48bb78;">&#129470; Robot Execution + Verification</b>'
            '&nbsp;{}'
            '&nbsp;<span style="color:#718096;font-size:.78rem;">Path executed step-by-step</span>'
            '<br><br>'
            '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem;">'
            '<div>{}{}</div><div>{}{}</div><div>{}{}</div><div>{}{}</div>'
            '</div></div>'.format(
                _bdg(r["success"]),
                _ir("Steps",str(ex["steps"])),
                _ir("EE X","{:+.4f}m".format(fe[0]) if fe else "-"),
                _ir("EE Y","{:+.4f}m".format(fe[1]) if fe else "-"),
                _ir("EE Z","{:+.4f}m".format(fe[2]) if fe else "-"),
                _ir("Final err","{:.2f}mm".format(ex["final_err_mm"])),
                _ir("Tolerance","60mm"),
                _ir("IK ok","Yes" if ik["converged"] else "No"),
                _ir("RRT ok","Yes" if rrt["success"] else "No/fallback")),
            unsafe_allow_html=True)

        with st.expander("&#128200; IK Convergence (error vs iteration)", expanded=False):
            st.plotly_chart(fig_ik_conv(st.session_state.get("p13_ik_per_iter",[])),
                            use_container_width=True, key="p13_ikc")
        if r.get("stages"):
            with st.expander("&#9201; Per-Stage Timing", expanded=False):
                rows = [{"Stage":s["stage"],"OK":"Y" if s["success"] else "N",
                         "ms":round(s["duration_ms"],1),"Error":s.get("error","")}
                        for s in r["stages"]]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ---- motion history --------------------------------------------------
    if hist:
        st.markdown('<div class="ph13-hdr">&#128203; Motion History</div>',
                    unsafe_allow_html=True)
        rows = []
        for i,rr in enumerate(hist):
            ik_r  = rr.get("ik",{})
            rrt_r = rr.get("rrt",{})
            ex_r  = rr.get("execution",{})
            rows.append({
                "#": i+1, "Step": rr.get("step_name",""),
                "IK": ("Conv" if ik_r.get("converged") else "Div") +
                      " {:.1f}mm".format(ik_r.get("error_mm",0)),
                "RRT wpts": rrt_r.get("n_waypoints",0),
                "RRT ms":   round(rrt_r.get("planning_ms",0),1),
                "Collision":"OK" if rrt_r.get("collision_free",True) else "HIT",
                "Steps":    ex_r.get("steps",0),
                "Err mm":   round(ex_r.get("final_err_mm",0),2),
                "Result":   "OK" if rr.get("success") else "FAIL",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ---- log + joint info -----------------------------------------------
    with st.expander("&#128220; Phase-13 Pipeline Log", expanded=False):
        lm = st.session_state.get("p13_log",[])
        if lm:
            st.markdown("<br>".join(
                '<span style="font-family:monospace;font-size:.76rem;color:#a0aec0;">{}</span>'.format(m)
                for m in reversed(lm[-60:])), unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:#4a5568;">No log entries yet.</p>',
                        unsafe_allow_html=True)

    with st.expander("&#128307; Joint Info", expanded=False):
        ji = snap.get("joint_info",[])
        if ji:
            df = pd.DataFrame(ji)
            df.columns = ["#","Joint","Angle(deg)","Min(deg)","Max(deg)","MaxVel(rad/s)"]
            st.dataframe(df, use_container_width=True, hide_index=True)
