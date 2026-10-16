"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path, replay_events
from exercises.exercise_4_audit import build_graph

load_dotenv()


# ─── Database helpers for Streamlit ─────────────────────────────────────────
async def _fetch_recent_sessions() -> list[dict[str, Any]]:
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp)  AS started,
                   MAX(timestamp)  AS last_event,
                   MAX(risk_level) AS worst_risk,
                   COUNT(*)        AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 20
            """
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def _fetch_calibration_stats() -> dict[str, Any]:
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT 
                COUNT(*) AS total_reviews,
                AVG(confidence) AS avg_confidence,
                SUM(CASE WHEN decision = 'approve' THEN 1 ELSE 0 END) AS approved_count,
                SUM(CASE WHEN decision = 'auto' THEN 1 ELSE 0 END) AS auto_count,
                SUM(CASE WHEN decision = 'escalate' THEN 1 ELSE 0 END) AS escalate_count,
                SUM(CASE WHEN decision = 'reject' THEN 1 ELSE 0 END) AS reject_count
            FROM audit_events
            WHERE action IN ('commit', 'auto_approve')
            """
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else {}


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None
if "selected_replay_thread" not in st.session_state:
    st.session_state.selected_replay_thread = None


# ─── Page setup & Styling ──────────────────────────────────────────────────
st.set_page_config(
    page_title="HITL PR Review Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-med {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 3px 8px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 8px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .card-box {
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        background: #ffffff;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🛡️ HITL PR Review Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">LangGraph Human-in-the-Loop PR triage & autonomous code review with durable audit trails</div>',
    unsafe_allow_html=True,
)


# ─── Sidebar — recent sessions & calibration ───────────────────────────────
with st.sidebar:
    st.header("📋 Recent Sessions")
    try:
        recent_sessions = asyncio.run(_fetch_recent_sessions())
    except Exception as e:
        recent_sessions = []
        st.warning(f"Unable to read audit DB: {e}")

    if recent_sessions:
        for sess in recent_sessions[:10]:
            t_id = sess["thread_id"]
            short_id = f"{t_id[:8]}..."
            risk = sess.get("worst_risk", "med")
            badge_class = f"badge-{risk}"
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(
                    f"**{short_id}** <span class='{badge_class}'>{risk.upper()}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"{sess['pr_url'].split('/')[-1]} | {sess['events']} evts")
            with cols[1]:
                if st.button("Inspect", key=f"insp_{t_id}"):
                    st.session_state.selected_replay_thread = t_id
                    st.rerun()
            st.divider()
    else:
        st.caption("No audit sessions recorded yet.")

    # Calibration section (Bonus Challenge 2)
    st.header("📊 Confidence Calibration")
    try:
        calib = asyncio.run(_fetch_calibration_stats())
        if calib.get("total_reviews", 0) > 0:
            avg_c = calib.get("avg_confidence", 0) or 0
            st.metric("Avg Review Confidence", f"{avg_c:.1%}")
            col_a, col_b = st.columns(2)
            col_a.metric("Approved", calib.get("approved_count", 0))
            col_b.metric("Escalated", calib.get("escalate_count", 0))
            col_c, col_d = st.columns(2)
            col_c.metric("Auto Commits", calib.get("auto_count", 0))
            col_d.metric("Rejected", calib.get("reject_count", 0))
        else:
            st.caption("No completed review decisions yet.")
    except Exception:
        st.caption("Calibration data unavailable.")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start", clear_on_submit=False):
    st.markdown("##### 🚀 Start or Resume PR Review")
    col_input, col_sample = st.columns([3, 1])
    with col_input:
        pr_url = st.text_input(
            "Target Pull Request URL",
            value=st.session_state.pr_url or "https://github.com/VinUni-AI20k/PR-Demo/pull/1",
            placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
            help="Enter a GitHub pull request URL to review",
        )
    with col_sample:
        st.caption("Quick pick demo PRs:")
        sample = st.selectbox(
            "Demo PRs",
            [
                "Custom URL",
                "PR #1 — Task priority (human_approval)",
                "PR #2 — User login + sync (escalate)",
            ],
            index=0,
            label_visibility="collapsed",
        )
        if sample == "PR #1 — Task priority (human_approval)":
            pr_url = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"
        elif sample == "PR #2 — User login + sync (escalate)":
            pr_url = "https://github.com/VinUni-AI20k/PR-Demo/pull/2"

    submitted = st.form_submit_button("Run review", type="primary", use_container_width=True)


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload.get("confidence", 0.0)
    st.info(f"⚖️ **Approval requested** — Confidence score: **{conf:.0%}** (Human In The Loop required)")
    
    with st.container():
        st.markdown(f"**LLM Reasoning:** {payload.get('confidence_reasoning', 'N/A')}")
        st.markdown("### Summary of Proposed Changes")
        st.markdown(payload.get("summary", ""))

        st.markdown("### Proposed Review Comments")
        comments = payload.get("comments", [])
        if comments:
            for c in comments:
                sev = c.get("severity", "suggestion")
                sev_color = {
                    "blocker": "🔴",
                    "issue": "🟠",
                    "suggestion": "🔵",
                    "nit": "⚪",
                }.get(sev, "🔵")
                st.markdown(f"- {sev_color} **[{sev.upper()}]** `{c.get('file')}:{c.get('line') or '?'}` — {c.get('body')}")
        else:
            st.caption("No specific line comments generated.")

        with st.expander("📄 View Unified Diff Preview"):
            st.code(payload.get("diff_preview", ""), language="diff")

        feedback = st.text_input("Reviewer Feedback / Revision Notes (optional)", key="approval_feedback")

        st.write("---")
        col1, col2, col3 = st.columns(3)
        if col1.button("✅ Approve & Post Comment", type="primary", use_container_width=True):
            return {"choice": "approve", "feedback": feedback}
        if col2.button("❌ Reject Review", use_container_width=True):
            return {"choice": "reject", "feedback": feedback}
        if col3.button("✏️ Edit & Request Revision", use_container_width=True):
            return {"choice": "edit", "feedback": feedback}

    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload.get("confidence", 0.0)
    st.error(f"🚨 **Strong Escalation** — Confidence score: **{conf:.0%}** (< 58% Escalate Threshold)")
    
    with st.container():
        st.markdown(f"**LLM Reasoning:** {payload.get('confidence_reasoning', 'N/A')}")
        
        risks = payload.get("risk_factors", [])
        if risks:
            st.warning("**Identified Security & Architectural Risks:**\n" + "\n".join(f"- ⚠️ {r}" for r in risks))
            
        st.markdown("### PR Summary")
        st.markdown(payload.get("summary", ""))

        st.write("---")
        st.markdown("### ❓ Clarifying Questions for Reviewer")
        st.write("Please answer the LLM's questions so it can re-synthesize an informed review:")

        with st.form("escalation_form"):
            questions = payload.get("questions", [])
            answers: dict[str, str] = {}
            for i, q in enumerate(questions):
                st.markdown(f"**Q{i+1}:** {q}")
                answers[q] = st.text_input(
                    f"Your answer to Q{i+1}",
                    key=f"esc_q_{i}",
                    label_visibility="collapsed",
                    placeholder="Provide context, intent, or resolution...",
                )
            
            submitted_answers = st.form_submit_button("Submit answers & Re-synthesize Review", type="primary", use_container_width=True)
            if submitted_answers:
                return answers

    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.selected_replay_thread = None

    with st.spinner("Fetching PR from GitHub + analyzing code diff with LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload.get("kind")
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming agent flow with your input..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    a = final.get("analysis")
    final_conf = a.confidence if a else 0.0

    st.write("---")
    if action.startswith("auto"):
        st.balloons()
        st.success(
            f"🎉 **Auto-Approved!** (Confidence: **{final_conf:.0%}** ≥ 73%)\n\n"
            f"The agent automatically posted the review comment directly to the PR without requiring human intervention."
        )
        if a and a.summary:
            st.info(f"**LLM Summary:** {a.summary}")
        st.markdown(f"👉 [**View Automated Comment on GitHub**]({st.session_state.pr_url})")

    elif action.startswith("committed"):
        st.success(
            f"✅ **Review Successfully Committed!**\n\n"
            f"Comment posted to PR by reviewer `{os.environ.get('GITHUB_USER', 'you')}`."
        )
        if a and a.summary:
            st.info(f"**Final Summary:** {a.summary}")
        st.markdown(f"👉 [**View Review Comment on GitHub**]({st.session_state.pr_url})")

    elif action == "rejected":
        st.warning(f"🚫 **Review Rejected** — no comment was posted to the PR.")
    else:
        st.info(f"**Final Action:** `{action}`")

    st.caption(
        f"Session Thread ID: `{st.session_state.thread_id}`  ·  "
        f"Replay in terminal: `uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )

    # Show session timeline
    with st.expander("📜 View Audit Trail Events for this session", expanded=True):
        try:
            events = asyncio.run(replay_events(st.session_state.thread_id))
            if events:
                for ev in events:
                    r_color = {"low": "🟢", "med": "🟡", "high": "🔴"}.get(ev["risk_level"], "⚪")
                    st.markdown(
                        f"{r_color} **{ev['action']}** | Conf: `{ev['confidence']:.0%}` | Risk: `{ev['risk_level']}` | "
                        f"Decision: `{ev['decision']}` | Reviewer: `{ev['reviewer_id'] or '-'}` ({ev['execution_time_ms']}ms)"
                    )
                    if ev.get("reason"):
                        st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {ev['reason']}")
        except Exception as e:
            st.caption(f"Unable to load events: {e}")


# ─── Inspect Past Session (Sidebar Trigger) ────────────────────────────────
if st.session_state.selected_replay_thread:
    insp_thread = st.session_state.selected_replay_thread
    st.write("---")
    st.subheader(f"🔍 Inspecting Past Audit Session: `{insp_thread}`")
    try:
        events = asyncio.run(replay_events(insp_thread))
        if events:
            for ev in events:
                r_color = {"low": "🟢", "med": "🟡", "high": "🔴"}.get(ev["risk_level"], "⚪")
                st.markdown(
                    f"{r_color} **{ev['action']}** | Conf: `{ev['confidence']:.0%}` | Risk: `{ev['risk_level']}` | "
                    f"Decision: `{ev['decision']}` | Reviewer: `{ev['reviewer_id'] or '-'}` ({ev['execution_time_ms']}ms)"
                )
                if ev.get("reason"):
                    st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ {ev['reason']}")
        else:
            st.warning("No audit events found for this session.")
    except Exception as e:
        st.error(f"Error fetching session events: {e}")
