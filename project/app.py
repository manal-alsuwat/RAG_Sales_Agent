"""
Beamdata Secure AI Sales Agent - Streamlit UI

Same logic as the CLI (agent.live_chat):
  - Services mode  : pick a service -> grounded RAG answer, session ends.
  - Support mode   : up to 3 messages, each one passes the LLM-as-a-Judge
                     gate before RAG; a threat blocks, logs and terminates.
Run:  streamlit run project/app.py
"""

import streamlit as st

from agent import (
    load_policies,
    build_vectorstore,
    build_chain,
    full_pipeline,
    AGENT_MODEL,
)
from llm_judge import judge_ticket, MODEL_NAME as JUDGE_MODEL

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Beamdata Secure Sales Agent",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* ---------- layout ---------- */
      .block-container { padding-bottom: 4rem; }

      /* ---------- type scale (theme-aware colors) ---------- */
      .bd-title   { font-size: 1.75rem; font-weight: 700; letter-spacing:-.01em; margin:0; line-height:1.4;
                    white-space:nowrap; overflow:visible; }
      .bd-sub     { font-size: 1rem; opacity:.72; margin:.35rem 0 1.5rem 0; }
      .muted      { opacity:.72; }

      /* ---------- sidebar ---------- */
      .sb-brand   { font-size:1.15rem; font-weight:700; margin-bottom:.75rem; }
      .sb-label   { font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; opacity:.6; margin:.25rem 0 .35rem 0; }
      .model-box  { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.8rem; line-height:1.6;
                    padding:.55rem .7rem; border-radius:8px;
                    background: color-mix(in srgb, currentColor 7%, transparent); }
      .model-box b { opacity:.65; font-weight:600; display:inline-block; width:3.4rem; }
      .side-info h4 { margin:.9rem 0 .2rem 0; font-size:.92rem; font-weight:600; }
      .side-info p  { font-size:.84rem; line-height:1.5; margin:0; opacity:.72; }

      /* ---------- landing ---------- */
      .landing { text-align:center; padding:3.5rem 0 1.5rem 0; }
      .landing h2 { font-size:1.9rem; font-weight:700; margin:0 0 .4rem 0; }
      .landing p  { font-size:1.05rem; opacity:.72; margin:0; }
      .stButton > button[kind="primary"],
      .stButton > button[data-testid="baseButton-primary"] {
          height:3.4rem; font-size:1rem; font-weight:600; border-radius:10px; }
      .stButton > button { border-radius:8px; }

      /* ---------- chat ---------- */
      .badge { display:inline-block; padding:.22rem .65rem; border-radius:999px;
               font-size:.72rem; font-weight:700; letter-spacing:.04em; margin-bottom:.35rem; }
      .badge-ok     { background:#16a34a; color:#fff; }
      .badge-threat { background:#dc2626; color:#fff; }
      .badge-info   { background:#4f46e5; color:#fff; }
      .attack-chip  { display:inline-block; margin:.1rem .3rem .4rem 0; padding:.15rem .55rem;
                      border-radius:6px; background:#f59e0b; color:#1f2937; font-size:.74rem; font-weight:600; }
      .services-label { font-size:.95rem; font-weight:600; margin:1rem 0 .5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

SERVICES = [
    "AI Strategy",
    "AI Implementation",
    "AI Infrastructure",
    "Data and Cloud Infrastructure",
    "Data Analytics and Data Science",
]
MAX_TECH_MESSAGES = 3

CONTACT_BLOCK = """
**Thank you for using Beamdata Services.**

If you need anything else, feel free to contact us:
- Contact Form: https://beamdata.ai/contact/
- Email: info@beamdata.ai
- Phone: 365-795-0102
"""


# ---------------------------------------------------------------------------
# Resources (built once per process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base and models…")
def load_resources():
    policies = load_policies()
    vectorstore = build_vectorstore(policies)
    chain = build_chain()
    return vectorstore, chain


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def reset_session():
    st.session_state.mode = None            # None | "services" | "tech_support"
    st.session_state.messages = []          # [{"role","content","status","attack_type"}]
    st.session_state.history = []           # [{"user","assistant"}]  (last 3 used)
    st.session_state.message_count = 0
    st.session_state.terminated = False
    st.session_state.ended = False


if "mode" not in st.session_state:
    reset_session()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sb-brand">🛡️ Beamdata Secure Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Models</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="model-box"><b>Agent</b>{AGENT_MODEL}<br><b>Judge</b>{JUDGE_MODEL}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sb-label" style="margin-top:1.1rem">Modes</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="side-info">
          <h4>🔍 Explore Services</h4>
          <p>Pick one of Beamdata's five core services and get a grounded overview.</p>
          <h4>🔧 Technical Support</h4>
          <p>Describe your issue or suggestion. Up to {MAX_TECH_MESSAGES} messages per session;
             every message is screened before it reaches the assistant.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.mode is not None:
        st.divider()
        st.markdown('<div class="sb-label">Session</div>', unsafe_allow_html=True)
        label = "Explore Services" if st.session_state.mode == "services" else "Technical Support"
        st.markdown(f'<span class="badge badge-info">{label.upper()}</span>', unsafe_allow_html=True)
        if st.session_state.mode == "tech_support":
            remaining = MAX_TECH_MESSAGES - st.session_state.message_count
            st.caption(f"Remaining messages: **{remaining}** / {MAX_TECH_MESSAGES}")
        st.write("")
        if st.button("↩️ New session", use_container_width=True):
            reset_session()
            st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="bd-title">🤖 Beamdata Assistant</div>'
    '<div class="bd-sub">RAG-grounded answers · every request screened by an LLM-as-a-Judge security gate</div>',
    unsafe_allow_html=True,
)

vectorstore, chain = load_resources()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_assistant(msg):
    with st.chat_message("assistant", avatar="🤖"):
        status = msg.get("status")
        if status == "THREAT DETECTED":
            st.markdown('<span class="badge badge-threat">🚨 THREAT DETECTED</span>', unsafe_allow_html=True)
            attacks = msg.get("attack_type") or []
            if attacks:
                chips = "".join(f'<span class="attack-chip">{a}</span>' for a in attacks)
                st.markdown(chips, unsafe_allow_html=True)
        elif status == "ALLOWED":
            st.markdown('<span class="badge badge-ok">✅ ALLOWED</span>', unsafe_allow_html=True)
        st.markdown(msg["content"])


def render_history():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(msg["content"])
        else:
            render_assistant(msg)


# ---------------------------------------------------------------------------
# Landing (no mode chosen yet)
# ---------------------------------------------------------------------------
if st.session_state.mode is None:
    st.markdown(
        '<div class="landing"><h2>Welcome 👋</h2>'
        "<p>How can I help you today?</p></div>",
        unsafe_allow_html=True,
    )
    _, c1, c2, _ = st.columns([1, 2, 2, 1])
    with c1:
        if st.button("🔍 Explore Services", use_container_width=True, type="primary"):
            st.session_state.mode = "services"
            st.rerun()
    with c2:
        if st.button("🔧 Get Technical Support", use_container_width=True, type="primary"):
            st.session_state.mode = "tech_support"
            st.rerun()
    st.stop()


# ---------------------------------------------------------------------------
# Services flow
# ---------------------------------------------------------------------------
if st.session_state.mode == "services":
    render_history()

    if not st.session_state.ended:
        st.markdown('<div class="services-label">Select a service</div>', unsafe_allow_html=True)
        rows = [st.columns(3), st.columns(3)]
        slots = [c for row in rows for c in row][:len(SERVICES)]
        for col, name in zip(slots, SERVICES):
            if col.button(name, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": name})
                with st.spinner("Fetching information…"):
                    result = full_pipeline(
                        vectorstore,
                        chain,
                        row={"ticket_text": name, "action": "allow", "risk_level": "low"},
                        history_text="MODE: services",
                    )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["reply"],
                        "status": result["status"],
                        "attack_type": result.get("attack_type", []),
                    }
                )
                st.session_state.ended = True
                st.rerun()
    else:
        st.success(CONTACT_BLOCK)
        st.markdown('<p class="muted">Services session complete — start a new session from the sidebar.</p>', unsafe_allow_html=True)
    st.stop()


# ---------------------------------------------------------------------------
# Technical support flow
# ---------------------------------------------------------------------------
render_history()

if st.session_state.terminated:
    st.error("🚨 Session terminated by policy. The incident has been logged and forwarded to the Security Operations Center (SOC).")
    st.stop()

if st.session_state.message_count >= MAX_TECH_MESSAGES:
    st.warning(f"🔒 Message limit reached ({MAX_TECH_MESSAGES}). Session ended. Have a nice day! 👋")
    st.stop()

if not st.session_state.messages:
    st.info(
        "🔧 **Technical Support Mode.** Please describe your issue or suggestion. "
        f"You can send up to {MAX_TECH_MESSAGES} messages in this session."
    )

user_input = st.chat_input("Describe your issue or suggestion…")

if user_input and user_input.strip():
    user_input = user_input.strip()
    st.session_state.message_count += 1
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    history_text = "\n".join(
        f"User: {h['user']}\nAssistant: {h['assistant']}"
        for h in st.session_state.history[-3:]
    )

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🛡️ Screening request…"):
            judge_result = judge_ticket(user_input)

        mock_row = {
            "ticket_text": user_input,
            "action": judge_result["action"],
            "risk_level": judge_result["risk_level"],
            "reason": judge_result["reason"],
        }

        with st.spinner("Thinking…"):
            result = full_pipeline(
                vectorstore,
                chain,
                row=mock_row,
                history_text=history_text,
                mode="tech_support",
            )

    st.session_state.history.append({"user": user_input, "assistant": result["reply"]})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["reply"],
            "status": result["status"],
            "attack_type": result.get("attack_type", []),
        }
    )

    if result.get("terminate_session"):
        st.session_state.terminated = True

    st.rerun()
