import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
import html

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sarcastibot",
    page_icon="🤖",
    layout="centered",
)

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0D0D0F;
    --surface:   #16161A;
    --surface2:  #1E1E24;
    --border:    #2A2A35;
    --accent:    #7B61FF;
    --accent2:   #FF6161;
    --text:      #E8E8F0;
    --muted:     #7A7A8C;
    --user-bg:   #1B1B28;
    --bot-bg:    #1A1A1E;
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Main container ── */
.block-container {
    max-width: 760px !important;
    padding: 0 1.5rem 6rem 1.5rem !important;
    margin: 0 auto !important;
}

/* ── Header ── */
.chat-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.chat-header .avatar {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.25rem;
}
.chat-header h1 {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.2 !important;
}
.chat-header .subtitle {
    font-size: 0.72rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.04em;
}
.status-dot {
    width: 8px; height: 8px;
    background: #4AE68A;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Message bubbles ── */
.msg-row {
    display: flex;
    gap: 0.6rem;
    margin-bottom: 1.1rem;
    align-items: flex-start;
}
.msg-row.user { flex-direction: row-reverse; }

.msg-icon {
    width: 32px; height: 32px; flex-shrink: 0;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
    margin-top: 2px;
}
.msg-icon.bot { background: linear-gradient(135deg, var(--accent), var(--accent2)); }
.msg-icon.user { background: var(--surface2); border: 1px solid var(--border); }

.msg-bubble {
    max-width: 78%;
    padding: 0.7rem 1rem;
    border-radius: 14px;
    font-size: 0.92rem;
    line-height: 1.6;
    word-break: break-word;
}
.msg-bubble.bot {
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-top-left-radius: 4px;
    color: var(--text);
}
.msg-bubble.user {
    background: linear-gradient(135deg, #3a2f7a, #2b2b50);
    border: 1px solid #4a3f8f;
    border-top-right-radius: 4px;
    color: var(--text);
}
.msg-meta {
    font-size: 0.68rem;
    color: var(--muted);
    margin-top: 3px;
    font-family: 'JetBrains Mono', monospace;
}
.msg-row.user .msg-meta { text-align: right; }

/* ── Typing indicator ── */
.typing-dots span {
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    margin: 0 2px;
    animation: bounce 1.2s ease-in-out infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 100% { transform: translateY(0);  opacity: 0.5; }
    50%       { transform: translateY(-5px); opacity: 1; }
}

/* ── Chat input bar ── */
[data-testid="stChatInputContainer"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: min(760px, 100vw) !important;
    padding: 0.75rem 1.5rem 1rem !important;
    background: var(--bg) !important;
    border-top: 1px solid var(--border) !important;
    z-index: 999 !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(123,97,255,0.15) !important;
}
[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    border-radius: 10px !important;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--muted);
}
.empty-state .big-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.empty-state h3 { color: var(--text); font-size: 1.1rem; margin-bottom: 0.4rem; }
.empty-state p  { font-size: 0.85rem; line-height: 1.6; }

/* ── Clear button ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    padding: 0.3rem 0.75rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    border-color: var(--accent2) !important;
    color: var(--accent2) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model init (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return ChatMistralAI(
        model="ministral-3b-latest",
        temperature=0.7,
        max_tokens=2048,
    )


# ── Session state ─────────────────────────────────────────────────────────────
if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = [
        SystemMessage(content="You are a sarcastic assistant.")
    ]
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []   # [{role, content, time}]


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <div class="avatar">🤖</div>
    <div>
        <h1>Sarcastibot</h1>
        <div class="subtitle"><span class="status-dot"></span>ministral-3b-latest · sarcasm level: maximum</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Clear button
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("clear", key="clear_btn"):
        st.session_state.lc_messages = [
            SystemMessage(content="You are a sarcastic assistant.")
        ]
        st.session_state.display_messages = []
        st.rerun()


# ── Render messages ───────────────────────────────────────────────────────────
if not st.session_state.display_messages:
    st.markdown("""
    <div class="empty-state">
        <div class="big-icon">💬</div>
        <h3>Oh, you finally showed up.</h3>
        <p>Go ahead — ask me something. I'll try to contain my enthusiasm.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.display_messages:
        escaped_content = html.escape(msg["content"])
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="msg-icon user">👤</div>
                <div>
                    <div class="msg-bubble user">{escaped_content}</div>
                    <div class="msg-meta">{msg["time"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-row">
                <div class="msg-icon bot">🤖</div>
                <div>
                    <div class="msg-bubble bot">{escaped_content}</div>
                    <div class="msg-meta">{msg["time"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Input handling ────────────────────────────────────────────────────────────
user_input = st.chat_input("Say something… if you must.")

if user_input:
    prompt = user_input.strip()
    if not prompt:
        st.stop()
    
    # Timestamp
    now = datetime.now().strftime("%H:%M")

    # Add user message to display
    st.session_state.display_messages.append({
        "role": "user",
        "content": prompt,
        "time": now,
    })

    # Add to LangChain history
    st.session_state.lc_messages.append(HumanMessage(content=prompt))

    # ── Typing indicator + response ──
    with st.spinner(""):
        st.markdown("""
        <div class="msg-row">
            <div class="msg-icon bot">🤖</div>
            <div class="msg-bubble bot">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            model = load_model()
            response = model.invoke(st.session_state.lc_messages)
            reply = response.content
            st.session_state.lc_messages.append(AIMessage(content=reply))
            st.session_state.display_messages.append({
                "role": "bot",
                "content": reply,
                "time": datetime.now().strftime("%H:%M"),
            })
        except Exception as e:
            st.error(f"Something broke. Even I'm surprised. Error: {e}")

    st.rerun()