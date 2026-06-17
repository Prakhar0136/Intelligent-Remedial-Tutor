import streamlit as st
import tempfile
import os
from agent import QuizzerAgent

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Intelligent Remedial Tutor",
    page_icon="🎓",
    layout="wide"
)

# ==================================================
# DEBUG
# ==================================================

print("APP STARTED")

# ==================================================
# SESSION STATE
# ==================================================

defaults = {
    "setup_complete": False,
    "tutor": None,
    "current_question": None,
    "answered": False,
    "eval_result": None,
    "remediation": "",
    "total_attempts": 0,
    "score": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==================================================
# CSS
# ==================================================

st.markdown("""
<style>

/* Main app */
.stApp {
    background-color: #f8f9fb;
}

/* Main content text */
h1, h2, h3 {
    color: #111827 !important;
}

p, label, span {
    color: #374151;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1f2937;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* File uploader */
section[data-testid="stSidebar"] .stFileUploader label {
    color: white !important;
}

/* Text input labels */
section[data-testid="stSidebar"] .stTextInput label {
    color: white !important;
}

/* Button */
.stButton > button {
    background-color: #2563eb;
    color: white !important;
    border-radius: 8px;
    border: none;
}

.stButton > button:hover {
    background-color: #1d4ed8;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
}

/* Radio buttons */
.stRadio label {
    color: #111827 !important;
    font-size: 16px;
}

/* Question card */
div[data-testid="stVerticalBlock"] {
    color: #111827;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("⚙️ Setup")

    api_key = st.text_input(
        "Groq API Key",
        type="password"
    )

    uploaded_file = st.file_uploader(
        "Upload Textbook (PDF)",
        type=["pdf"]
    )

    if st.button(
        "Initialize Tutor 🚀",
        use_container_width=True
    ):

        if not api_key:
            st.warning("Enter Groq API key.")
            st.stop()

        if not uploaded_file:
            st.warning("Upload a PDF.")
            st.stop()

        try:

            with st.status(
                "Creating Tutor...",
                expanded=True
            ) as status:

                st.write("Saving PDF...")

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:

                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                st.write("Loading Tutor...")

                tutor = QuizzerAgent(
                    groq_api_key=api_key
                )

                st.write("Processing PDF...")

                chunk_count = tutor.ingest_pdf(
                    tmp_path
                )

                os.unlink(tmp_path)

                st.session_state.tutor = tutor
                st.session_state.setup_complete = True

                status.update(
                    label=f"Ready! {chunk_count} chunks loaded.",
                    state="complete"
                )

        except Exception as e:

            st.error(
                f"Initialization failed:\n\n{str(e)}"
            )

# ==================================================
# LANDING PAGE
# ==================================================

if not st.session_state.setup_complete:

    st.markdown("""
    <h1 style="color:#111827;">
        🎓 Intelligent Remedial Tutor
    </h1>

    <p style="color:#374151;font-size:18px;">
        Upload a PDF to start your AI-powered learning journey.
    </p>
    """, unsafe_allow_html=True)

    st.stop()

# ==================================================
# SAFETY CHECK
# ==================================================

if st.session_state.tutor is None:
    st.error("Tutor not initialized.")
    st.stop()

# ==================================================
# LAYOUT
# ==================================================

col_main, col_stats = st.columns([3, 1])

# ==================================================
# STATS
# ==================================================

with col_stats:

    st.subheader("📊 Performance")

    st.metric(
        "Attempts",
        st.session_state.total_attempts
    )

    st.metric(
        "Score",
        st.session_state.score
    )

    if st.button("Reset Session"):

        st.session_state.current_question = None
        st.session_state.answered = False
        st.session_state.eval_result = None
        st.session_state.remediation = ""
        st.session_state.total_attempts = 0
        st.session_state.score = 0

        st.rerun()

# ==================================================
# GENERATE QUESTION
# ==================================================

if st.session_state.current_question is None:

    try:

        with st.spinner(
            "Generating question..."
        ):

            st.session_state.current_question = (
                st.session_state.tutor
                .generate_question()
            )

            st.session_state.answered = False

    except Exception as e:

        st.error(
            f"Question generation failed:\n\n{str(e)}"
        )

        st.stop()

q = st.session_state.current_question

# ==================================================
# QUESTION DISPLAY
# ==================================================

with col_main:

    with st.container(border=True):

        st.caption(
            f"Topic: {q.concept}"
        )

        st.subheader(
            q.question
        )

        user_choice = st.radio(
            "Choose your answer:",
            q.options,
            index=None,
            disabled=st.session_state.answered
        )

    if (
        not st.session_state.answered
        and st.button(
            "Submit Answer",
            type="primary",
            disabled=user_choice is None
        )
    ):

        try:

            selected_index = (
                q.options.index(
                    user_choice
                )
            )

            mapping = {
                0: "A",
                1: "B",
                2: "C",
                3: "D"
            }

            student_letter = mapping[
                selected_index
            ]

            with st.spinner(
                "Evaluating..."
            ):

                result = (
                    st.session_state.tutor
                    .evaluate_answer(
                        q,
                        student_letter
                    )
                )

                st.session_state.eval_result = result
                st.session_state.answered = True
                st.session_state.total_attempts += 1

                if result.is_correct:

                    st.session_state.score += 1

                else:

                    st.session_state.remediation = (
                        st.session_state.tutor
                        .generate_remediation(
                            result
                        )
                    )

            st.rerun()

        except Exception as e:

            st.error(
                f"Evaluation failed:\n\n{str(e)}"
            )

# ==================================================
# RESULTS
# ==================================================

if (
    st.session_state.answered
    and st.session_state.eval_result
):

    result = st.session_state.eval_result

    if result.is_correct:

        st.success(
            "🎉 Correct Answer!"
        )

        st.balloons()

    else:

        st.error(
            f"❌ Incorrect.\n\nCorrect Answer: {q.correct_option}"
        )

        st.markdown(
            "### 🔍 Diagnostic"
        )

        st.info(
            result.diagnostic
        )

        st.markdown(
            "### 📖 Remediation"
        )

        st.markdown(
            st.session_state.remediation
        )

    st.divider()

    if st.button(
        "Next Question ➡️"
    ):

        st.session_state.current_question = None
        st.session_state.answered = False
        st.session_state.eval_result = None

        st.rerun()