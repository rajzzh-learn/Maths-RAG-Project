"""
Streamlit chat interface for the Math Tutor RAG Agent.
Run with:  streamlit run math_tutor/app.py
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path for Streamlit Cloud
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from math_tutor.config import get_config
from math_tutor.ingest import build_vector_store
from math_tutor.rag_chain import build_rag_chain, convert_history
from math_tutor.file_utils import (
    extract_text_from_pdf,
    extract_text_from_txt,
    image_to_base64_uri,
    is_image,
    SUPPORTED_EXTS,
)
from math_tutor.components.paste_button import paste_image_button

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Math Tutor — Class 12 CBSE",
    page_icon="📐",
    layout="wide",
)

st.title("📐 Class 12 Math Tutor")
st.caption("Powered by your NCERT notes, exemplar, and PYQs — 2027 Board Exam Edition")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    current_provider = get_config("LLM_PROVIDER", "openai").lower()
    st.caption(f"🤖 **Active Provider**: `{current_provider.upper()}`")
    REPO = "https://github.com/rajzzh-learn/Maths-RAG-Project/blob/main"

    st.header("🗂️ Chapters")
    st.markdown("""
**Part 1**

1. Relations and Functions
2. Inverse Trigonometric Functions
3. Matrices
4. Determinants
5. Continuity and Differentiability
6. Application of Derivatives

---

**Part 2**

7. Integrals
8. Application of Integrals
9. Differential Equations
10. Vector Algebra
11. Three-Dimensional Geometry
12. Linear Programming
13. Probability
""")

    st.divider()

    # ── Study Material Links ───────────────────────────────────────────────
    st.header("📂 Study Material")

    with st.expander("📚 NCERT Solutions"):
        st.markdown(f"""
- [Ch 1 — Relations and Functions]({REPO}/Ncert%20Solutions/Class%2012%20Maths%20Chapter%201%20Relations%20And%20Functions.pdf)
- [Ch 2 — Inverse Trigonometric Functions]({REPO}/Ncert%20Solutions/Class%2012%20Maths%20Chapter%202%20Inverse%20Trigonometric%20Functions.pdf)
- [Ch 3 — Matrices]({REPO}/Ncert%20Solutions/Class%2012%20Maths%20Chapter%203%20Matrices.pdf)
- [Ch 4 — Determinants]({REPO}/Ncert%20Solutions/Class%2012%20Maths%20Chapter%204%20Determinants.pdf)
- [Ch 5 — Continuity and Differentiability]({REPO}/Ncert%20Solutions/Class%2012%20Maths%20Chapter%205%20Continuity%20And%20Differentiability.pdf)
- [Ch 6 — Application of Derivatives]({REPO}/Ncert%20Solutions/Class%2012%20Maths%20Chapter%206%20Application%20Of%20Derivatives.pdf)
- [Ch 8 — Application of Integrals]({REPO}/Ncert%20Solutions/Class%2012%20Maths%20Chapter%208%20Application%20Of%20Integrals.pdf)
- [Ch 9 — Differential Equations]({REPO}/Ncert%20Solutions/Class%2012%20Maths%20Chapter%209%20Differential%20Equations.pdf)
- [Ch 10 — Vector Algebra]({REPO}/Ncert%20Solutions/Class%2012%20Maths%20Chapter%2010%20Vector%20Algebra.pdf)
- [Ch 11 — Three Dimensional Geometry]({REPO}/Ncert%20Solutions/Class%2012%20Maths%20Chapter%2011%20Three%20Dimensional%20Geometry.pdf)
- [Ch 12 — Linear Programming]({REPO}/Ncert%20Solutions/Class%2012%20Maths%20Chapter%2012%20Linear%20Programming.pdf)
- [Ch 13 — Probability]({REPO}/Ncert%20Solutions/Class%2012%20Maths%20Chapter%2013%20Probability.pdf)
""")

    with st.expander("🏫 SSM Question Papers"):
        st.markdown(f"""
- [SSM School Math Test]({REPO}/SSM%20Question%20Paper%20so%20far/SSM%20School%20Math%20Test.pdf)
""")

    with st.expander("📖 Book — Part 1 (lemh1dd)"):
        st.markdown(f"""
- [Chapter 1 — Relations and Functions]({REPO}/Book/lemh1dd/lemh101.pdf)
- [Chapter 2 — Inverse Trigonometric Functions]({REPO}/Book/lemh1dd/lemh102.pdf)
- [Chapter 3 — Matrices]({REPO}/Book/lemh1dd/lemh103.pdf)
- [Chapter 4 — Determinants]({REPO}/Book/lemh1dd/lemh104.pdf)
- [Chapter 5 — Continuity and Differentiability]({REPO}/Book/lemh1dd/lemh105.pdf)
- [Chapter 6 — Application of Derivatives]({REPO}/Book/lemh1dd/lemh106.pdf)
- [Appendix 1]({REPO}/Book/lemh1dd/lemh1a1.pdf)
- [Appendix 2]({REPO}/Book/lemh1dd/lemh1a2.pdf)
- [Answers]({REPO}/Book/lemh1dd/lemh1an.pdf)
- [Practice Sets]({REPO}/Book/lemh1dd/lemh1ps.pdf)
""")

    with st.expander("📖 Book — Part 2 (lemh2dd)"):
        st.markdown(f"""
- [Chapter 7 — Integrals]({REPO}/Book/lemh2dd/lemh201.pdf)
- [Chapter 8 — Application of Integrals]({REPO}/Book/lemh2dd/lemh202.pdf)
- [Chapter 9 — Differential Equations]({REPO}/Book/lemh2dd/lemh203.pdf)
- [Chapter 10 — Vector Algebra]({REPO}/Book/lemh2dd/lemh204.pdf)
- [Chapter 11 — Three-Dimensional Geometry]({REPO}/Book/lemh2dd/lemh205.pdf)
- [Chapter 12 — Linear Programming]({REPO}/Book/lemh2dd/lemh206.pdf)
- [Chapter 13 — Probability]({REPO}/Book/lemh2dd/lemh207.pdf)
- [Answers]({REPO}/Book/lemh2dd/lemh2an.pdf)
- [Practice Sets]({REPO}/Book/lemh2dd/lemh2ps.pdf)
""")

    with st.expander("🔬 Exemplar"):
        st.markdown(f"""
- [Ch 1 — Relations and Functions]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Maths%20Chapter%201%20-%20Relations%20And%20Functions%20(Book%20Solutions).pdf)
- [Ch 2 — Inverse Trigonometric Functions]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Maths%20Chapter%202%20-%20Inverse%20Trigonometric%20Functions%20(Book%20Solutions).pdf)
- [Ch 3 — Matrices]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Maths%20Chapter%203%20-%20Matrices%20(Book%20Solutions).pdf)
- [Ch 4 — Determinants]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Maths%20Chapter%204%20-%20Determinants%20(Book%20Solutions).pdf)
- [Ch 5 — Continuity and Differentiability]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Maths%20Chapter%205%20-%20Continuity%20And%20Differentiability%20(Book%20Solutions).pdf)
- [Ch 6 — Application of Derivatives]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Maths%20Chapter%206%20-%20Application%20Of%20Derivatives%20(Book%20Solutions).pdf)
- [Ch 7 — Integrals]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Maths%20Chapter%207%20-%20Integrals%20(Book%20Solutions)%20(1).pdf)
- [Ch 8 — Application of Integrals]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Maths%20Chapter%208%20-%20Application%20Of%20Integrals%20(Book%20Solutions).pdf)
- [Ch 9 — Differential Equations]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Maths%20Chapter%209%20-%20Differential%20Equations%20(Book%20Solutions).pdf)
- [Ch 10 — Vector Algebra]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Maths%20Chapter%2010%20-%20Vector%20Algebra%20(Book%20Solutions).pdf)
- [Ch 11 — Three-Dimensional Geometry]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Maths%20Chapter%2011%20-%20Three%20Dimensional%20Geometry%20(Book%20Solutions).pdf)
- [Ch 12 — Linear Programming]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Maths%20Chapter%2012%20-%20Linear%20Programming%20(Book%20Solutions).pdf)
- [Ch 13 — Probability]({REPO}/Exemplar/NCERT%20Exemplar%20for%20Class%2012%20Maths%20Chapter%2013%20-%20Probability%20(Book%20Solutions).pdf)
""")

    with st.expander("📝 Notes"):
        st.markdown(f"""
- [Ch 1 — Relations and Functions]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Maths%20Chapter%201%20-%20Free%20PDF.pdf)
- [Ch 2 — Inverse Trigonometric Functions]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Maths%20Chapter%202%20-%20Free%20PDF.pdf)
- [Ch 3 — Matrices]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Maths%20Chapter%203%20-%20Free%20PDF.pdf)
- [Ch 4 — Determinants]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Maths%20Chapter%204%20-%20Free%20PDF.pdf)
- [Ch 5 — Continuity and Differentiability]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Maths%20Chapter%205%20-%20Free%20PDF.pdf)
- [Ch 6 — Application of Derivatives]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Maths%20Chapter%206%20-%20Free%20PDF.pdf)
- [Ch 7 — Integrals]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Maths%20Chapter%207%20-%20Free%20PDF.pdf)
- [Ch 8 — Application of Integrals]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Maths%20Chapter%208%20-%20Free%20PDF.pdf)
- [Ch 9 — Differential Equations]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Maths%20Chapter%209%20-%20Free%20PDF.pdf)
- [Ch 10 — Three-Dimensional Geometry]({REPO}/Notes/Three-Dimensional%20Geometry%20Class%2012%20Maths%20Notes%20-%20Free%20PDF.pdf)
- [Ch 12 — Linear Programming]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Maths%20Chapter%2012%20-%20Free%20PDF.pdf)
- [Ch 13 — Probability]({REPO}/Notes/CBSE%20Notes%20Class%2012%20Maths%20Chapter%2013%20-%20Free%20PDF.pdf)
""")

    with st.expander("⭐ Important Questions"):
        st.markdown(f"""
- [Ch 1 — Relations and Functions]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Maths%20Chapter%201%20-%20Free%20PDF.pdf)
- [Ch 2 — Inverse Trigonometric Functions]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Maths%20Chapter%202%20-%20Free%20PDF.pdf)
- [Ch 3 — Matrices]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Maths%20Chapter%203%20-%20Free%20PDF.pdf)
- [Ch 4 — Determinants]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Maths%20Chapter%204%20-%20Free%20PDF.pdf)
- [Ch 5 — Continuity and Differentiability]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Maths%20Chapter%205%20-%20Free%20PDF.pdf)
- [Ch 6 — Application of Derivatives]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Maths%20Chapter%206%20-%20Free%20PDF.pdf)
- [Ch 7 — Integrals]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Maths%20Chapter%207%20-%20Free%20PDF.pdf)
- [Ch 8 — Application of Integrals]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Maths%20Chapter%208%20-%20Free%20PDF.pdf)
- [Ch 9 — Differential Equations]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Maths%20Chapter%209%20-%20Free%20PDF.pdf)
- [Ch 10 — Vector Algebra]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Maths%20Chapter%2010%20-%20Free%20PDF.pdf)
- [Ch 11 — Three-Dimensional Geometry]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Maths%20Chapter%2011%20-%20Free%20PDF.pdf)
- [Ch 12 — Linear Programming]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Maths%20Chapter%2012%20-%20Free%20PDF.pdf)
- [Ch 13 — Probability]({REPO}/Important%20Questions/Important%20Questions%20Class%2012%20Maths%20Chapter%2013%20-%20Free%20PDF.pdf)
""")

    with st.expander("🧠 Competency Based Questions"):
        st.markdown(f"""
- [Competency Based Questions — Vol 1]({REPO}/Competency%20Based%20Questions/Math_Grade12_V1.pdf)
- [Competency Based Questions — Vol 2]({REPO}/Competency%20Based%20Questions/Mathematics_12Vol2.pdf)
""")

    with st.expander("📄 Previous Year Questions (PYQ)"):
        st.markdown(f"""
- [2024 Question Paper (Set 1)]({REPO}/PYQ/65-1-1%20MATHEMATICS.pdf)
- [2024 Question Paper (Set 1 Alt)]({REPO}/PYQ/65-1-1%20Mathematcs.pdf)
- [2024 Question Paper]({REPO}/PYQ/65-1-1_Mathematics.pdf)
- [2024 Question Paper (Combined)]({REPO}/PYQ/65_1_1_Mathematics.pdf)
- [2025 Question Paper]({REPO}/PYQ/2413-1_65-1-1_Mathematics.pdf)
""")

    with st.expander("📋 Secret Assignments"):
        st.markdown(f"""
- [Day 1]({REPO}/Secret%20Assignment/Class%2012th%20Day%201%20Secret%20Assignment.pdf)
- [Day 2]({REPO}/Secret%20Assignment/Class%2012th%20Day%202%20Secret%20Assignment.pdf)
- [Day 3]({REPO}/Secret%20Assignment/Class%2012th%20Day%203%20Secret%20Assignment.pdf)
- [Day 8]({REPO}/Secret%20Assignment/Class%2012th%20Day%208%20Secret%20Assignment.pdf)
- [Day 12]({REPO}/Secret%20Assignment/Class%2012th%20Day%2012%20Secret%20Assignment.pdf)
- [Day 13]({REPO}/Secret%20Assignment/Class%2012th%20Day%2013%20Secret%20Assignment.pdf)
- [Day 15]({REPO}/Secret%20Assignment/Class%2012th%20Day%2015%20Secret%20Assignment.pdf)
- [Day 19]({REPO}/Secret%20Assignment/Class%2012th%20Day%2019%20Secret%20Assignment.pdf)
- [Day 22]({REPO}/Secret%20Assignment/Class%2012th%20Day%2022%20Secret%20Assignment.pdf)
- [Day 24]({REPO}/Secret%20Assignment/Class%2012th%20Day%2024%20Secret%20Assignment.pdf)
- [Day 27]({REPO}/Secret%20Assignment/Class%2012th%20Day%2027%20Secret%20Assignment.pdf)
- [Day 30]({REPO}/Secret%20Assignment/Class%2012th%20Day%2030%20Secret%20Assignment.pdf)
- [November Day 3]({REPO}/Secret%20Assignment/November%20Class%2012th%20Day%203%20Secret%20Assignment.pdf)
- [November Day 6]({REPO}/Secret%20Assignment/November%20Class%2012th%20Day%206%20Secret%20Assignment.pdf)
- [November Day 9]({REPO}/Secret%20Assignment/November%20Class%2012th%20Day%209%20Secret%20Assignment.pdf)
- [November Day 12]({REPO}/Secret%20Assignment/November%20Class%2012th%20Day%2012%20Secret%20Assignment.pdf)
""")

    st.divider()
    st.markdown("**💡 Try asking:**")
    st.markdown("""
- *Teach me Integration by Parts with examples*
- *Give me 5 HOTS questions on Probability — Bayes' theorem*
- *What are the standard results for Definite Integrals?*
- *Solve: Find the area bounded by y = x² and y = x*
- *What were the most repeated PYQ topics in 2024?*
- *Give me all important formulae for 3D Geometry*
""")
    st.divider()
    if st.button("🔄 Rebuild Vector Store"):
        with st.spinner("Re-ingesting all PDFs … (this takes a few minutes)"):
            st.session_state.pop("rag_chain", None)
            st.session_state.pop("vector_store", None)
            build_vector_store(force_rebuild=True)
        st.success("Vector store rebuilt!")

# ── Init vector store & chain (cached in session state) ───────────────────
if "vector_store" not in st.session_state:
    with st.spinner("⚙️ Loading your study material … first load takes ~1 min"):
        st.session_state["vector_store"] = build_vector_store(force_rebuild=False)

# Re-instantiate if provider changed or not initialized
target_provider = get_config("LLM_PROVIDER", "openai").lower()
if "rag_chain" not in st.session_state or st.session_state.get("_active_provider") != target_provider:
    try:
        st.session_state["rag_chain"] = build_rag_chain(st.session_state["vector_store"])
        st.session_state["_active_provider"] = target_provider
    except Exception as e:
        st.error(
            f"⚠️ **LLM Initialization Error**: {e}\n\n"
            "👉 If using Groq, ensure `LLM_PROVIDER = \"groq\"` and `GROQ_API_KEY = \"gsk_...\"` in Streamlit Secrets.\n"
            "👉 If using OpenAI, please ensure `OPENAI_API_KEY` is added to Streamlit Secrets."
        )
        st.stop()

# ── Chat history ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your Class 12 Math teacher, here to help you "
                "ace your **2027 CBSE Board Exams**.\n\n"
                "Tell me which chapter or topic you want to study, "
                "or ask me to generate **important formulae**, **HOTS questions**, "
                "or **step-by-step solutions** for any chapter! "
                "You can also **attach an image, PDF, or text file** using the 📎 button, "
                "or **paste a screenshot** with the 📋 button."
            ),
        }
    ]

# Display existing messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("attachment_name"):
            st.caption(f"📎 **Attached:** `{msg['attachment_name']}`")
            if msg.get("attachment_preview"):
                with st.expander("👁️ Attachment preview", expanded=False):
                    if msg.get("attachment_is_image"):
                        st.image(msg["attachment_preview"], use_container_width=True)
                    else:
                        st.text(msg["attachment_preview"][:2000])
        st.markdown(msg["content"])


# ── Helper: call vision-capable LLM directly (image path) ──────────────
def _ask_vision_llm(question: str, image_data_uri: str, context_text: str) -> str:
    """
    Send a multimodal (text + image) message to Google Gemini Flash (free tier).
    Vision is always handled by Gemini regardless of LLM_PROVIDER,
    since Groq has no vision models and OpenAI requires paid credits.
    Falls back gracefully when GOOGLE_API_KEY is not set.
    """
    import base64, re as _re
    from google import genai
    from google.genai import types

    google_key = get_config("GOOGLE_API_KEY")
    if not google_key:
        return (
            "⚠️ **Image analysis requires a `GOOGLE_API_KEY`** (free).\n\n"
            "👉 Get one at https://aistudio.google.com/app/apikey — it's free, no billing needed.\n"
            "Then add `GOOGLE_API_KEY = \"AIza...\"` to your `.env` file or Streamlit Secrets and reload the app."
        )

    # Extract raw base64 bytes from the data URI (data:<mime>;base64,<data>)
    match = _re.match(r"data:(?P<mime>[^;]+);base64,(?P<data>.+)", image_data_uri)
    if not match:
        return "⚠️ Could not parse the attached image. Please try uploading it again."
    mime_type = match.group("mime")
    image_bytes = base64.b64decode(match.group("data"))

    prompt = (
        "You are an expert Class 12 CBSE Mathematics teacher.\n\n"
        f"Context from the student's study materials:\n{context_text}\n\n"
        f"The student has attached an image and asks:\n{question}"
    )

    client = genai.Client(api_key=google_key)
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
    )
    return response.text




# -- Input area: file uploader + paste button + chat input -----------------
ext_list = ", ".join(f".{e}" for e in sorted(SUPPORTED_EXTS))

col_upload, col_paste = st.columns([3, 1], vertical_alignment="bottom")
with col_upload:
    uploaded_file = st.file_uploader(
        f"📎 Attach a file — {ext_list}",
        type=list(SUPPORTED_EXTS),
        label_visibility="visible",
        help="Attach an image (diagram / question screenshot), PDF, or .txt file.",
    )
with col_paste:
    paste_result = paste_image_button(
        "📋 Paste image",
        background_color="#444654",
        hover_background_color="#565869",
        key="clipboard_paste",
    )

if user_input := st.chat_input("Ask your Mathematics teacher …"):
    attachment_name: str | None = None
    attachment_text: str | None = None
    attachment_image_uri: str | None = None
    attachment_preview = None
    attachment_is_image = False

    # Clipboard paste takes priority over file uploader
    if paste_result.image_data is not None:
        import io as _io
        buf = _io.BytesIO()
        paste_result.image_data.save(buf, format="PNG")
        file_bytes = buf.getvalue()
        attachment_name = "pasted-image.png"
        attachment_image_uri, _ = image_to_base64_uri(file_bytes, attachment_name)
        attachment_preview = file_bytes
        attachment_is_image = True
    elif uploaded_file is not None:
        attachment_name = uploaded_file.name
        file_bytes = uploaded_file.read()
        if is_image(attachment_name):
            attachment_image_uri, _ = image_to_base64_uri(file_bytes, attachment_name)
            attachment_preview = file_bytes
            attachment_is_image = True
        elif attachment_name.lower().endswith(".pdf"):
            attachment_text = extract_text_from_pdf(file_bytes)
            attachment_preview = attachment_text
        else:
            attachment_text = extract_text_from_txt(file_bytes)
            attachment_preview = attachment_text

    user_msg: dict = {
        "role": "user",
        "content": user_input,
        "attachment_name": attachment_name,
        "attachment_preview": attachment_preview,
        "attachment_is_image": attachment_is_image,
    }
    st.session_state["messages"].append(user_msg)

    with st.chat_message("user"):
        if attachment_name:
            st.caption(f"📎 **Attached:** `{attachment_name}`")
            if attachment_preview is not None:
                with st.expander("👁️ Attachment preview", expanded=False):
                    if attachment_is_image:
                        st.image(attachment_preview, use_container_width=True)
                    else:
                        st.text(str(attachment_preview)[:2000])
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            recent_messages = st.session_state["messages"][:-1]
            if len(recent_messages) > 4:
                recent_messages = recent_messages[-4:]
            clean_history = [{"role": m["role"], "content": m["content"]} for m in recent_messages]
            chat_history = convert_history(clean_history)

            try:
                if attachment_image_uri is not None:
                    retriever = st.session_state["vector_store"].as_retriever(
                        search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20},
                    )
                    source_docs = retriever.invoke(user_input)
                    context_text = "\n\n".join(
                        f"Document {i+1} (Source: {d.metadata.get('source','?')}, "
                        f"Page: {d.metadata.get('page','?')}):\n{d.page_content}"
                        for i, d in enumerate(source_docs)
                    )
                    answer = _ask_vision_llm(user_input, attachment_image_uri, context_text)
                    sources = source_docs
                elif attachment_text is not None:
                    augmented = f"{user_input}\n\n--- Attached file: {attachment_name} ---\n{attachment_text}"
                    result = st.session_state["rag_chain"].invoke(
                        {"question": augmented, "chat_history": chat_history}
                    )
                    answer = result["answer"]
                    sources = result.get("source_documents", [])
                else:
                    result = st.session_state["rag_chain"].invoke(
                        {"question": user_input, "chat_history": chat_history}
                    )
                    answer = result["answer"]
                    sources = result.get("source_documents", [])

                st.markdown(answer)

                if sources:
                    with st.expander("📎 Sources from your study material", expanded=False):
                        seen = set()
                        for doc in sources:
                            src = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "?")
                            label = f"{src}  — page {page}"
                            if label not in seen:
                                st.markdown(f"- `{label}`")
                                seen.add(label)

                st.session_state["messages"].append({"role": "assistant", "content": answer})

            except Exception as e:
                err_msg = str(e)
                provider = get_config("LLM_PROVIDER", "openai").lower()
                if "rate_limit" in err_msg.lower() or "quota" in err_msg.lower() or "429" in err_msg:
                    if provider == "groq":
                        st.warning("⚠️ **Groq Rate Limit**: Please wait ~10-15 seconds and try again.")
                    else:
                        st.error(
                            "⚠️ **OpenAI Quota / Rate Limit Exceeded**: Check billing on "
                            "[OpenAI Billing](https://platform.openai.com/account/billing/overview) "
                            "or switch to Groq in Secrets."
                        )
                else:
                    st.error(f"⚠️ Error processing your request: {err_msg}")
