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


# -- Helper: vision LLM call -----------------------------------------------
def _ask_vision_llm(question: str, image_data_uri: str, context_text: str) -> str:
    provider = get_config("LLM_PROVIDER", "openai").lower()
    vision_content = [
        {
            "type": "text",
            "text": (
                "You are an expert Class 12 CBSE Mathematics teacher.\n\n"
                f"Context from the student's study materials:\n{context_text}\n\n"
                f"The student has attached an image and asks:\n{question}"
            ),
        },
        {"type": "image_url", "image_url": {"url": image_data_uri}},
    ]
    if provider == "groq":
        from openai import OpenAI
        client = OpenAI(api_key=get_config("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model=get_config("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
            messages=[{"role": "user", "content": vision_content}],
            max_tokens=4096, temperature=0.3,
        )
        return resp.choices[0].message.content
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=get_config("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=get_config("OPENAI_MODEL", "gpt-4o"),
            messages=[{"role": "user", "content": vision_content}],
            max_tokens=4096, temperature=0.3,
        )
        return resp.choices[0].message.content
    else:
        return (
            "⚠️ **Image vision is not supported for the `watsonx` provider.** "
            "Please switch to `groq` or `openai`, or describe your question in text."
        )

# ── Chat input ─────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask your Math teacher …"):
    # Show student message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get answer from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            # Limit chat history to the last 2 turns (4 messages) to minimize prompt token footprint
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

                # Show source references (collapsed)
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
                        st.warning(
                            "⚠️ **Groq Rate Limit (RPM/TPM)**: Groq's free tier has a per-minute token limit. Please wait ~10-15 seconds and try again."
                        )
                    else:
                        st.error(
                            "⚠️ **OpenAI Quota / Rate Limit Exceeded**: Your OpenAI account has run out of credits or reached its usage limit.\n\n"
                            "👉 Please check your billing on [OpenAI Billing](https://platform.openai.com/account/billing/overview) or switch to Groq (free) in Secrets."
                        )
                else:
                    st.error(f"⚠️ Error processing your request: {err_msg}")
