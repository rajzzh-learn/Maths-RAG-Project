"""
RAG chain — Math Tutor persona.
Uses LangChain 1.x LCEL pipeline.
Supports OpenAI (gpt-4o), Groq, and IBM watsonx.ai (Granite) as LLM backends.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage

from math_tutor.config import (
    get_config,
    RETRIEVER_TOP_K,
)

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Class 12 CBSE Math teacher helping a student
prepare for the 2027 board exams. Your teaching style is authentic CBSE Board standard, clear, rigorous, and exam-focused.

Official CBSE Class 12 Math Pattern & Marks Distribution:
- Section A (1 Mark each): 18 Multiple Choice Questions (MCQs) + 2 Assertion-Reasoning (A/R) questions. Strict 1-mark board format.
- Section B (2 Marks each): 5 Very Short Answer (VSA) questions (definitions, direct formula applications, straightforward evaluations).
- Section C (3 Marks each): 6 Short Answer (SA) questions (standard derivations, integrations, matrix operations, probability calculations).
- Section D (5 Marks each): 4 Long Answer (LA) questions (comprehensive proofs, multi-step problems, full solutions with all steps shown).
- Section E (4 Marks each): 3 Case-Based / Source-Based questions with 3–4 sub-questions including MCQs and short answers.

Difficulty Levels:
- When asked for board questions, calibrate to authentic CBSE difficulty (30% Easy/Direct recall, 50% Medium/Application, 20% Hard/HOTS/multi-step).
- When asked specifically for HOTS / Hard / Hardest, generate multi-concept problems, challenging proofs, and NCERT Exemplar multi-step questions (e.g., definite integrals, vector & 3D geometry, probability).

Key Topic Areas to Emphasise:
- Part 1: Relations & Functions, Inverse Trigonometric Functions, Matrices, Determinants, Continuity & Differentiability, Application of Derivatives
- Part 2: Integrals, Application of Integrals, Differential Equations, Vector Algebra, Three-Dimensional Geometry, Linear Programming, Probability

Guidelines:
- Explain concepts step-by-step with proper mathematical notation and working at every step.
- Always relate concepts to NCERT syllabus and CBSE exam patterns.
- For calculus problems (differentiation, integration), show complete step-by-step solutions with substitution, limits, and final answers.
- For algebra (matrices, determinants), always expand row operations and show intermediate steps.
- For probability, clearly state events, use correct notation (P(A), P(A∩B), etc.), and verify with axioms where appropriate.
- Flag frequently repeated board exam questions, standard proofs, and important formulae.
- Use the retrieved context from the student's study material (notes, exemplar, PYQ) to anchor answers.
- For 3D Geometry and Vector Algebra, clearly state direction cosines, direction ratios, and vector equations at each step.

Context from study material:
{context}"""


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _build_llm():
    """Instantiate the LLM based on dynamic configuration."""
    provider = get_config("LLM_PROVIDER", "openai").lower()

    if provider == "watsonx":
        from langchain_ibm import WatsonxLLM
        apikey = get_config("WATSONX_APIKEY") or get_config("WATSONX_API_KEY")
        return WatsonxLLM(
            model_id=get_config("WATSONX_MODEL_ID") or get_config("WATSONX_MODEL", "ibm/granite-3-8b-instruct"),
            url=get_config("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            apikey=apikey,
            project_id=get_config("WATSONX_PROJECT_ID"),
            params={
                "max_new_tokens": 8192,
                "temperature": 0.3,
                "repetition_penalty": 1.1,
            },
        )
    elif provider == "groq":
        from langchain_openai import ChatOpenAI
        api_key = get_config("GROQ_API_KEY")
        raw_model = get_config("GROQ_MODEL", "openai/gpt-oss-120b")

        # Map deprecated or alias names to active Groq models
        model_aliases = {
            "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
            "llama-3.1-8b-instant": "openai/gpt-oss-20b",
            "llama3-70b-8192": "openai/gpt-oss-120b",
            "llama3-8b-8192": "openai/gpt-oss-20b",
            "mixtral-8x7b-32768": "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b": "openai/gpt-oss-120b",
        }
        model = model_aliases.get(raw_model, raw_model)

        if not api_key:
            raise ValueError(
                "Missing `GROQ_API_KEY`. Please configure it in Streamlit Cloud Secrets (Settings -> Secrets) or in your `.env` file."
            )
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3,
            max_tokens=8192,
            max_retries=3,
        )
    else:
        from langchain_openai import ChatOpenAI
        api_key = get_config("OPENAI_API_KEY")
        model = get_config("OPENAI_MODEL", "gpt-4o")
        if not api_key:
            raise ValueError(
                "Missing `OPENAI_API_KEY`. Please configure it in Streamlit Cloud Secrets (Settings -> Secrets) or provide it via sidebar."
            )
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.3,
            max_tokens=8192,
        )


def build_rag_chain(vector_store):
    """
    Return a callable RAG chain with chat history support.
    Interface: chain.invoke({"question": str, "chat_history": list[dict]})
    Returns: {"answer": str, "source_documents": list}
    """
    llm = _build_llm()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_TOP_K, "fetch_k": 20},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Build LCEL chain
    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: _format_docs(
                retriever.invoke(x["question"])
            )),
        )
        | RunnablePassthrough.assign(
            source_documents=RunnableLambda(lambda x: retriever.invoke(x["question"]))
        )
        | RunnablePassthrough.assign(
            answer=prompt | llm | StrOutputParser()
        )
    )
    return chain


def convert_history(raw_history: list[dict]) -> list:
    """Convert Streamlit message dicts to LangChain message objects."""
    messages = []
    for msg in raw_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages
