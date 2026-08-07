from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from memory import format_history_for_prompt
from websearch import web_search


def get_confidence(question, draft_answer, llm):
    """Ask the model to rate its own confidence in the draft answer."""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are evaluating the quality of a research answer. "
         "Given the question and the draft answer, rate your confidence that the answer is accurate and complete. "
         "Reply with ONLY a number between 0 and 100. Nothing else."),
        ("human", f"Question: {question}\n\nDraft answer: {draft_answer}")
    ])
    chain = prompt | llm
    try:
        result = chain.invoke({})
        score = int(''.join(filter(str.isdigit, result.content.strip()))[:3])
        return min(max(score, 0), 100)
    except:
        return 50  # assume uncertain if parsing fails


def build_chain(vectorstore, llm, system_prompt):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 5}),
        create_stuff_documents_chain(llm, prompt)
    )

def format_history_safe(history_text):
    return history_text.replace("{", "{{").replace("}", "}}")

def get_answer(question, vectorstore, llm, history, use_web=False):
    history_text = format_history_for_prompt(history, max_turns=4)
    history_block = f"\n\nPrevious conversation:\n{format_history_safe(history_text)}" if history_text else ""

    base_system = (
        "You are a research assistant and also a general-purpose AI with broad knowledge. "
        "You have access to retrieved context from research papers and documents — use it ONLY if it is directly relevant to the question. "
        "If the retrieved context is about a completely different topic than the question, IGNORE it entirely. "
        "For questions about yourself (your capabilities, what you know, your nature), answer from your own knowledge — do not look for answers in the research context. "
        "For general knowledge questions not covered by the context, answer from your own training. "
        "Always give the best and most honest answer you can."
        + history_block
    )

    web_results = []

    # --- Pass 1: answer from local RAG only ---
    if not use_web:
        system_prompt = base_system + "\n\nRetrieved context (use if relevant):\n{context}"
        chain = build_chain(vectorstore, llm, system_prompt)
        response = chain.invoke({"input": f"<|think|> {question}"})
        draft_answer = response["answer"]

        # --- Confidence check ---
        print("  🤔 Checking confidence...", flush=True)
        confidence = get_confidence(question, draft_answer, llm)
        print(f"  Confidence: {confidence}%", flush=True)

        if confidence >= 70:
            return draft_answer, []

        # --- Pass 2: confidence too low, search web ---
        print("  📊 Confidence below 70% — searching web...", flush=True)
        use_web = True

    # --- Web search pass ---
    web_text, web_results = web_search(question)
    web_block = f"\n\nWeb search results:\n{web_text}" if web_text else ""

    system_prompt = (
        base_system
        + web_block
        + "\n\nRetrieved context (use if relevant):\n{context}"
    )
    chain = build_chain(vectorstore, llm, system_prompt)
    response = chain.invoke({"input": f"<|think|> {question}"})
    return response["answer"], web_results