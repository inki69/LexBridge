def build_prompt(context: str, query: str) -> str:
    return f"""You are a U.S. Employment Law assistant. Answer the user's question using ONLY the information provided in the context below.

If the answer cannot be found in the context, respond with: "I don't have enough information in my knowledge base to answer this question."

Do NOT fabricate information, cite external sources, or answer from general knowledge.

Context:
{context}

Question: {query}

Answer:"""
