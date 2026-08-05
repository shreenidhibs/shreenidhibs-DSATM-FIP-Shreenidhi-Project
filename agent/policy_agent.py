import os
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from agents.ra_agent import RetrievalAgent


load_dotenv()


class PolicyHandbookAgent:
    """
    Main policy-answering agent.

    It uses RetrievalAgent to search institutional PDFs and then asks
    the language model to answer only from the retrieved context.
    """

    def __init__(self) -> None:
        groq_key = os.getenv("GROQ_API_KEY")

        if not groq_key:
            raise ValueError(
                "GROQ_API_KEY is missing in the .env file."
            )

        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,
            max_tokens=1024,
            api_key=groq_key
        )

        self.retrieval_agent = RetrievalAgent(
            relevance_threshold=0.30,
            number_of_results=5
        )

    def answer_from_context(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Generate an answer using only retrieved institutional context.
        """

        prompt = f"""
You are a Policy and Student Handbook Assistant.

Answer the question using only the institutional document context below.

Rules:

1. Give the direct answer first.
2. Do not use outside knowledge.
3. Do not invent policy details.
4. Mention the document name and page number.
5. Include relevant conditions, limits and exceptions.
6. If the context does not clearly answer the question, return exactly:

NOT_FOUND

Question:
{query}

Institutional document context:
{context}
"""

        response = self.llm.invoke(prompt)

        return response.content.strip()

    @staticmethod
    def not_found_response() -> str:
        return (
            "The requested information was not found in the uploaded "
            "institutional policy documents. Please check the latest "
            "official handbook or contact the academic office."
        )

    def run(self, query: str) -> dict[str, Any]:
        """
        Execute retrieval followed by grounded policy answering.
        """

        cleaned_query = query.strip()

        if not cleaned_query:
            return {
                "query": query,
                "answer": "Please enter a valid policy question.",
                "answer_source": "validation_error",
                "retrieval_status": "invalid_query",
                "web_search_used": False,
                "relevance_score": 0.0,
                "sources": []
            }

        retrieval_result = self.retrieval_agent.retrieve(
            cleaned_query
        )

        documents = retrieval_result["documents"]
        highest_score = retrieval_result["highest_score"]
        retrieval_status = retrieval_result["retrieval_status"]

        if documents:
            context = self.retrieval_agent.format_context(
                documents
            )

            answer = self.answer_from_context(
                query=cleaned_query,
                context=context
            )

            if answer.upper() != "NOT_FOUND":
                return {
                    "query": cleaned_query,
                    "answer": answer,
                    "answer_source": "institutional_pdf",
                    "retrieval_status": retrieval_status,
                    "web_search_used": False,
                    "relevance_score": highest_score,
                    "sources": self.retrieval_agent.build_sources(
                        documents
                    )
                }

        return {
            "query": cleaned_query,
            "answer": self.not_found_response(),
            "answer_source": "not_found_in_documents",
            "retrieval_status": retrieval_status,
            "web_search_used": False,
            "relevance_score": highest_score,
            "sources": []
        }


if __name__ == "__main__":
    agent = PolicyHandbookAgent()

    result = agent.run(
        "What is the minimum attendance requirement?"
    )

    print(result)
