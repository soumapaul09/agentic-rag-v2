from typing import Literal

from langchain_groq import ChatGroq
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from loguru import logger
from pydantic import BaseModel, Field

from config_loader import get_config
from search_utils import SearchEngine


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Relevance score: 'yes' if relevant, or 'no' if not relevant")


class AgenticRAG:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        search_engine: SearchEngine,
        grade_prompt: str,
        rewrite_prompt: str,
        generate_prompt: str,
    ):
        self.llm = ChatGroq(model=model_name, temperature=temperature)
        self.grader = ChatGroq(model=model_name, temperature=temperature)
        self.search_engine = search_engine
        self.grade_prompt = grade_prompt
        self.rewrite_prompt = rewrite_prompt
        self.generate_prompt = generate_prompt
        self.graph = self._build_graph()

    def _generate_query_or_respond(self, state: MessagesState):
        llm_response = self.llm.bind_tools([self.search_engine.hybrid_search]).invoke(state["messages"])
        return {"messages": [llm_response]}

    def _grade_documents(self, state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        user_question = state["messages"][0].content
        retrieved_context = state["messages"][-1].content

        grading_prompt = self.grade_prompt.format(question=user_question, context=retrieved_context)
        grading_response = self.grader.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": grading_prompt}]
        )
        relevance_score = grading_response.binary_score

        if relevance_score == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"

    def _rewrite_question(self, state: MessagesState):
        msg_history = state["messages"]
        original_question = msg_history[0].content
        rewrite_prompt = self.rewrite_prompt.format(question=original_question)
        rewritten_response = self.llm.invoke([{"role": "user", "content": rewrite_prompt}])
        return {"messages": [{"role": "user", "content": rewritten_response.content}]}

    def _generate_answer(self, state: MessagesState):
        user_question = state["messages"][0].content
        retrieved_context = state["messages"][-1].content
        answer_prompt = self.generate_prompt.format(question=user_question, context=retrieved_context)
        final_response = self.llm.invoke([{"role": "user", "content": answer_prompt}])
        return {"messages": [final_response]}

    def _build_graph(self):
        workflow = StateGraph(MessagesState)

        workflow.add_node("generate_query_or_respond", self._generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([self.search_engine.hybrid_search]))
        workflow.add_node("rewrite_question", self._rewrite_question)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.add_edge(START, "generate_query_or_respond")

        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )

        workflow.add_conditional_edges(
            "retrieve",
            self._grade_documents,
        )
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def query(self, user_query: str) -> str:
        try:
            result = self.graph.invoke({"messages": [{"role": "user", "content": user_query}]})
        except GraphRecursionError as e:
            logger.error(f"Graph recursion error: {e}")
            return "I'm sorry, I can't answer that question."
        return result["messages"][-1].content


def rag_dag(query: str) -> str:
    config = get_config()

    search_engine = SearchEngine(
        faiss_index_path=config.get("paths.faiss_index"),
        faiss_metadata_path=config.get("paths.faiss_metadata"),
        bm25_index_path=config.get("paths.bm25_index"),
        bm25_metadata_path=config.get("paths.bm25_metadata"),
        embedding_model_name=config.get("models.embedding.name"),
        reranker_model_name=config.get("models.reranker.name"),
        reciprocal_rank_k=config.get("search.reciprocal_rank_k")
    )

    rag = AgenticRAG(
        model_name=config.get("rag.agentic.default_model"),
        temperature=config.get("rag.agentic.default_temperature"),
        search_engine=search_engine,
        grade_prompt=config.get("prompts.grade"),
        rewrite_prompt=config.get("prompts.rewrite"),
        generate_prompt=config.get("prompts.generate")
    )
    return rag.query(query)


if __name__ == "__main__":
    config = get_config()

    search_engine = SearchEngine(
        faiss_index_path=config.get("paths.faiss_index"),
        faiss_metadata_path=config.get("paths.faiss_metadata"),
        bm25_index_path=config.get("paths.bm25_index"),
        bm25_metadata_path=config.get("paths.bm25_metadata"),
        embedding_model_name=config.get("models.embedding.name"),
        reranker_model_name=config.get("models.reranker.name"),
        reciprocal_rank_k=config.get("search.reciprocal_rank_k")
    )

    # query = "What is Aristotle's fallacy?"
    query = "explain all variables in logistic regression equation"
    rag = AgenticRAG(
        model_name=config.get("rag.agentic.default_model"),
        temperature=config.get("rag.agentic.default_temperature"),
        search_engine=search_engine,
        grade_prompt=config.get("prompts.grade"),
        rewrite_prompt=config.get("prompts.rewrite"),
        generate_prompt=config.get("prompts.generate")
    )
    response = rag.query(query)
    print(f"Query: {query}")
    print(f"Response: {response}")