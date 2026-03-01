from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import Graph, StateGraph

from config_loader import get_config
from search_utils import SearchEngine


class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "The messages in the conversation"]
    search_results: Annotated[list[str], "The search results from hybrid search"]


class SimpleRAG:
    def __init__(self, model_name: str, temperature: float, top_k: int, search_engine: SearchEngine, system_prompt: str):
        self.llm = ChatGroq(model=model_name, temperature=temperature)
        self.top_k = top_k
        self.search_engine = search_engine
        self.system_prompt = system_prompt
        self.graph = self._build_graph()

    def _search_node(self, state: AgentState) -> AgentState:
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, HumanMessage):
            raise ValueError("Last message must be from human")

        results = self.search_engine.hybrid_search(last_msg.content, top_k=self.top_k)
        return {"messages": state["messages"], "search_results": results}

    def _answer_node(self, state: AgentState) -> AgentState:
        last_msg = state["messages"][-1]
        if not isinstance(last_msg, HumanMessage):
            raise ValueError("Last message must be from human")

        context_text = "\n\n".join(state["search_results"])
        system_prompt = self.system_prompt.format(context=context_text)

        llm_response = self.llm.invoke(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": last_msg.content}]
        )

        updated_messages = list(state["messages"]) + [AIMessage(content=llm_response.content)]
        return {"messages": updated_messages, "search_results": state["search_results"]}

    def _build_graph(self) -> Graph:
        workflow = StateGraph(AgentState)

        workflow.add_node("search", self._search_node)
        workflow.add_node("answer", self._answer_node)

        workflow.add_edge("search", "answer")
        workflow.set_entry_point("search")
        workflow.set_finish_point("answer")

        return workflow.compile()

    def query(self, user_query: str) -> str:
        initial_state = {"messages": [HumanMessage(content=user_query)], "search_results": []}
        result = self.graph.invoke(initial_state)
        return result["messages"][-1].content


def rag_chain(query: str) -> str:
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

    rag = SimpleRAG(
        model_name=config.get("rag.simple.default_model"),
        temperature=config.get("rag.simple.default_temperature"),
        top_k=config.get("rag.simple.default_top_k"),
        search_engine=search_engine,
        system_prompt=config.get("prompts.simple_rag_system")
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

    query = "explain all variables in logistic regression equation?"
    rag = SimpleRAG(
        model_name=config.get("rag.simple.default_model"),
        temperature=config.get("rag.simple.default_temperature"),
        top_k=config.get("rag.simple.default_top_k"),
        search_engine=search_engine,
        system_prompt=config.get("prompts.simple_rag_system")
    )
    response = rag.query(query)
    print(f"Query: {query}")
    print(f"Response: {response}")