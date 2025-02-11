import os
import dotenv
import json
import uuid
import gradio as gr
import logging
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from langfuse.callback import CallbackHandler

from models import AgentState

dotenv.load_dotenv()

class ConversationalAgent():
    def __init__(self):
        self.model_name = 'llama3.1'
        self.data_folder = "data"

        self.get_data()
        self.create_embeddings()
        self.setup_tools()
        self.setup_prompt()
        self.setup_llm()
        self.setup_runnables()
        self.setup_graph()

    def get_data(self):
        shirts_filepath = os.path.join(self.data_folder, "Shirts.json")
        with open(shirts_filepath, 'r', encoding='utf-8') as file:
            shirts_info = json.load(file)

        self.info = shirts_info
        
        faq_filepath = os.path.join(self.data_folder, "FAQ.txt")
        with open(faq_filepath, 'r', encoding='utf-8-sig') as text_file:
            text = text_file.read()

        qa_pairs = []
        parts = text.split("Q:")
        for part in parts:
            if not part.strip():
                continue

            qa_pairs.append("Q:" + part)

        self.documents = qa_pairs

    def create_embeddings(self):
        embedding_model = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-m")
        doc_splits = [Document(page_content=doc) for doc in self.documents]
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=embedding_model,
        )
        self.retriever = vectorstore.as_retriever(k=4)

    def get_retriever_tool(self):
        retriever_tool = create_retriever_tool(
            self.retriever,
            "retrieve_FAQs",
            "Search and return FAQs with answers.",
        )
        return retriever_tool
    
    @tool("Send_support_request")
    def support_request_tool(request: str) -> str:
        """Sends request to support
        
        Args:
            request: Key details for the support team
        """
        
        return f"This request has been sent to the support: \n{request}"

    def setup_tools(self):
        rag_tool = self.get_retriever_tool()
        self.tools = [rag_tool, self.support_request_tool]

    def setup_prompt(self):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                        You are a conversational agent for TeeCustomizer, a virtual platform for designing and ordering customizable t-shirts. 
                        Your responsibilities include assisting users in selecting t-shirt styles, colors, sizes, and printing options. 
                        You should also answer frequently asked questions about t-shirt customization, order process, and platform policies, and log support requests when needed.
                        When interacting with users, follow these guidelines:
                        1. Greet the user warmly and ask how you can assist them.
                        2. Provide clear, concise, and friendly responses.
                        3. When a user asks for customization advice, reference the t-shirt customization document to suggest suitable options.
                        4. For FAQs, refer to the FAQ document to provide accurate information.
                        5. If a user has a support request, send a detailed support request. Then tell the user that the support request has been submitted.
                        6. Ensure the conversation remains helpful, professional, and user-focused.

                        If you cannot help the user, send a request to support and inform the user about it.

                        Don't mention that you received information from documents.

                        Begin by welcoming the user and asking if they need assistance with designing their custom t-shirt.

                        Information about T-Shirts customization:
                        {shirts_info}
                    """
                ),
                ("placeholder", "{messages}")
            ]
        ).partial(shirts_info=self.info)

    def setup_llm(self):
        self.llm = ChatOllama(model=self.model_name, temperature=0)

    def handle_tool_error(self, state) -> dict:
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    def create_tool_node_with_fallback(self) -> dict:
        return ToolNode(self.tools).with_fallbacks(
            [RunnableLambda(self.handle_tool_error)], exception_key="error"
        )
    
    def setup_runnables(self):
        self.assistant_runnable = self.prompt | self.llm.bind_tools(self.tools)
        self.reserve_runnable = self.prompt | self.llm

    def setup_graph(self):
        workflow = StateGraph(AgentState)

        # Define the nodes
        workflow.add_node("assistant", Assistant(self.assistant_runnable, self.reserve_runnable))
        workflow.add_node("tools", self.create_tool_node_with_fallback())

        workflow.add_edge(START, "assistant")
        workflow.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        workflow.add_edge("tools", "assistant")
        memory = MemorySaver()
                    
        self.graph = workflow.compile(checkpointer=memory)

    def chat(self, question, history):
        thread_id = str(uuid.uuid4())
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        chat_history = []
        for message in history:
            if message['role'] == 'user':
                chat_history.append(HumanMessage(content=message['content']))
            elif message['role'] == 'assistant':
                chat_history.append(AIMessage(content=message['content']))

        if history:
            self.graph.update_state(config, {"messages": history})

        langfuse_handler = CallbackHandler()
        config["callbacks"] = [langfuse_handler]

        try:
            agent_result = self.graph.invoke(
                    {"messages": HumanMessage(content=question)}, 
                    config
            )
        except Exception as e:                  
            logging.error(f' | agent error: {e}')
            raise gr.Error(f"An unexpected error occurred: {e}", duration = 0)
        
        return agent_result["messages"][-1].content


class Assistant:
    def __init__(self, runnable: Runnable, reserve_runnable: Runnable):
        self.runnable = runnable
        self.reserve_runnable = reserve_runnable
        self.counter = 0

    def __call__(self, state: AgentState, config: RunnableConfig):
        while True:
            
            self.counter += 1
            if self.counter > 5:
                result = self.reserve_runnable.invoke(state)
            else:
                result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}