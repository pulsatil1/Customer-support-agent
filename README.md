# Customer-support-agent
This project showcases a conversational agent designed for a virtual platform that sells T-shirts. The agent is powered by:

- **Local LLM Model**: [Llama 3.1 (8B)](https://huggingface.co/meta-llama/Llama-3.1-8B) via [Ollama](https://ollama.com/)
- **Embedding Model**: [Snowflake's Arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m)
- **Agent Framework**: [LangChain+LangGraph](https://langchain-ai.github.io/langgraph) for building the agent
- **Logging**: [Langfuse](https://langfuse.com/) for agent activity logs
- **User Interface**: [Gradio](https://www.gradio.app/) for a simple and interactive UI

## Features

- **Primary Language**: English, but the agent can also handle any language supported by Llama-3.1-8B.  
- **Extended Language Support**: For better results in other languages consider using larger, more powerful models.

## Future Improvements

- **Streaming Responses**: Implement a real-time streaming response feature.
- **Quality Assessment**: Introduce methods to evaluate the model's responses (e.g., leveraging LLM-as-a-judge).
- **Agent Testing Mechanism**: Develop a robust testing framework to compare performance metrics after changing architecture, models, hyperparameters, or prompts.
- **Support request**: Replace the “dummy” support request function with an actual integration to a support team.

## References and Inspiration

- [Simple Conversational Agent Tutorial](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_conversational_agent.ipynb)
- [Medium: How to Build AI Agents with LangGraph](https://medium.com/@lorevanoudenhove/how-to-build-ai-agents-with-langgraph-a-step-by-step-guide-5d84d9c7e832)
- [Medium: Privacy-First RAG with Deepseek-r1 and Ollama](https://blog.gopenai.com/how-to-build-a-privacy-first-rag-using-deepseek-r1-langchain-and-ollama-c5133a8514dd)
- [LLMs in Finance](https://github.com/hananedupouy/LLMs-in-Finance)
- [AI Engineering Hub: Agentic RAG](https://github.com/patchy631/ai-engineering-hub/tree/main/agentic_rag)
- [Claude RAG Agent](https://www.mongodb.com/developer/products/atlas/claude_3_5_sonnet_rag/)
- [LangGraph Agentic RAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)
- [LangGraph CRAG Local Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag_local)
- [LangGraph Customer Support Tutorial](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support)

---
