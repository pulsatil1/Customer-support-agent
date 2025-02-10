import os
import gradio as gr
import logging
import requests
from core import ConversationalAgent


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="error.log",
    )

    port = os.environ.get("PORT")
    if not port:
        port = 8080

    agent = ConversationalAgent()

    chat_bot = gr.ChatInterface(
        fn=agent.chat,
        type='messages',
        title="Customer support agent",
        description=f"Chat with conversational agent",
        examples=["Can I add my own design to the t-shirt?", "Do you offer kids’ sizes?"], 
        show_progress="full",
        save_history=True,
        theme="ocean"
        )
    chat_bot.launch(
        server_port=int(port)
    )