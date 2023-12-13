# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# OpenAI Chat completion
import os
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools
from dotenv import load_dotenv

import asyncio

from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    AssistantRolePrompt,
)

load_dotenv()

RAQA_PROMPT_TEMPLATE = """
Use the provided context to answer the user's query. 

You may not answer the user's query unless there is specific context in the following text.

If you do not know the answer, or cannot answer, please respond with "I don't know".

Context:
{context}
"""

USER_PROMPT_TEMPLATE = """
User Query:
{user_query}
"""

def load_vector_db_from_local_file(file_path="data/KingLear.txt"):
    """generates the vector database object base on a local file"""
    
    # load text file and split into chunk of documents
    text_loader = TextFileLoader(file_path)
    documents = text_loader.load_documents()
    text_splitter = CharacterTextSplitter()
    split_documents = text_splitter.split_texts(documents)
    
    # initialize vector db and build from list of documents
    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))
    return vector_db
    
    
def get_formatted_prompts(vector_db_retriever: VectorDatabase, user_query: str):

    raqa_prompt = SystemRolePrompt(RAQA_PROMPT_TEMPLATE)
    user_prompt = UserRolePrompt(USER_PROMPT_TEMPLATE)

    context_list = vector_db_retriever.search_by_text(user_query, k=4)
    
    context_prompt = ""
    for context in context_list:
        context_prompt += context[0] + "\n"

    formatted_system_prompt = raqa_prompt.create_message(context=context_prompt)

    formatted_user_prompt = user_prompt.create_message(user_query=user_query)
    
    return formatted_system_prompt, formatted_user_prompt

@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():

    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):

    settings = cl.user_session.get("settings")

    client = AsyncOpenAI()

    # print(f"This is the message received by the user : {message.content}")

    # this the loading of the vector database
    vector_db = load_vector_db_from_local_file()

    formatted_system_prompt, formatted_user_prompt = list(
        get_formatted_prompts(
            vector_db_retriever=vector_db,
            user_query=message.content
            )
            )
    
    # print(f"formatted_system_prompt : {formatted_system_prompt}")
    # print(f"formatted_user_prompt : {formatted_user_prompt}")

    formatted_messages =[formatted_system_prompt, formatted_user_prompt]

    msg = cl.Message(content="")

    # Call OpenAI
    async for stream_resp in await client.chat.completions.create(
        messages=formatted_messages, stream=True, **settings
    ):
        token = stream_resp.choices[0].delta.content
        if not token:
            token = ""
        await msg.stream_token(token)

    # print(f"This is the message sent by the model : {msg.content}")

    # Send and close the message stream
    await msg.send()
