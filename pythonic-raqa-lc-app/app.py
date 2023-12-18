from operator import itemgetter
import chainlit as cl
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from utils import ArxivLoader, PineconeIndexer

system_template = """
Use the provided context to answer the user's query.

You may not answer the user's query unless there is specific context in the following text.

If you do not know the answer, or cannot answer, please respond with "I don't know".

Context:
{context}
"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]

prompt = ChatPromptTemplate(messages=messages)
chain_type_kwargs = {"prompt": prompt}

@cl.author_rename
def rename(orig_author: str):
    rename_dict = {"RetrievalQA": "Learning about Nuclear Fission"}
    return rename_dict.get(orig_author, orig_author)

@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():

    msg = cl.Message(content=f"Initializing the Application...")
    await msg.send()

    # load documents from Arxiv
    axloader = ArxivLoader()
    axloader.main()

    # load embedder and the retriever
    pi = PineconeIndexer()
    pi.load_embedder()
    retriever=pi.get_vectorstore().as_retriever()
    print(pi.index.describe_index_stats())

    # build llm
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    msg = cl.Message(content=f"Application is ready !")
    await msg.send()

    cl.user_session.set("llm", llm)
    cl.user_session.set("retriever", retriever)

@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):

    llm = cl.user_session.get("llm")
    retriever = cl.user_session.get("retriever")

    retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | retriever,
        "question": itemgetter("question")
        }
        | RunnablePassthrough.assign(
            context=itemgetter("context")
        )
        | {
            "response": prompt  | llm,
            "context": itemgetter("context"),
        }
    )

    answer = retrieval_augmented_qa_chain.invoke({"question" : message.content})
    
    await cl.Message(content=answer["response"].content).send()





