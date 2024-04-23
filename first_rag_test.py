from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAI

from friends_vector_store import FriendsVectorStore
from script_document_loader import ScriptDocumentLoader

llm_client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
)


retriever = FriendsVectorStore().as_retriever()


# prompt_template = hub.pull("rlm/rag-prompt-mistral") # {context}, {question}
prompt_template = hub.pull("athroniaeth/rag-prompt-mistral-custom-2") # {context}, {input}

# prompt = """
# Context information is below.
# ---------------------
# {context}
# ---------------------
# Given the context information and not prior knowledge, answer the query.
# Query: {question}
# Answer:
# """

# prompt_string = """Answer the following question based only on the provided context:
# <context>
# {context}
# </context>
# Question: {question}
# Answer:
# """

# prompt_template = PromptTemplate.from_template(prompt_string)


def input_question():
    rag_chain = prompt() | llm_client | StrOutputParser()
    question = input("## Enter your question: ")
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n\n")
    print("## finished invoking the rag_chain")


def prompt():
    context = retriever | ScriptDocumentLoader.documents_as_context
    return {"context": context, "input": RunnablePassthrough()} | prompt_template


while True:
    input_question()
