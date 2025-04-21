from langchain.llms import LlamaCpp
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import requests
import json
import warnings

warnings.filterwarnings("ignore")

# Initialize your LLM
# llm = LlamaCpp(
#     model_path="models/llama3.2.gguf",
#     temperature=0.5,
#     n_ctx=4096,
#     verbose=True
# )
llm = Ollama(model="llama3.2",
             num_ctx = 4096,
             temperature=0,
             verbose=True)


# Vectorstore setup (assumed to be already configured)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# qdrant = Qdrant.from_existing_collection(
#     collection_name="sop_collection",
#     embeddings=embeddings,
#     # location="http://localhost:6333"
#     path="qdrant_data"
# )
collection_name="sop_collection"
persist_path="qdrant_data"
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
qdrant = Qdrant.from_existing_collection(
    collection_name=collection_name,
    path=persist_path,
    embedding=embedding
)

retriever = qdrant.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def search_sop_tool(query: str):
    print(", Custom: Query from Search SOP: ", query)
    output = qa_chain({"query": query})
    print("\n--- Retrieved SOP Answer ---\n")
    print(output["result"])
    return output["result"]
    # return qa_chain.run(query)

# def generate_bearer_token(_: str):
#     response = requests.post("https://reqres.in/api/login", json={
#         "username": "admin",
#         "password": "password123"
#     })
#     return response.json().get("token", "")

# def call_api(params: str):
#     p = json.loads(params)
#     response = requests.request(
#         method=p.get("method", "GET"),
#         url=p["url"],
#         headers=p.get("headers", {}),
#         json=p.get("body", {})
#     )
#     return response.text

# Register Tools

def combined_token_and_call(_: str):
    # Step 1: Get token
    token_resp = requests.post("https://reqres.in/api/login", json={
        "email": "eve.holt@reqres.in",
        "password": "cityslicka"
    })
    token = token_resp.json().get("token", "")

    # Step 2: Make an API call using the token
    api_resp = requests.get(
        "https://reqres.in/api/users/2",
        headers={"Authorization": f"Bearer {token}"}
    )
    return f"Token: {token}\nResponse: {api_resp.text}"

tools = [
    Tool(
        name="search_sop",
        func=search_sop_tool,
        description="Search SOPs for instructions"
    ),
    Tool(
        name="combined_token_and_call",
        func=combined_token_and_call,
        description="Generate bearer token for authentication then make the API calls"
    )
]

# Memory for reasoning steps
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize the ReAct Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION ,
    verbose=True,
    memory=memory
)

# Sample ticket resolution
if __name__ == "__main__":
    #   ###### Gathers the SOPs well enough. ####### #

    # ticket_text = "Terminate employee with empID 11223 as the employee has not been performing."
    # result = agent.run(f"""
    # You are an IT automation agent. Your job is to resolve the following request step-by-step.

    # Ticket: {ticket_text}

    # Think through the steps and use available tools to complete the task.
    # """)
    # print("\nFinal Response:\n", result)

# Working but hellucinating - repeating itself.
    # ticket_text = "Perform all the necessary steps to Terminate employee with empID 11223 as the employee has not been performing."
    # result = agent.run(f"""
    # You are an IT automation agent. Your job is to resolve the following ticket step-by-step by following below strategy:
    # 1. First, use `search_sop` to get any necessary procedures.
    # 2. Then, if the SOP provides API details, use `auth_then_call` to perform the action.
    # 3.  Once the action is complete, respond with the result.
    # 4. Do not repeat the search unless it failed.
    
    # Ticket: {ticket_text}                   
    # """)
    # print("\nFinal Response:\n", result)

    ticket_text = "Perform all the necessary steps to Terminate employee with empID 11223 as the employee has not been performing."
    result = agent.run(f"""
    You are an intelligent assistant that transforms Standard Operating Procedures (SOPs) and performs the actions to achieve the task.
        
    query: {ticket_text}                   
    """)
    print("\nFinal Response:\n", result)

                
       