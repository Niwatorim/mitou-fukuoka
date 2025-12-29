"""
get graph
embed content
retrieval engine
LLM question
"""

"""
___________________________________________________________________________________
│mainLabel │SampleNode                                                            │
╞══════════╪══════════════════════════════════════════════════════════════════════╡
│"File"    │(:File {name: "App.jsx"})                                             │
├──────────┼──────────────────────────────────────────────────────────────────────┤
│"Frontend"│(:Frontend {name: "div",properties: []})                              │
├──────────┼──────────────────────────────────────────────────────────────────────┤
│"Function"│(:Function {name: "App",type: "function_declaration",params: "()"})   │
├──────────┼──────────────────────────────────────────────────────────────────────┤
│"Import"  │(:Import {name: "'react'",source: "'react'",import_items: []})        │
├──────────┼──────────────────────────────────────────────────────────────────────┤
│"Variable"│(:Variable {value_type: "call_expression",name: "count",type: "array_d│
│          │estructure",value: "useState(0)"})                                    │
└──────────┴──────────────────────────────────────────────────────────────────────┘



"""


from tree_sitter import Language, Parser, Query, QueryCursor, Node
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, ChatGoogle,Browser

from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv
import streamlit as st



graph = Neo4jGraph()
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from neo4j import GraphDatabase
import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


#create vector model
EMBED = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

#function for queries:
def query(question:str):
    """user input
    returns: embedding vector
    """
    return EMBED.embed_query(question)

#initial load -> embed everything
vector_index = Neo4jVector.from_existing_graph(
    EMBED,
    search_type="hybrid",
    node_label="n",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)


#use vector thingy to retrieve vectors
vector_retriever = vector_index.as_retriever()


driver=GraphDatabase.driver(
    uri=os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"],
            os.environ["NEO4J_PASSWORD"])
)

def create_fulltext_index(tx):
    query="""
    CREATE FULLTEXT INDEX `fulltext_entity_id`
    FOR (n:_Entity__)
    ON EACH [n.id];
    """
    tx.run(query)

#to execute query
def create_index():
    with driver.session() as session:
        session.execute_write(create_fulltext_index)
        print("Fulltext index created successfully.")
try:
    create_index()
except Exception as e:
    print(f"error: {e}")
driver.close()



class Entities(BaseModel):
    """Identifying information about entities"""

    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that"
        "appear in the text",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entitiesfrom the text."
        ),
        (
            "human",
            "Use the given format to extract information from the following"
            "input: {question}",
        )
    ]
)

entity_chain=llm





"""
graph = Neo4jGraph()
graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
)

vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)
vector_retriever = vector_index.as_retriever()

driver = GraphDatabase.driver(
        uri = os.environ["NEO4J_URI"],
        auth = (os.environ["NEO4J_USERNAME"],
                os.environ["NEO4J_PASSWORD"]))

def create_fulltext_index(tx):
    query = '''
    CREATE FULLTEXT INDEX `fulltext_entity_id` 
    FOR (n:__Entity__) 
    ON EACH [n.id];
    '''
    tx.run(query)

# Function to execute the query
def create_index():
    with driver.session() as session:
        session.execute_write(create_fulltext_index)
        print("Fulltext index created successfully.")

# Call the function to create the index
try:
    create_index()
except:
    pass

# Close the driver connection
driver.close()


class Entities(BaseModel):
    ""Identifying information about entities.""

    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)


entity_chain = llm.with_structured_output(Entities)

entity_chain.invoke("Who are Nonna Lucia and Giovanni Caruso?")

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    print(f"Generated Query: {full_text_query}")
    return full_text_query.strip()


# Fulltext index query
def graph_retriever(question: str) -> str:
    ""
    Collects the neighborhood of entities mentioned
    in the question
    ""
    result = ""
    entities = entity_chain.invoke(question)
    for entity in entities.names:
        response = graph.query(
            ""CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            "",
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    return result
print(graph_retriever("Who is Nonna Lucia?"))

def full_retriever(question: str):
    graph_data = graph_retriever(question)
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]
    final_data = f""Graph data:
{graph_data}
vector data:
{"#Document ". join(vector_data)}
    ""
    return final_data

    

template = ""Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:""
prompt = ChatPromptTemplate.from_template(template)

chain = (
        {
            "context": full_retriever,
            "question": RunnablePassthrough(),
        }
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke(input="Who is Nonna Lucia? Did she teach anyone about restaurants or cooking?")
"""