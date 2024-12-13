import os
import json
import torch
from prompt import default_prompt
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from dotenv import load_dotenv

load_dotenv()

device = 0 if torch.cuda.is_available() else -1


def read_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def split_into_chunks(data, chunk_size=1500, chunk_overlap=200):
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    documents = [item["content"] for item in data]
    metadatas = [{"title": item["title"]} for item in data]

    chunks = text_splitter.create_documents(documents, metadatas)

    return chunks


def build_graph(chunks, model_name, embedding_model):
    embedding = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"), model=embedding_model
    )

    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name=model_name)

    doc_transformer = LLMGraphTransformer(llm=llm, prompt=default_prompt)

    try:
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
        )
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return

    # drop existing graph
    with graph._driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

    title = chunks[0].metadata["title"]
    i = 0
    for chunk in chunks:
        if title != chunk.metadata["title"]:
            title = chunk.metadata["title"]
            i = 0
        else:
            i += 1

        # create unique id for the chunk
        chunk_id = chunk.metadata["title"] + "_" + str(i)

        # embed the chunk
        chunk_embedding = embedding.embed_query(chunk.page_content)

        # Add the Document and Chunk nodes to the graph
        properties = {
            "title": title,
            "chunk_id": chunk_id,
            "text": chunk.page_content,
            "embedding": chunk_embedding,
        }

        graph.query(
            """
            MERGE (d:Title {id: $title})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text
            MERGE (d)<-[:PART_OF]-(c)
            WITH c
            CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
            """,
            properties,
        )

        # Create entities and relationships
        graph_docs = doc_transformer.convert_to_graph_documents([chunk])

        # Map entities in graph to chunk nodes
        for doc in graph_docs:
            chunk_node = Node(id=chunk_id, type="Chunk")

            for node in doc.nodes:
                doc.relationships.append(
                    Relationship(source=chunk_node, target=node, type="HAS_ENTITY")
                )

        # add documents to graph
        graph.add_graph_documents(graph_docs)

    # Create the vector index
    graph.query(
        """
        CREATE VECTOR INDEX `vector`
        FOR (c: Chunk) ON (c.embedding)
        OPTIONS {indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
        }};
        """
    )

    return graph


def main():
    # Check environment variables
    required_env_vars = [
        "OPENAI_API_KEY",
        "NEO4J_URI",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
    ]
    for var in required_env_vars:
        if not os.getenv(var):
            raise EnvironmentError(f"Missing required environment variable: {var}")

    # Read the data
    data = read_json("../data/output.json")

    # Split the data into chunks
    chunks = split_into_chunks(data)

    model_name = "gpt-4o"
    embedding_model = "text-embedding-3-large"

    # Vectorize the chunks and build the graph
    graph = build_graph(chunks, model_name, embedding_model)


# run the main function
main()
