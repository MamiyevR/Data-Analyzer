import os
from langchain_community.graphs.neo4j_graph import Neo4jGraph
from pyvis.network import Network
from dotenv import load_dotenv

load_dotenv()


def visualize_graph(file_name: str):
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
    )

    # Sample Cypher query to retrieve nodes and relationships
    cypher_query = """
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        LIMIT 1
    """

    with graph._driver.session() as session:
        results = session.run(cypher_query)

        net = Network(
            cdn_resources="remote", directed=True, height="500px", width="100%"
        )

        # Process the results
        for result in results:
            node_1 = result["n"]
            node_2 = result["m"]
            relationship = result["r"]

            print(relationship)
            # Add the nodes and relationships to the network
            net.add_node(node_1.element_id)
            net.add_node(node_2.element_id)
            net.add_edge(node_1.element_id, node_2.element_id)

    # save html format
    net.show(file_name, notebook=False)


visualize_graph("../data/graph.html")
