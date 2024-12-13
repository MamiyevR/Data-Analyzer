# Q2 Causal Relationship Visualizer

### Main libraries Used

- `langchain`: We extensively use langchain to connect llm with graph database to build causal graph
- `neo4j`: To store graph
- `pyvis`: Visualization

## Implementation

### 1. Data Preparation

The input data is read from a JSON file and split into manageable chunks for processing. This is done using the `build_graph.py` script.

### 2. Building the Graph

The `build_graph.py` script also handles the creation of the graph in Neo4j. It embeds the chunks using OpenAI embeddings and adds nodes and relationships to the graph.

More specifically, we embedd each chunk of text and add it to the graph using cypher:

```cypher
MERGE (d:Title {id: $title})
MERGE (c:Chunk {id: $chunk_id})
SET c.text = $text
MERGE (d)<-[:PART_OF]-(c)
WITH c
CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
```

Next, using LLMGraphTransformer and GPT4 model to generate a set of graph docs of nodes and relationships.

And finally, set of nodes and relationships are added to graph.

This code is heavily inspired from [here](https://graphacademy.neo4j.com/courses/llm-knowledge-graph-construction/3-python-create-graph/2-graph-builder-process/)

### 3. Visualizing the Graph

The `visualizer.py` script provides a visual interface to explore the causal graph. It uses the PyVis library to create an interactive visualization of the graph.

For better visualization, please go to neo4j aura instance where data will be generated once you run build_graph. Run following cypher to populate interactive graph:

```cypher
MATCH (n)-[r]->(m)
RETURN n, r, m
```

## Conclusion

This system effectively visualizes causal relationships between financial knowledge using Neo4j as the graph database. The implementation includes data preparation, graph building, and visualization components, providing a comprehensive solution for exploring causal relationships in financial data.
