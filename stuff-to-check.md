# Learning GraphRAG with Neo4j: A Comprehensive Guide

Let me teach you GraphRAG step by step with practical examples. I'll build from basics to advanced patterns.

## Part 1: Understanding the Foundation

GraphRAG isn't just about querying a graph - it's about **combining vector similarity with graph structure** to get better context. Think of it like this:

- **Vector search alone**: "Find code that looks similar to this"
- **Graph traversal alone**: "Find code connected to this"
- **GraphRAG**: "Find similar code AND everything connected to it"

## Part 2: Basic Hybrid Search Pattern

```python
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools import tool
from langchain_core.documents import Document

# Initialize
graph = Neo4jGraph()
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key
)

# LESSON 1: Vector Search on Graph Nodes
# This creates a vector index on your nodes so you can do semantic search
vector_store = Neo4jVector.from_existing_graph(
    embeddings,
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="component_embeddings",  # Name of the vector index
    node_label="Component",  # Which nodes to index
    text_node_properties=["name", "code", "description"],  # What text to embed
    embedding_node_property="embedding",  # Where to store the embedding
)

# Now you can do semantic search!
def semantic_search(query: str, k: int = 3):
    """
    TEACHING POINT: This finds nodes that are semantically similar to your query
    Example: "authentication logic" might find LoginComponent, AuthService, etc.
    """
    results = vector_store.similarity_search(query, k=k)
    CONSOLE.print(f"[cyan]Found {len(results)} similar components[/cyan]")
    return results
```

## Part 3: Expanding Context with Graph Traversal

```python
@tool
def graphrag_retrieval(query: str) -> str:
    """
    TEACHING POINT: The core GraphRAG pattern
    1. Find relevant nodes via vector similarity
    2. Expand to connected nodes via graph traversal
    3. Return enriched context
    """
    
    # Step 1: Vector search to find starting nodes
    similar_docs = vector_store.similarity_search(query, k=2)
    
    if not similar_docs:
        return "No relevant components found"
    
    # Extract node IDs from the metadata
    node_ids = [doc.metadata.get("id") or doc.metadata.get("name") for doc in similar_docs]
    
    # Step 2: Graph traversal to get context
    # This Cypher query finds the node AND its neighborhood
    cypher_query = """
    MATCH (n:Component)
    WHERE n.name IN $node_names
    
    // Get the node itself
    WITH n
    
    // Get connected components (1-hop neighbors)
    OPTIONAL MATCH (n)-[r]-(connected:Component)
    
    // Return everything
    RETURN n.name as component,
           n.code as code,
           n.description as description,
           collect(DISTINCT {
               type: type(r),
               direction: CASE 
                   WHEN startNode(r) = n THEN 'outgoing'
                   ELSE 'incoming'
               END,
               connected: connected.name,
               connected_description: connected.description
           }) as connections
    LIMIT 5
    """
    
    results = graph.query(cypher_query, params={"node_names": node_ids})
    
    # Step 3: Format the enriched context
    context = []
    for record in results:
        comp_context = f"Component: {record['component']}\n"
        comp_context += f"Description: {record['description']}\n"
        comp_context += f"Code:\n{record['code']}\n"
        
        if record['connections']:
            comp_context += "\nConnections:\n"
            for conn in record['connections']:
                if conn['connected']:  # Filter out null connections
                    comp_context += f"  - {conn['direction']} {conn['type']} {conn['connected']}\n"
        
        context.append(comp_context)
    
    return "\n\n---\n\n".join(context)


# Example usage
result = graphrag_retrieval("user authentication flow")
CONSOLE.print(Panel(result, title="GraphRAG Context", style="green"))
```

## Part 4: Multi-Hop Reasoning

```python
@tool
def deep_context_retrieval(component_name: str, depth: int = 2) -> str:
    """
    TEACHING POINT: Multi-hop traversal
    Sometimes you need to go deeper - find dependencies of dependencies
    
    Use cases:
    - "What's the full call chain for this function?"
    - "What are all transitive dependencies?"
    - "What components could be affected by changes here?"
    """
    
    cypher_query = """
    MATCH path = (start:Component {name: $component_name})-[*1..$depth]-(connected)
    WHERE connected:Component OR connected:Function OR connected:Variable
    
    WITH start, connected, path,
         length(path) as hop_distance,
         [r in relationships(path) | type(r)] as relationship_chain
    
    RETURN DISTINCT
        connected.name as related_component,
        labels(connected) as node_types,
        hop_distance,
        relationship_chain,
        connected.description as description
    ORDER BY hop_distance, related_component
    LIMIT 20
    """
    
    results = graph.query(cypher_query, params={
        "component_name": component_name,
        "depth": depth
    })
    
    # Organize by distance
    context_by_hop = {}
    for record in results:
        hop = record['hop_distance']
        if hop not in context_by_hop:
            context_by_hop[hop] = []
        
        context_by_hop[hop].append({
            'name': record['related_component'],
            'types': record['node_types'],
            'path': ' -> '.join(record['relationship_chain']),
            'description': record['description']
        })
    
    # Format output
    output = f"Deep context for: {component_name}\n\n"
    for hop, items in sorted(context_by_hop.items()):
        output += f"{'='*50}\n"
        output += f"Distance: {hop} hop{'s' if hop > 1 else ''}\n"
        output += f"{'='*50}\n"
        for item in items:
            output += f"  • {item['name']} ({', '.join(item['types'])})\n"
            output += f"    Path: {item['path']}\n"
            if item['description']:
                output += f"    Description: {item['description']}\n"
        output += "\n"
    
    return output
```

## Part 5: Pattern-Based Retrieval

```python
@tool
def find_similar_patterns(component_name: str) -> str:
    """
    TEACHING POINT: Graph pattern matching
    Find components with similar structural patterns
    
    Example: "Find all components that follow the same architectural pattern"
    - Same number of dependencies
    - Same types of relationships
    - Similar graph neighborhood structure
    """
    
    cypher_query = """
    // First, get the pattern of the source component
    MATCH (source:Component {name: $component_name})
    OPTIONAL MATCH (source)-[r]->(dep)
    WITH source, 
         count(dep) as dep_count,
         collect(DISTINCT labels(dep)) as dep_types,
         collect(DISTINCT type(r)) as rel_types
    
    // Now find components with similar patterns
    MATCH (candidate:Component)
    WHERE candidate.name <> source.name
    OPTIONAL MATCH (candidate)-[r2]->(dep2)
    WITH source, candidate,
         dep_count,
         count(dep2) as candidate_dep_count,
         collect(DISTINCT labels(dep2)) as candidate_dep_types,
         collect(DISTINCT type(r2)) as candidate_rel_types
    
    // Calculate similarity (simple version)
    WITH source, candidate,
         dep_count, candidate_dep_count,
         CASE 
             WHEN dep_count = 0 THEN 0
             ELSE abs(dep_count - candidate_dep_count) * 1.0 / dep_count
         END as structure_diff
    
    WHERE structure_diff < 0.5  // Less than 50% difference
    
    RETURN candidate.name as similar_component,
           candidate.description as description,
           candidate_dep_count as dependencies,
           structure_diff
    ORDER BY structure_diff
    LIMIT 10
    """
    
    results = graph.query(cypher_query, params={"component_name": component_name})
    
    output = f"Components with similar patterns to {component_name}:\n\n"
    for record in results:
        output += f"• {record['similar_component']}\n"
        output += f"  Dependencies: {record['dependencies']}\n"
        output += f"  Similarity: {(1 - record['structure_diff']) * 100:.1f}%\n"
        if record['description']:
            output += f"  Description: {record['description']}\n"
        output += "\n"
    
    return output
```

## Part 6: Subgraph Extraction for Complex Queries

```python
@tool
def extract_test_relevant_subgraph(component_name: str) -> str:
    """
    TEACHING POINT: Subgraph extraction
    For testing, you want ALL relevant context - not just nearby nodes
    
    This extracts:
    - The component itself
    - All functions it contains
    - All components it depends on
    - All data it uses
    - All events it emits/listens to
    """
    
    cypher_query = """
    MATCH (component:Component {name: $component_name})
    
    // Get all functions in this component
    OPTIONAL MATCH (component)-[:CONTAINS]->(func:Function)
    
    // Get all dependencies
    OPTIONAL MATCH (component)-[:IMPORTS|USES]->(dep:Component)
    
    // Get all data/state
    OPTIONAL MATCH (component)-[:USES_STATE|READS|WRITES]->(data:Variable)
    
    // Get all props/inputs
    OPTIONAL MATCH (component)-[:RECEIVES]->(prop:Prop)
    
    // Get events
    OPTIONAL MATCH (component)-[:EMITS]->(event:Event)
    OPTIONAL MATCH (component)-[:LISTENS_TO]->(listened_event:Event)
    
    RETURN component.name as name,
           component.code as code,
           component.testableAttributes as testable_attrs,
           collect(DISTINCT func.name) as functions,
           collect(DISTINCT dep.name) as dependencies,
           collect(DISTINCT data.name) as state_vars,
           collect(DISTINCT prop.name) as props,
           collect(DISTINCT event.name) as emitted_events,
           collect(DISTINCT listened_event.name) as consumed_events
    """
    
    results = graph.query(cypher_query, params={"component_name": component_name})
    
    if not results:
        return f"Component {component_name} not found"
    
    record = results[0]
    
    # Format as rich context for test generation
    context = f"""
Component: {record['name']}

TESTABLE ATTRIBUTES:
{record['testable_attrs'] if record['testable_attrs'] else 'None specified'}

FUNCTIONS TO TEST:
{chr(10).join(f'  - {f}' for f in record['functions'] if f) or '  None'}

DEPENDENCIES (need to mock/stub):
{chr(10).join(f'  - {d}' for d in record['dependencies'] if d) or '  None'}

STATE VARIABLES (need to test state changes):
{chr(10).join(f'  - {s}' for s in record['state_vars'] if s) or '  None'}

PROPS/INPUTS (need to test with different values):
{chr(10).join(f'  - {p}' for p in record['props'] if p) or '  None'}

EVENTS EMITTED (need to verify):
{chr(10).join(f'  - {e}' for e in record['emitted_events'] if e) or '  None'}

EVENTS CONSUMED (need to trigger):
{chr(10).join(f'  - {e}' for e in record['consumed_events'] if e) or '  None'}

CODE:
{record['code']}
"""
    
    return context
```

## Part 7: Putting It All Together in an Agent

```python
# Create an enhanced agent with all GraphRAG tools
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key
)

graphrag_system_prompt = f"""
You are an expert code analyst with access to a knowledge graph of a codebase.

You have access to multiple retrieval strategies:
1. graphrag_retrieval: Hybrid search (semantic + graph structure)
2. deep_context_retrieval: Multi-hop traversal for deep dependencies
3. find_similar_patterns: Find architecturally similar components
4. extract_test_relevant_subgraph: Get comprehensive test context

WHEN TO USE EACH TOOL:
- Use graphrag_retrieval for: "Find components related to X", "What handles Y?"
- Use deep_context_retrieval for: "What are all dependencies?", "Full call chain?"
- Use find_similar_patterns for: "Similar components?", "Same architecture?"
- Use extract_test_relevant_subgraph for: "Generate tests for X"

Graph Schema:
{graph.schema}

Always explain your reasoning and which tool you chose.
"""

agent = create_agent(
    llm,
    tools=[
        graphrag_retrieval,
        deep_context_retrieval,
        find_similar_patterns,
        extract_test_relevant_subgraph,
        query_graph  # Keep the basic query tool too
    ],
    system_prompt=graphrag_system_prompt,
    middleware=[handle_errors]
)

# Example queries that showcase different strategies
test_queries = [
    "What components are involved in user authentication? I need the full picture.",
    "Find components similar to LoginForm in architecture",
    "I need to write tests for the ShoppingCart component, give me everything I need to know",
    "What are all the dependencies of the PaymentProcessor, including indirect ones?"
]

for query in test_queries:
    CONSOLE.print(f"\n{'='*70}")
    CONSOLE.print(f"[bold cyan]Query: {query}[/bold cyan]")
    CONSOLE.print(f"{'='*70}\n")
    
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": query
        }]
    })
    
    CONSOLE.print(Panel(
        result["messages"][-1].content,
        title="GraphRAG Result",
        style="green"
    ))
```

## Key Takeaways

**What makes this GraphRAG?**
1. ✅ **Hybrid search**: Vector similarity + graph structure
2. ✅ **Context expansion**: Start with similar nodes, expand via relationships
3. ✅ **Relationship-aware**: Understands how entities connect
4. ✅ **Multi-hop reasoning**: Can traverse deep dependency chains
5. ✅ **Pattern matching**: Finds structural similarities

**For your test generation use case:**
- Use `extract_test_relevant_subgraph` to get comprehensive component context
- Use `deep_context_retrieval` to understand all dependencies you need to mock
- Use `graphrag_retrieval` when users ask vague questions like "test the auth flow"
- Use `find_similar_patterns` to reuse test patterns from similar components

The power of GraphRAG is that you get **contextually relevant** information, not just **similar** information. You understand not just what a component does, but what it connects to, what depends on it, and how it fits in the larger system.
