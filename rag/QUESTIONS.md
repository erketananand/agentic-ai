# RAG System Questions & Comparison

A comprehensive guide for testing and understanding the differences between Graph RAG, Semantic RAG, Lexical RAG, and Hybrid RAG implementations.

---

## Table of Contents
1. [Semantic RAG Questions](#semantic-rag-questions)
2. [Lexical RAG Questions](#lexical-rag-questions)
3. [Graph RAG Questions](#graph-rag-questions)
4. [Hybrid RAG Questions](#hybrid-rag-questions)
5. [Triple Hybrid RAG Questions](#triple-hybrid-rag-questions)
6. [Comparison Table](#comparison-table)
7. [Quick Decision Guide](#quick-decision-guide)
8. [Example: Same Question on All RAG Types](#example-same-question-on-all-rag-types)

---

## Semantic RAG Questions

**Best for:** Understanding design patterns, finding similar implementations, conceptual queries

- "How is user authentication implemented?"
- "Show me the permission checking logic"
- "What's the pattern for database operations?"
- "How do notifications get sent to users?"
- "Describe the error handling approach in this codebase"
- "How is configuration managed?"
- "What's the pattern for scheduler tasks?"
- "How does billing calculations work?"
- "Show me how audit logging is implemented"
- "What patterns are used for integrating external services?"
- "How is the task service structured?"
- "What's the approach to data validation?"
- "Explain the workflow for creating a new user"
- "How are background jobs scheduled and executed?"
- "What's the pattern for handling permissions?"

---

## Lexical RAG Questions

**Best for:** Finding specific functions, classes, API endpoints, exact implementations

- "What does the authenticate function do?"
- "Find the create_user function"
- "Show me the billing.py file"
- "What parameters does check_permission take?"
- "Where is the email_client used?"
- "Find all references to the database connection"
- "What's in the main.py file?"
- "Show me the TaskService class"
- "How is the Slack webhook called?"
- "Find the get_audit_log function"
- "Show me the notifications module"
- "Find the scheduler implementation"
- "What's in the models.py file?"
- "Where is the audit log created?"
- "Find the send_email function"
- "Show me the post_to_slack function"
- "What's the database connection string?"

---

## Graph RAG Questions

**Best for:** Understanding code dependencies, system architecture, impact analysis, multi-hop relationships

- "What does the auth module depend on?"
- "How does main.py orchestrate the entire system?"
- "What's the relationship between task_service and scheduler?"
- "How do notifications interact with integrations?"
- "What modules import from the database module?"
- "How are audit_log and permissions connected?"
- "What's the dependency chain for the billing module?"
- "Which modules are directly or indirectly used by notifications?"
- "How do email_client and slack_webhook integrate with the notification system?"
- "What are all the dependencies of the auth module (2 hops)?"
- "How does the permissions module connect to other systems?"
- "What's the impact of changes to the database module?"
- "How is the scheduler connected to task_service?"
- "What external integrations does the system use and where?"
- "Which modules depend on the models module?"
- "How does the reports module interact with other parts of the system?"
- "What's the relationship between billing and payments?"
- "How many layers deep is the dependency chain for main.py?"
- "Which modules are leaf nodes (no outgoing dependencies)?"
- "What's the dependency graph between notifications, email_client, and slack_webhook?"

---

## Hybrid RAG Questions

**Best for:** Complex queries mixing exact terms with conceptual understanding (Semantic + Lexical)

- "How does the auth module validate user credentials and what are the security checks?"
- "Explain the notification system including how it integrates with email_client"
- "What's the relationship between TaskService and the scheduler, and how do they coordinate?"
- "Show me how billing calculations work and what database queries are involved"
- "How does the permissions module enforce access control and where is it integrated?"
- "What are the main responsibilities of main.py and what modules does it orchestrate?"
- "Describe the audit_log implementation and how it tracks user actions"
- "How do the Slack and email integrations work together in the notifications system?"
- "What's the complete flow from user creation through permission assignment?"
- "How is error handling and logging implemented across the application?"
- "Explain how the task_service creates and manages tasks with scheduling"
- "How does the database module handle connections and what operations does it support?"
- "What's the relationship between reports and other modules?"
- "How are billing transactions recorded and audited?"
- "Show me how the scheduler triggers notifications through task_service"

---

## Triple Hybrid RAG Questions

**Best for:** Comprehensive analysis combining conceptual understanding, exact code locations, and architectural dependencies (Semantic + Lexical + Graph)

- "Explain the complete authentication flow including how it integrates with permissions, audit_log, and database"
- "How does the notification system work end-to-end with its dependencies on integrations and schedulers?"
- "Describe the task lifecycle from creation through scheduling, execution, and notification"
- "Show me the dependency chain for user creation including all modules involved and their interactions"
- "How do billing calculations work, including database dependencies, audit trails, and any API calls?"
- "What's the complete architecture showing how main.py orchestrates all system modules?"
- "Explain the relationship between email_client, slack_webhook, and the notification system with code examples"
- "How does the permissions system enforce access control across auth, audit_log, and other modules?"
- "Show the complete flow of a scheduled task from scheduler to task_service to notifications"
- "How are user actions tracked through audit_log and what modules trigger audit events?"
- "What are all the external integrations and how do they connect to the core system?"
- "Describe the database layer including connection handling, all modules that depend on it, and operations"
- "How do the reports module interact with other parts of the system and what data do they use?"
- "Show the complete dependency graph for the notifications module"
- "How do error handling, logging, and audit trails work together across the system?"
- "Explain the validation layers from user input through to database persistence"
- "What's the relationship between models, permissions, and database operations?"
- "How does the system handle concurrent operations and scheduling?"
- "Show me all places where integrations are called and their purpose in the workflow"
- "Describe a complete end-to-end scenario: user creates task → task gets scheduled → notification is sent"

---

## Comparison Table

| Aspect | Semantic RAG | Lexical RAG | Hybrid RAG | Triple Hybrid RAG | Graph RAG |
|---|---|---|---|---|---|
| **Primary Strength** | Conceptual understanding, semantic similarity | Fast keyword matching, exact locations | Balanced keyword + concept matching | **All three combined** | Dependency mapping, architecture |
| **Example Query** | "How is auth managed?" | "Show me authenticate()" | "How does authenticate() validate?" | "How does auth work with permissions and audit_log?" | "What does auth depend on?" |
| **Query Type** | "How/Why" questions | "What/Where" questions | Mixed complex queries | **Complex multi-faceted queries** | "How do X and Y relate?" |
| **Index Type** | Vector embeddings | BM25 keyword index | Both embeddings + keywords | **Embeddings + Keywords + Graph** | Knowledge graph (nodes & edges) |
| **Search Method** | Vector similarity | Term frequency-IDF | Ensemble + RRF ranking | **All three methods + deduplication** | Neighbourhood traversal |
| **Extraction Method** | Recursive chunking (500-1200 tokens) | AST or recursive chunks | Both chunking strategies | **Both + metadata extraction** | LLM relationship extraction |
| **Speed** | Medium (embedding lookup) | Fast (keyword match) | Medium | **Medium (all three parallel)** | Depends on graph size |
| **Memory Usage** | High (embeddings stored) | Low (BM25 only) | Medium (both indices) | **High (all three indices + graph)** | Low (graph only) |
| **Best For** | Design patterns, implementations | Specific functions, APIs, exact code | Complex technical queries | **End-to-end system analysis** | System architecture, dependencies |
| **Struggles With** | Exact code locations | Conceptual understanding | None (hybrid advantage) | **None (comprehensive approach)** | Understanding code logic |
| **Context Size** | ~4 chunks of 500 tokens | ~4 chunks of 1200 tokens | ~4 merged results | **~8 merged from three strategies** | Graph neighbourhood (2-3 hops) |
| **Scalability** | Good (vector DBs scale well) | Excellent (BM25 is lightweight) | Good (ensemble approach) | **Good (parallel retrieval)** | Good (graph traversal efficient) |
| **Relationship Discovery** | Limited (chunk-based) | Limited (chunk-based) | Limited (chunk-based) | **Enhanced (includes graph)** | **Native support** |
| **Multi-hop Analysis** | Not designed for it | Not designed for it | Not designed for it | **Supported via graph component** | **Native support** |
| **Example Output** | "The auth system validates credentials using..." | "authenticate() at auth.py:45" | "authenticate() validates using check_permission()..." | "auth.py:32 authenticate() → permissions.check_permission() → audit_log.log_action(); auth depends on: permissions, audit_log, database" | "auth → database, auth → permissions, auth → audit_log" |
| **LLM Model** | Groq Llama 3.1 8B | Groq Llama 3.1 8B | Groq Llama 3.1 8B | **Groq Llama 3.1 8B** | Groq Llama 3.1 8B (for extraction) |
| **Chunking Strategy** | Recursive text splitting | AST-based or recursive | Both available | **Both + chunk metadata** | N/A (full file processing) |
| **Response Type** | Explanation | Code snippet + location | Mixed answer + locations | **Complete analysis with all three perspectives** | Entity relationships |

---

## Quick Decision Guide

Choose the best RAG type based on your question:

| Your Question | Best RAG Type | Why |
|---|---|---|
| "Explain how X works" | **Semantic RAG** | Designed for conceptual understanding |
| "Find where function Y is defined" | **Lexical RAG** | Exact keyword matching finds code locations |
| "How does X function work and where is it called?" | **Hybrid RAG** | Combines exact matches with context |
| "What modules does X depend on?" | **Graph RAG** | Relationship-focused architecture analysis |
| "Show me the impact of changing module X" | **Graph RAG** | Traverses dependency graph |
| "Find implementations similar to pattern Z" | **Semantic RAG** | Vector similarity captures patterns |
| "List all calls to database.query()" | **Lexical RAG** | Term frequency finds exact matches |
| "How do auth, permissions, and audit_log work together?" | **Triple Hybrid RAG** | Best for multi-module end-to-end flows |
| "What's the architecture of the system?" | **Graph RAG** | Visualizes system topology |
| "Mixed: find code AND understand relationships" | **Hybrid RAG** | Balanced approach for dual concerns |
| "Which functions implement authentication?" | **Semantic RAG** | Semantic similarity finds related code |
| "Where is the send_email function?" | **Lexical RAG** | Direct keyword lookup |
| "How does the billing module calculate charges?" | **Semantic RAG** | Conceptual pattern matching |
| "What's the dependency chain from main to database?" | **Graph RAG** | Multi-hop path traversal |
| "Describe complete end-to-end workflow with code examples and architecture" | **Triple Hybrid RAG** | Comprehensive analysis across all three dimensions |
| "Show me how module X interacts with Y, Z, including code locations and dependencies" | **Triple Hybrid RAG** | Combines exact code, concepts, and relationships |

---

## Example: Same Question on All RAG Types

### Question: "How does the notification system work with the integrations?"

#### Semantic RAG Response
**Focus:** Conceptual understanding and patterns
```
The notification system sends alerts through email and Slack by calling 
integration handlers that format messages according to the target platform. 
It abstracts the notification logic from the delivery mechanism, allowing 
multiple integrations to be added without modifying core notification code.
```

#### Lexical RAG Response
**Focus:** Exact code locations and API calls
```
notifications.py:45-89 defines the notify() function which calls:
- email_client.send() at line 67
- slack_webhook.post() at line 75
Both integration modules are imported at the top of notifications.py
```

#### Graph RAG Response
**Focus:** System architecture and relationships
```
Knowledge Graph context:
  notifications -[IMPORTS]-> email_client
  notifications -[IMPORTS]-> slack_webhook
  notifications -[CALLS]-> email_client.send()
  notifications -[CALLS]-> slack_webhook.post()
  notifications -[DEPENDS_ON]-> integrations
  email_client -[DEFINES]-> EmailService
  slack_webhook -[DEFINES]-> SlackWebhook
```

#### Hybrid RAG Response
**Focus:** Combined code and concept understanding
```
The notifications.py module implements a notify() function that integrates 
with external services through email_client.send() (line 67) and 
slack_webhook.post() (line 75). This allows the system to send notifications 
through multiple channels while keeping notification logic centralized.
```

#### Triple Hybrid RAG Response
**Focus:** All three perspectives combined
```
SEMANTIC (Conceptual):
The notification system sends alerts through multiple channels by abstracting 
delivery mechanisms from core logic.

LEXICAL (Code Locations):
notifications.py:45-89 defines notify() which calls:
- email_client.send() at line 67
- slack_webhook.post() at line 75

GRAPH (Architecture):
Knowledge Graph:
  notifications -[IMPORTS]-> email_client, slack_webhook
  notifications -[CALLS]-> email_client.send(), slack_webhook.post()
  integrations -[CONTAINS]-> email_client, slack_webhook
```

---

### Question: "Where is user creation handled?"

#### Semantic RAG Response
**Focus:** Design pattern and implementation approach
```
User creation is typically handled through an authentication or user service 
that validates input, checks for duplicates, hashes passwords, stores the 
record in the database, and may trigger side effects like audit logging or 
welcome notifications.
```

#### Lexical RAG Response
**Focus:** Exact function and file locations
```
create_user() function is defined in auth.py at line 32
It's called from main.py at line 45
Related validation happens in permissions.py:check_permission() at line 12
```

#### Graph RAG Response
**Focus:** Dependencies and relationships involved
```
Knowledge Graph context:
  auth -[DEFINES]-> create_user()
  auth -[IMPORTS]-> models
  auth -[DEPENDS_ON]-> database
  auth -[DEPENDS_ON]-> audit_log
  auth -[DEPENDS_ON]-> permissions
  main -[CALLS]-> auth.create_user()
  create_user -[USES]-> database.save_user()
  create_user -[USES]-> audit_log.log_action()
```

#### Hybrid RAG Response
**Focus:** Complete workflow with code references
```
User creation is handled by the create_user() function in auth.py (line 32), 
which validates input against models.py schemas, stores the user in the database 
via database.py operations (line 50), logs the action in audit_log.py (line 55), 
and assigns default permissions through permissions.py (line 60).
```

#### Triple Hybrid RAG Response
**Focus:** Complete end-to-end analysis with all dimensions
```
SEMANTIC (Workflow):
User creation involves validation, storage, audit logging, and permission setup 
in a coordinated flow.

LEXICAL (Code References):
create_user() in auth.py:32 calls:
- models.User validation at line 40
- database.save_user() at line 50
- audit_log.log_action() at line 55
- permissions.assign_default() at line 60

GRAPH (Dependencies):
  auth -[DEPENDS_ON]-> [database, audit_log, permissions, models]
  auth.create_user -[CALLS]-> [database.save_user, audit_log.log_action, permissions.assign_default]
  main -[CALLS]-> auth.create_user
```

---

## Testing Strategy

### Step 1: Run Each RAG System
```bash
# Semantic RAG with RecursiveCharacterTextSplitter
python rag/1_semantic_rag/1_semantic_rag_using_recursive_text_splitter.py

# Semantic RAG with AST
python rag/1_semantic_rag/2_semantic_rag_using_ast.py

# Lexical RAG with BM25, RecursiveCharacterTextSplitter
python rag/2_lexical_rag/1_lexical_rag_using_bm25_recursive_text_splitter.py

# Lexical RAG with BM25, AST
python rag/2_lexical_rag/2_lexical_rag_using_bm25_ast.py

# Graph RAG (LLM-based extraction)
python rag/3_graph_rag/1_graph_rag_using_llm_graph_builder.py

# Graph RAG (Chunking-based, no LLM during indexing)
python rag/3_graph_rag/2_graph_rag_using_chucking_graph_builder.py

# Hybrid RAG (Semantic + Lexical)
python rag/4_hybrid_rag/1_hybrid_rag_with_semantic_lexical.py

# Triple Hybrid RAG (Semantic + Lexical + Graph)
python rag/4_hybrid_rag/2_hybrid_rag_with_semantic_lexical_graph.py
```

### Step 2: Ask the Same Question to Each
Try asking all 4 systems the same question and compare:
- Which gave the most useful answer?
- Which was fastest?
- Which revealed new insights?

### Step 3: Use the Decision Guide
For each new question, consult the Quick Decision Guide first to pick the best RAG system.

---

## Notes

- **Semantic RAG**: Uses HuggingFace embeddings for conceptual similarity (slow but accurate for patterns)
- **Lexical RAG**: Uses BM25 keyword indexing (fast but limited to exact terms and synonyms)
- **Graph RAG**: Extracts relationships and builds a knowledge graph (best for architecture understanding)
- **Hybrid RAG**: Combines semantic + lexical with Reciprocal Rank Fusion (balanced for complex queries)
- **Triple Hybrid RAG**: Combines semantic + lexical + graph with parallel retrieval and deduplication (best for comprehensive analysis)

All systems use **Groq's Llama 3.1 8B model** for efficiency while maintaining quality reasoning.
