# RAG architecture overview

First, I will lay out a diagram of my architecture, and each part of the pipeline will be detailed below (I thank ChatGPT for the drawing!).
```
User Query
   │
   ▼
Intent Classifier / Router (LLM + Rules)
   │  └─> JSON routing info: {intent, sources, mode, entities, confidence}
   ▼
Retrieval Planner (Deterministic / Rule-based)
   │  └─> Retrieval plan: {indices, top_k, retrieval mode, query expansions}
   ▼
Retrieval Executor / Hybrid Retriever
   ├─ Vector Store (semantic embeddings, namespaces: docs, GitHub, Slack)
   ├─ BM25 Index (tokenized text, same chunk IDs)
   └─ Metadata DB (authoritative info: timestamps, IDs, PR/issue state)
   │
   ▼
Optional Reranker LLM (3–6 most relevant chunks)
   │
   ▼
Synthesizer / Generation LLM
   ├─ Generates grounded response
   ├─ Provides sources / URLs
   ├─ Flags uncertainty
   └─ Can call Metadata DB for authoritative info
   │
   ▼
Final Response → User (with evidence, provenance)
```

## Step 1: Data Ingestion

Our setup consists of three databases, updated via our data ingestion logic:
 
1.	Vector Store for Embeddings + Lightweight Metadata (e.g. Pinecone-based): stores semantic embeddings, supports semantic search \ retrieval using text embeddings. Performance is tuned for ANN vector search. To support source-oriented retrieval, partition vector store by namespace (docs, github, slack) and keep rich metadata.
2.	Index Metadata DB for authoritative, ground-truth registry of all chunks in vector store (e.g. small Redis store).
3.	BM25 index for keyword search (e.g. ElasticSearch)
 
 
- TRADEOFF ALERT - Using namespace partitioning for different sources is easier to maintain and allows for source-targeted queries. The alternative is multiple stores, each maintained by a separate engine, which would make sense if there are great discrepancies in scale / compliance requirements for each source. I would opt for ease of maintenance here (unless we know something substantial about scale relations between sources).
 
### When are vector store and BM25 index updated?
 
We perform asynchronous update via ingestion workers that run on schedule or event-driven triggers, such as:
1.	GitHub webhook update (e.g., newly performed merge)
2.	Slack webhook update / scheduled web API call (e.g., new message posted in channel)
3.	Watch / Fetch relevant markdown files (e.g. monitor repo, folder, docs site, pull updated or new .md files, etc.)
 
  
### Preprocessing
 
- Documentation (markdown): remove formatting characters such as "---", table borders, whitespace, but preserve ## Headings. Remove front matter if exists.
- GitHub: normalize text by stripping Markdown only in PRs, remove signatures, identify labels and assignees, extract structured fields, but keep relevant structured metadata such as issue number, PR number, state, repo…
- Slack: remove Slack formatting. According to Slack API: remove markers like <@user>, emoji, attachments. Also, flatten threaded messages for context.
 
### Chunking

- Documentation (markdown): split by semantic units relevant for .md, like headings, sections, or some token threshold that makes sense (default here for example is 1600 tokens). Optionally split at headings (#, ##..), and if a section is too large, do sliding window chunking with a smaller token size. Important to keep code blocks intact. Could also be done more automatically with Llamaindex (source is the comments here: https://www.reddit.com/r/Rag/comments/1k9m4vs/advice_needed_best_way_to_chunk_markdown_from_a/). Store chunk metadata: URI, last modified timestamp, author…
- GitHub: split long issues / PRs into chunks of 200-500 tokens (source for threshold: GPT5 + Gemini). Extract code snippets as separate chunks. Alternatively, chunk by comments /thread turns, and join too-short comments into conversation windows with sliding chunking. Store chunk metadata: source type (issue, pr, comment), repo name, issue/PR number, created_at…
- Slack: probably good to chunk by thread or else we lose context across chunks. Alternatively, use timestamp-based chunking, since Slack conversations are in bursts. Within each thread / timestamp window, chunk into 500 tokens per chunk. Store chunk metadata: channel name, thread ID, timestamp, author.. See MLOps section for comment about privacy issues with Slack data.
 
- TRADEOFF ALERT - larger chunks are better for retaining context within each vector, but you might lose the granularity of the information if you are trying to embed larger amounts of text in a single fixed-size vector. I believe that the modular chunking schemes that take into account the data source account for this nicely.
 
### Embedding
 
Generate embeddings per chunk with off-the-shelf embedding model (e.g. OpenAI), and update embedding into vector store under namespace corresponding to source.
 
 
### Metadata DB Update
 
Apart from the light metadata we maintain inside each chunk embedding, we need to record authoritative chunk info in the metadata DB: canonical chunk ID, timestamp, IDs of messages / repos, pointers to vector store embeddings…
Note that this DB is always up-to-date (the authoritative ground truth DB for the chunks), used for status checks, filtering by metadata such as date, source or author, etc.
 
### BM25 Index Update
 
BM25 receives the same "logical" chunk as the vector store (same ID, same boundaries in original file, same metadata, a chunk has one "identity" across both stores), but preprocessing is different: tokenized lexical text, in lowercase,  strip punctuation, because BM25 is exact-word match, like a search engine, see https://github.com/ev2900/BM25_Search_Example). While embeddings care about semantic meaning, BM25 just needs to have the literal keyword tokens to be searched.
 
TRADEOFF ALERT - it might be overkill to combine BM25 (lexical retrieval) with the semantic vector retrieval done on the embeddings. Semantic search only can miss exact-match keyword information (and this seems to be important in our usecase, which utilizes many exact keywords like IDs, error codes, APIs, version numbers…). This is guaranteed to boost recall, but we might pay in the added overhead of having to update and maintain another store. I would opt to use it, because of the developer-query usecase which often needs keyword search.
 
## Step 2: Retrieval - Intent Classification, Routing Data
 
 
We design for two capabilities:
1.	Targeted retrieval for specific question types (as defined: factual lookup, status check).
2.	Broad retrieval + synthesis for cross-source questions.
 
Thus, our RAG system needs to first classify which question type the user query intended, and also suggest preferred sources. The system will use a double-faceted classification mechanism:

1. First, use fast rule-based heuristics (regex) to identify obvious cases of status checks. For example, search for regex expressions containing "issue #\d+", "PR…", "status", "open/closed"…
2.	Fallback to an LLM intent classifier with a small prompt that classifies into DOCS_LOOKUP, GITHUB_STATUS, SYNTHESIS, UNKNOWN, and also yields confidence score. Specifically, the intent classifier will return the following routing info, JSON-formatted:
   - Intent type (the above labels)
   - Data sources to query 
   - Retrieval mode = semantic search, exact ID lookup, hybrid between the two 
   - Query "transformations" = relevant entities extracted by the LLM intent classifier relevant for the query, see below example 
   - Confidence 

For example, a user query like "What is the signature of getUserProfile?" might get the following routing output:
```
{
  "intent": "factual_lookup",
  "sources": ["docs"],
  "retreival_mode": "semantic",
  "extracted_entities": {
    "function_name": "getUserProfile"
  },
  "confidence": 0.92
}
```
 
Later downstream, this combination of intent and sources will tell the model to only retrieve on documentation and markdown, the retrieval mode tells it to run semantic (vector) and keyword (BM25) search, and the function name entity needs to be included as a cue in the search.
 
In the case of synthesis, the LLM should return more than one source, the retrieval method should be "hybrid", with an extracted entity relevant for the topic at hand.
 
 
 
 
## Step 3: Retrieval Planning, Retrieval Algorithm
 
Downstream retrieval pipeline uses the JSON routing data, using the first part of the retrieval layer: the retrieval planner. This component outputs a (machine-executable) retrieval plan, which is an informed JSON summary of the routing data based on the confidence output of the previous model. The following fields are included in the retrieval plan:
 
1.	Which index to query
2.	How many chunks to retrieve
3.	Which retrieval mode to use (semantic, BM25, hybrid, metadata only…)
4.	Whether to expand the query or use keywords
 
This is meant to incorporate the confidence information that the previous component outputs into the routing decision. Differently from the intent classifier, I would not use an LLM here, but a deterministic rule based Python class (maybe a decision tree), because latency should be very low, and the system instructions output by it should be highly predictable. 
 
Also, the retrieval planner is supposed to output actual machine-runnable queries (either for the semantic store, or for the BM25 index, or for the metadata DB), to be run in the retrieval algorithm.
 
After the retrieval planner, we run the actual retrieval algorithm:
 
- As mentioned, use a hybrid retriever, that combines semantic vector store ANN search with lexical BM25 keyword search for best precison-recall balancing.
- The retrieval execution is based exactly on the output of the planner, which outputs machine-runnable code. Specifically, it should run the vector search (embed query per source, then search vector index namespace corresponding to source with top_k parameter that was specified by the planner. Each of the k results is chunk text + light chunk metadata). If specified by the planner output, also run BM25 index search for keywords.
- If lookup parameter is SYNTHESIS (see intent classifier section), run parallel search across all sources to decrease latency.
- Optional (ChatGPT suggestion): use a freshness filter. For GH status prefer metadata DB / API to other chunks, for docs, prefer pages with a more recent last_modified, etc.
 
So the output from this stage is an array of top-k chunks per source with metadata and relavance scores.
 
## Step 4: Synthesizer LLM!
 
Two components:
1.	Re-ranking LLM that picks 3-6 most relevant passages across all sources. The reranker is given the query + a set of candidate passages yielded by previous score.
2.	LLM grounded generation of response, with an engineered prompt + context for:-
- Ask the model to answer the user with a short "source" list for each claim, including URLs to sources. 
- Label uncertain statements as such and display raw chunks relevant to the uncertain statement. 
- Could use MCP tool call for authoritative metadata DB searches (e.g. issue state, PR mergeability), that is, the LLM actually calls the API in realtime. 
- Add hallucination penalty!!
 
This is the final answer returned to the user.
 
 
# Production Strategy and MLOps
 
 
## API 
 
Note that the API (e.g., FastAPI) for the RAG system must:
- Receive user queries
- Authenticate
- Call the pipeline: Router (intent classifiers) -> Retreival Planner -> Retrieval Executor -> Reranker LLM -> Prompt Builder -> Final LLM generator.
 
For this to work, the services that must be orchestrated by FastAPI behind the scenes are:
 
1.	Vector store (e.g. Pinecone)
2.	BM25 index (ElasticSearch)
3.	MetadataDB (Postgres, Redis) - authoritative for RAG artifacts
4.	LLM provider (e.g. OpenAI, Anthropic), for final LLM generator
5.	LLM provider for more specific finetuned tasks such as intent classification and reranking

The API should be async-capable to handle multiple simultaneous retrieval and LLM calls efficiently, especially when reranking or fetching from multiple sources.

I will not be writing the code, but here are the minimal responsibilities and endpoints for this:
 
### Minimal responsibilities:
- auth & rate limit
- call intent classifier (router)
- ask retrieval planner for plan
- execute plan via retrieval executors (talk to vector store, BM25, metadata DB)
- run reranker (if configured)
- construct prompt & call generation model
- return answer + evidence + provenance + tool links
 
### Minimal endpoints:
- POST /v1/query — main query endpoint
- GET /healthz — liveness
- GET /readyz — readiness 
- POST /admin/reindex — admin trigger for ingestion (authenticated)
 
 
## Containerization
 
Containerize the following separately:
1.	API / Orchestrator. Needs async support, isolates dependencies.
2.	Data ingestion workers. Are ".worker" dockerfiles, will be long running because of periodic reading of docs, GH, slack, and updates of vector store / BM25 / metadata DB.
3.	Vector store. Use Pinecone client (needed only if self hosted, otherwise API client inside API container), to ensure consistent versioning and configs.
4.	BM25 - ElasticSearch container.
5.	Metadata DB - PostgreSQL container.
6.	Reranker LLM container - containerize to allow it to scale independantly from API.
7.	No need for LLM service containerization since we are using external API client (inside API container, OpenAPI or Anthropic).

## Container Orchestration and Deployment

Orchestrate the above containers with OpenShift:
 
- Pods and services: Each component runs in an OpenShift pod. the services expose stable endpoints and routes expose API externally.

- Use Persistent Volumes for BM25 indices and metadata DB to survive restarts (persistance)

- Scaling and health: use Horizontal Pod Autoscaler for API, workers, reranker, do liveness/readiness probes to ensure self-healing.
  - Autoscaling triggers: We were asked to scale, so let API pods scale based on query rate or CPU/memory usage. Let ingestion workers scale based on backlog of new documents or embeddings. Let reranker pods scale based on number of candidate passages to score.

- Config and secrets: use ConfigMaps for config, Secrets for credentials (LLM keys,GH/Slack tokens).

- FINAL WORKFLOW: API receives query → router → retrieval planner → executor → reranker → final LLM → response.
- 
- Ingestion workers keep vector store, BM25, and metadata DB up to date (see "When are vector store and BM25 index updated?" above)|



## Monitoring

Here are KPIs I would monitor to ensure the RAG system is reliable, responsive, and provides high-quality answers (according to the correct question type, querying the correct source, etc.).


### System performance

- Retrieval accuracy = % of the top-k passages retrieved by semantic vector store search and BM25 that contain relevant content.
- Routing correctness = how many of queries are correctly labeled by intent classifier (factual, status check, synthesis)
- Data freshness = average age of latest vector store \ BM25 chunks over time. This can also "quantify" drift
- Error / failure rate = errors like API errors, retreival failures, LLM generation errors (could be redundant if we have pod health, but this is not pod-specific but the general error rate of the system)
### Answer quality

- Hallucination rate = fraction of responses with unsupported facts, incorrect sources or references, etc.
- Coverage of evidence = % answers linked to at least one passage from some source (the more the model provides responses that are not based on the retrieved chunks, the worse this metric is)

### "Operational" stuff
- CPU, memory, GPU utilization per pod: API, ingestion workers, reranking, synthesis LLM..
- Pod health - how many crashes/restarts..
- Throughput and latency - average query processing time, RPS / QPS, TTFB..


Logging of metrics can be done via an OpenShift monitoring dashboard. 

Some of these metrics require human labeling (retrieval accuracy, routing correctness, hallucination rate...). However, practically it would be better to approximate:
- Use human review only for a subsample of queries (query periodically)
- Create synthetic benchmarks (evaluation set) with known answers.
- Use usage signals from users (clicks, feedback, how long to next query) as proxy for relevance.


## CI / CD

I already described the flow throughout the explanation of the architecture above, but here's a diagram (thanks to ChatGPT):
![rag_cicd_pipeline.png](..%2Frag_cicd_pipeline.png)


