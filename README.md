# TacticalGraph

TacticalGraph is a high-performance Knowledge Graph (Neo4j) data ingestion engine and GraphRAG platform designed for tactical football/soccer analytics using Kaggle Transfermarkt data.

---

## Repository Structure

```
tactical-graph/
├── data-pipeline/               # Ingestion pipeline & Neo4j ETL loaders
│   ├── config.py                # Pydantic BaseSettings (.env loading & validation)
│   ├── database.py              # Neo4j driver pool, session execution & retries
│   ├── dataset.py               # DatasetManager (Kaggle dataset download & retries via kagglehub)
│   ├── schema.py                # SchemaInstaller (uniqueness constraints & indexes)
│   ├── watermark.py             # WatermarkManager for delta incremental updates
│   ├── utils.py                 # Subgraph extraction & DEV_MODE filtering helpers
│   ├── main.py                  # CLI pipeline orchestrator
│   ├── schema.md                # Graph Data Model & Cypher Schema specification
│   ├── cyphers.json             # Test Cypher query suite & few-shot examples
│   └── loaders/                 # Domain-grouped ETL loaders
│       ├── base_loader.py       # Abstract BaseLoader class
│       ├── reference_loader.py  # Countries, Competitions, NationalTeams
│       ├── entities_loader.py   # Clubs, Players, PlayerValuations
│       ├── transfers_loader.py  # Transfers
│       ├── matches_loader.py    # Games, Club Games
│       ├── appearances_loader.py# Appearances & Game Lineups
│       └── game_events_loader.py# Game Events
├── graphrag-app/                # GraphRAG LLM Engine & REST API
│   ├── config.py                # GraphRAGSettings Pydantic model
│   ├── database.py              # Application-wide single shared Neo4j driver singleton
│   ├── utils.py                 # Response text formatters & clean text extraction helpers
│   ├── models.py                # Pydantic request/response & structured LLM output models
│   ├── few_shot_bank.py         # Curated NL -> Cypher few-shot prompt bank
│   ├── agent.py                 # GraphRAG agent router (tool selection & domain guardrails)
│   ├── main.py                  # FastAPI REST API & lifespan shutdown handler
│   ├── requirements.txt         # Pinned GraphRAG Python dependencies
│   └── tools/                   # LangChain & custom tool modules
│       ├── custom_text2cypher.py# Dynamic Cypher schema extraction & read-only translation tool
│       └── dedicated_templates.py# Encapsulated player scouting parameter extraction & query tool
├── .env.example                 # Environment variable template
└── README.md
```

---

## Architecture & Features

- **Application-Wide Driver Singleton**: Centralized shared Neo4j driver pool (`database.py`) initialized once on demand and reused across all tools and API endpoints with connection pooling and graceful FastAPI lifespan shutdown.
- **Dynamic Cypher Schema Introspection**: `CustomText2CypherTool` extracts active graph node labels, relationship types, and properties directly from Neo4j via Cypher procedures (`CALL db.schema.nodeTypeProperties()`) rather than static schema files.
- **Strict Domain Scope Guardrails**: Rejects out-of-scope non-football prompts (general knowledge, coding, weather, politics) in code logic without making database calls or executing Cypher queries.
- **Encapsulated Modular Tools**:
  - `get_replacement_candidates`: Encapsulates parameter extraction LLM calls, prior-season scouting rule calculations, multi-step Cypher execution, and executive report formatting.
  - `query_graph_with_custom_cypher`: Handles Text-to-Cypher generation, read-only security validation, Cypher execution, and clean text response formatting.

---

## Setup & Environment Configuration

### 1. Environment File setup
Copy `.env.example` to `.env` and fill in your Neo4j database credentials, Kaggle handles, and Gemini API key:
```bash
cp .env.example .env
```

Configuration parameters in `.env`:
```env
# Neo4j Database Credentials & Connection Endpoint
NEO4J_URI=neo4j+s://instance-id.databases.neo4j.io
NEO4J_USER=user_name
NEO4J_PASSWORD=password
NEO4J_DATABASE=db_name

# Kaggle Dataset Handles
KAGGLE_DATASET_HANDLE=davidcariboo/player-scores
TRANSFER_DATASET_HANDLE=mexwell/football-player-transfers

# Gemini & LLM Configuration
GEMINI_API_KEY=your_gemini_api_key
LLM_MODEL=gemini-3.5-flash-lite
```

### 2. Python Virtual Environment Setup
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies for both data pipeline and GraphRAG application:
pip install -r data-pipeline/requirements.txt
pip install -r graphrag-app/requirements.txt
```

---

## Running the Data Pipeline

The pipeline is orchestrated via `data-pipeline/main.py`. `DatasetManager` dynamically fetches and caches datasets from Kaggle via `kagglehub`.

```bash
cd data-pipeline
```

### Full Ingestion Mode
Loads all raw CSV data into Neo4j:
```bash
python main.py --mode full
```

### Incremental Ingestion Mode
Filters game datasets to process records with `game_date > last_watermark_date`:
```bash
python main.py --mode incremental
```

### Development Mode (`DEV_MODE`)
Sub-filters the dataset to a single target competition (default: Premier League `'GB1'`):
```bash
python main.py --dev
```

---

## Running the GraphRAG Application

The GraphRAG framework operates as a FastAPI REST application and a programmatic Python library.

### 1. Starting the FastAPI REST Server

From the project root directory:
```bash
python graphrag-app/main.py
```
Or directly with Uvicorn:
```bash
uvicorn graphrag-app.main:app --reload --port 8000
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### 2. REST API Endpoints

#### `GET /health`
Returns system operational status, active LLM model, and live Neo4j driver connectivity using the application driver pool.

#### `POST /chat`
Processes natural language questions, routes to appropriate tools, executes Cypher queries against Neo4j, and returns structured `GraphRAGResponse` payloads. Rejects out-of-scope non-football queries.

**Example Request:**
```json
POST /chat
Content-Type: application/json

{
  "query": "Who would have been a better defensive midfielder for Manchester United instead of Marouane Fellaini in 2013/14?"
}
```

**Example Response:**
```json
{
  "answer": "Executive Scouting Report: Replacement Candidates for Fellaini (Manchester United, Season 2012/2013)\n\n1. Benchmark Baseline (Fellaini):\n   - Role / Sub-Position: Defensive Midfield\n   - Formation Fit: 30.6% | Team Win Rate: 50.0%\n   - Market Valuation: EUR 28,000,000 | Goal Contributions: 11 goals, 5 assists (16 total)\n\n2. Positional Scouting Analysis (Top Replacement Candidates):\n   1) Arturo Vidal (Central Midfield):\n      - Match Activity: 2450 mins in 31 matches\n      - Formation Compatibility: 85.0% | Team Win Rate: 72.5%\n      - Market Valuation: EUR 35,000,000 | Goal Contributions: 10 goals, 8 assists (18 total)\n\n3. Final Recruitment Recommendation:\n   Primary recruitment target is Arturo Vidal, offering the optimal sub-position match, tactical formation fit, and team win-rate compatibility for Manchester United.",
  "cypher_used": "DYNAMIC_REPLACEMENT_CANDIDATES_CYPHER"
}
```

#### `POST /scout/replacements`
Dedicated endpoint executing dynamic scouting candidate Cypher queries directly via `execute_replacement_candidates_query` without LLM orchestration overhead.

**Example Request:**
```json
POST /scout/replacements
Content-Type: application/json

{
  "target_club": "Manchester United",
  "season": 2012,
  "position": "Midfield",
  "benchmark_player": "Fellaini",
  "min_minutes": 1000
}
```

### 3. Programmatic Python Entry Point

Import and call `query_graphrag` in your Python code:

```python
from agent import query_graphrag

response = query_graphrag("List top 5 transfers in history")

print("Cypher Executed:", response.cypher_used)
print("Answer:", response.answer)
```

---

## Graph Data Model & Documentation

For details on node labels, relationship types, property schemas, constraints, and performance indexes, refer to [schema.md](file:///C:/Projects/tactical-graph/data-pipeline/schema.md).

---

## Acknowledgments

Developed with **Antigravity CLI**.
