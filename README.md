# TacticalGraph

TacticalGraph is a high-performance Knowledge Graph (Neo4j) data ingestion engine and GraphRAG platform designed for tactical football/soccer analytics using Kaggle Transfermarkt data.

> Developed with **Antigravity CLI**.

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
│   ├── models.py                # Pydantic request/response & structured LLM output models
│   ├── few_shot_bank.py         # Curated NL -> Cypher few-shot prompt bank
│   ├── agent.py                 # GraphRAG agent orchestrator (LLM routing & synthesis)
│   ├── main.py                  # FastAPI REST API & CLI entry point
│   ├── requirements.txt         # Pinned GraphRAG Python dependencies
│   └── tools/                   # LangChain & custom tool modules
│       ├── custom_text2cypher.py# Read-only Text-to-Cypher translation tool
│       └── dedicated_templates.py# Pre-written scouting candidate replacement tool
├── .env.example                 # Environment variable template
└── README.md
```

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
NEO4J_URI=neo4j+s://4446f270.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Kaggle Dataset Handles
KAGGLE_DATASET_HANDLE=davidcariboo/player-scores
TRANSFER_DATASET_HANDLE=mexwell/football-player-transfers

# Gemini & LLM Configuration
GEMINI_API_KEY=your_gemini_api_key
GRAPH_LLM_MODEL=gemini-3.6-flash
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
Returns system operational status, active LLM model, and live Neo4j driver connectivity.

#### `POST /query`
Processes natural language questions, translates them to validated read-only Cypher or routes to tactical tools, executes against Neo4j, and returns synthesized insights.

**Example Request:**
```json
POST /query
Content-Type: application/json

{
  "query": "Who would have been a better defensive midfielder for Manchester United instead of Marouane Fellaini in 2013/14?"
}
```

**Example Response:**
```json
{
  "answer": "Based on data from TacticalGraph, Arturo Vidal (€35M valuation, 40.3% win rate, 38 goal contributions) offered a superior tactical fit over Marouane Fellaini (€28M valuation, 30.6% win rate, 34 goal contributions)...",
  "cypher_used": "DYNAMIC_REPLACEMENT_CANDIDATES_CYPHER (Parameterized)",
  "raw_data": [ ... ],
  "execution_time_ms": 4662.5
}
```

#### `POST /scout/replacements`
Dedicated endpoint calling `get_replacement_candidates` tool directly for fast player scouting without LLM orchestration overhead.

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
from graphrag import query_graphrag

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
