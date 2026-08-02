# TacticalGraph

TacticalGraph is a high-performance Knowledge Graph (Neo4j) data ingestion engine and GraphRAG platform designed for football/soccer analytics using Kaggle Transfermarkt data.

> Developed with **Antigravity CLI**.

---

## Repository Structure

```
tactical-graph/
├── data-pipeline/               # Ingestion pipeline & Neo4j ETL loaders
│   ├── config.py                # Pydantic BaseSettings (.env loading & validation)
│   ├── database.py              # Neo4j driver pool, session execution & retries
│   ├── dataset.py               # DatasetManager (Kaggle dataset download, caching & retries via kagglehub)
│   ├── schema.py                # SchemaInstaller (uniqueness constraints & indexes)
│   ├── watermark.py             # WatermarkManager for delta incremental updates
│   ├── utils.py                 # Subgraph extraction & DEV_MODE filtering helpers
│   ├── main.py                  # CLI pipeline orchestrator
│   ├── schema.md                # Graph Data Model & Cypher Schema specification
│   └── loaders/                 # Domain-grouped ETL loaders
│       ├── base_loader.py       # Abstract BaseLoader class
│       ├── reference_loader.py  # Countries, Competitions, NationalTeams
│       ├── entities_loader.py   # Clubs, Players, PlayerValuations
│       ├── transfers_loader.py  # Transfers
│       ├── matches_loader.py    # Games, Club Games
│       ├── appearances_loader.py# Appearances & Game Lineups
│       └── game_events_loader.py# Game Events
├── graphrag-app/                # Reserved for LLM engine and API layer
├── .env.example                 # Environment variable template
└── README.md
```

---

## Setup & Environment Configuration

### 1. Environment File setup
Copy `.env.example` to `.env` and fill in your Neo4j database credentials and optional Kaggle API token:
```bash
cp .env.example .env
```

Configuration parameters in `.env`:
```env
# Neo4j Database Credentials & Connection Endpoint
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j

# Kaggle Dataset Handles (Optional Kaggle API token for authentication)
KAGGLE_API_TOKEN=your_kaggle_api_token_here
KAGGLE_DATASET_HANDLE=davidcariboo/player-scores
TRANSFER_DATASET_HANDLE=mexwell/football-player-transfers
```

### 2. Python Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r data-pipeline/requirements.txt
```

---

## Running the Data Pipeline

The pipeline is orchestrated via `data-pipeline/main.py`. By default, `DatasetManager` dynamically fetches and caches the primary dataset (`davidcariboo/player-scores`) and supplementary transfer dataset (`mexwell/football-player-transfers`) from Kaggle via `kagglehub` (with tenacity exponential backoff retries for rate limits and automatic local caching to bypass redundant downloads).

```bash
cd data-pipeline
```

### Full Ingestion Mode
Loads all raw CSV data into Neo4j (automatically downloads/retrieves cached datasets via `kagglehub`):
```bash
python main.py --mode full
```

*Note: You can optionally pass `--data-dir PATH` to override dynamic kagglehub retrieval and use a custom local CSV directory:*
```bash
python main.py --mode full --data-dir ./data
```

### Incremental Ingestion Mode
Filters game datasets to process records with `game_date > last_watermark_date`:
```bash
python main.py --mode incremental
```

### Development Mode (`DEV_MODE`)
Sub-filters the dataset to a single target competition (default: Premier League `'GB1'`) to allow rapid testing while maintaining complete referential integrity:
```bash
# Filter Premier League (GB1) subgraph
python main.py --dev

# Filter target competition (e.g. La Liga 'ES1')
python main.py --dev-comp ES1
```

---

## Graph Data Model & Documentation

For details on node labels, relationship types, property schemas, constraints, and performance indexes, refer to [schema.md](file:///C:/Projects/tactical-graph/data-pipeline/schema.md).

---

## Acknowledgments

Developed with **Antigravity CLI**.
