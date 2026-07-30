# TacticalGraph

TacticalGraph is a high-performance Knowledge Graph (Neo4j) data ingestion engine and GraphRAG platform designed for football/soccer analytics using Kaggle Transfermarkt data.

---

## Repository Structure

```
tactical-graph/
├── data-pipeline/               # Ingestion pipeline & Neo4j ETL loaders
│   ├── config.py                # Pydantic BaseSettings (.env loading & validation)
│   ├── database.py              # Neo4j driver pool, session execution & retries
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
Copy `.env.example` to `.env` and fill in your Neo4j database credentials:
```bash
cp .env.example .env
```

Default credentials in `.env`:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
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

The pipeline is orchestrated via `data-pipeline/main.py`:

```bash
cd data-pipeline
```

### Full Ingestion Mode
Loads all raw CSV data into Neo4j:
```bash
python main.py --mode full --data-dir ./data
```

### Incremental Ingestion Mode
Filters game datasets to process records with `game_date > last_watermark_date`:
```bash
python main.py --mode incremental --data-dir ./data
```

### Development Mode (`DEV_MODE`)
Sub-filters the dataset to a single target competition (default: Premier League `'GB1'`) to allow rapid testing while maintaining complete referential integrity:
```bash
# Filter Premier League (GB1) subgraph
python main.py --dev --data-dir ./data

# Filter target competition (e.g. La Liga 'ES1')
python main.py --dev-comp ES1 --data-dir ./data
```

---

## Graph Data Model & Documentation

For details on node labels, relationship types, property schemas, constraints, and performance indexes, refer to [schema.md](file:///C:/Projects/tactical-graph/data-pipeline/schema.md).
