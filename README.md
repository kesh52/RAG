# Parameterized RAG Pipeline & Evaluation Project

This repository contains a modular RAG (Retrieval-Augmented Generation) pipeline integrated with PostgreSQL (pgvector) and Vertex AI services (Embeddings, Semantic Ranker, Gemini). It includes evaluation tests using the Ragas framework.

## Directory Structure

```
├── .env                         # Centralized environment configurations
├── requirements.txt             # Python project dependencies
├── pytest.ini                   # Pytest configuration file
├── architecture.md              # Pipeline architecture and class diagrams
├── config.yaml                  # Central configuration parameters
├── alembic.ini                  # Alembic migrations configuration file
├── alembic/                     # Alembic database migrations
│   ├── env.py                   # Migration environment setup
│   └── versions/                # Database version migration scripts
├── assets/                      # Visual and attachment assets storage
│   └── uploaded/                # Local target directory for crawler downloaded files
├── docs/                        # Project documentation
│   └── package_structure.md     # Package restructuring design plan
├── src/                         # Core Python package
│   ├── db/                      # Database package
│   │   └── pg_connector.py      # Database connection helper
│   ├── utils/                   # Shared utility tools
│   │   ├── config.py            # Configuration loader and env resolver
│   │   └── logger.py            # Centralized rotating logger setup
│   ├── embeddings/              # Embeddings package
│   │   ├── base.py              # Embedding service interface
│   │   └── vertex.py            # Vertex AI embedding implementation
│   ├── retrieval/               # Retrieval package
│   │   ├── base.py              # Retriever interface
│   │   └── postgres.py          # Postgres pgvector retriever
│   ├── reranking/               # Reranking package
│   │   ├── base.py              # Reranker interface
│   │   └── vertex.py            # Vertex Semantic Ranker implementation
│   ├── pipeline/                # Orchestrator package
│   │   └── orchestrator.py      # RAG Pipeline orchestrator
│   ├── feedback/                # Playground & user feedback store
│   │   ├── README.md            # Interactive playground & continuous feedback documentation
│   │   └── feedback_store.py    # PostgreSQL persistence, analytics & dataset promotion
│   └── etl/                     # ETL crawler package
│       ├── README.md            # ETL architecture & continuous feedback loop documentation
│       ├── confluence.py        # Confluence scrapers and BFS crawlers
│       ├── chunking.py          # Sliding-window text chunker
│       └── pipeline.py          # ETL orchestrator
├── scripts/                     # Executable scripts
│   ├── assets/                  # Seed generator mock assets (PDFs, BMP images)
│   ├── check_connection.py      # Test database connectivity and print diagnostic info
│   ├── clear_db.py              # Drop all tables in the connected database schema
│   ├── seed_db.py               # Seed PostgreSQL with documents and embeddings
│   ├── run_etl.py               # Run Confluence crawler ETL pipeline against a live target
│   ├── run_pipeline.py          # Run query pipeline via command line
│   ├── run_migrations.py        # Run Alembic database migrations programmatically
│   └── start_proxy.py           # Launch Cloud SQL Auth Proxy secure tunnel
├── evaluation/                  # Comparative evaluation runner
│   └── evaluate_ragas.py        # Compares basic vs advanced pipeline with Ragas
└── tests/                       # Unit and integration tests
    ├── conftest.py              # Pytest configuration and environment setup
    ├── test_db.py               # Database connection tests
    ├── test_pipeline.py         # Mock pipeline tests
    └── test_etl.py              # ETL pipeline unit tests
```

## Setup Instructions

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment**:
    Create a `.env` file in the root directory for credentials and GCP settings:
    ```ini
    DB_PASSWORD=your_password
    INSTANCE_CONNECTION_NAME=your_gcp_project:europe-west3:your_db_instance
    GCP_PROJECT=your_gcp_project
    GCP_LOCATION=europe-west3
    GOOGLE_CLOUD_QUOTA_PROJECT=your_gcp_project
    ```
    
    Non-sensitive parameters (such as models, pipeline sizes, and chunk settings) can be configured directly in [config.yaml](file:///Users/ilja/DEV/AI/config.yaml), which dynamically resolves any environment variables defined in `.env` using `${VAR_NAME:default}` notation.

3.  **Start Cloud SQL Auth Proxy**:
    To connect securely to the managed Cloud SQL database without public IP access, launch the Cloud SQL Auth Proxy:
    ```bash
    # Install the proxy command (macOS / Homebrew example)
    brew install cloud-sql-proxy

    # Start the proxy tunnel using configuration from .env
    python3 scripts/start_proxy.py
    ```
    Keep this process running in a dedicated terminal window while using the database.

    *Verify Connection (Optional)*:
    You can diagnose and test database connectivity and configuration parameters at any time by running:
    ```bash
    python3 -m scripts.check_connection
    ```

4.  **Run Database Migrations**:
    Apply the database schema (enabling pgvector, creating the documents table, generated FTS column, and HNSW/GIN indexes) using Alembic:
    ```bash
    python3 -m scripts.run_migrations
    ```
    This script programmatically runs the Alembic migration engine and applies the initial schema revision to your database instance.

5.  **Seed Database**:
    Ingest sample documents and generate/store vector embeddings into PostgreSQL:
    ```bash
    python3 -m scripts.seed_db
    ```

6.  **Run Pipeline**:
    Execute queries using the hybrid/reranked RAG pipeline:
    ```bash
    python3 -m scripts.run_pipeline "What index does PostgreSQL use for vector similarity search?"

    File as source example
    python3 -m scripts.run_pipeline 'What elements make up Spring Batch according to the manual?'
    ```

7.  **Run Evaluations**:
    Run comparative performance evaluations using Ragas:
    ```bash
    python3 -m evaluation.evaluate_ragas
    ```

8.  **Run Tests**:
    Execute pytest to run database connection and pipeline mock unit tests:
    ```bash
    python3 -m pytest
    ```

9.  **Interactive Admin Dashboard & Continuous Feedback Loop**:
    Launch the Streamlit management dashboard featuring ETL crawler triggers, pgvector database explorer, Ragas evaluation runner, and the interactive remediation playground:
    ```bash
    python3 -m streamlit run admin_app.py
    ```

    - **Multimodal Remediation & Feedback**: Drag and drop incident report PDFs, analyze issues, rate answers, and triage failures into regression benchmark datasets.
    - Detailed documentation: See [src/feedback/README.md](file:///Users/ilja/DEV/AI/src/feedback/README.md).
    - ETL crawler architecture: See [src/etl/README.md](file:///Users/ilja/DEV/AI/src/etl/README.md).


