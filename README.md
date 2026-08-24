# Parameterized RAG Pipeline & Evaluation Project

This repository contains a modular RAG (Retrieval-Augmented Generation) pipeline integrated with PostgreSQL (pgvector) and Vertex AI services (Embeddings, Semantic Ranker, Gemini). It includes evaluation tests using the Ragas framework.

## Directory Structure

```
├── .env                         # Centralized environment configurations
├── requirements.txt             # Python project dependencies
├── pytest.ini                   # Pytest configuration file
├── architecture.md              # Pipeline architecture and class diagrams
├── config.yaml                  # Central configuration parameters
├── db/                          # Database migration definitions
│   └── changelog.yaml           # Liquibase database schema definition
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
│   └── pipeline/                # Orchestrator package
│       └── orchestrator.py      # RAG Pipeline orchestrator
├── scripts/                     # Executable scripts
│   ├── seed_db.py               # Seed PostgreSQL with documents and embeddings
│   ├── run_pipeline.py          # Run query pipeline via command line
│   ├── run_migrations.py        # Run database migrations using pyliquibase
│   └── start_proxy.py           # Launch Cloud SQL Auth Proxy secure tunnel
├── evaluation/                  # Comparative evaluation runner
│   └── evaluate_ragas.py        # Compares basic vs advanced pipeline with Ragas
└── tests/                       # Unit and integration tests
    ├── conftest.py              # Pytest configuration and environment setup
    ├── test_db.py               # Database connection tests
    └── test_pipeline.py         # Mock pipeline tests
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

4.  **Run Database Migrations**:
    Apply the database schema (including enabling pgvector and creating the documents table) using Liquibase:
    ```bash
    python3 -m scripts.run_migrations
    ```
    *Notes:*
    * *Since Liquibase is Java-based, you must have a Java Runtime Environment (JRE) installed on your system to run the migrations.*
    * *On macOS, if you encounter `[SSL: CERTIFICATE_VERIFY_FAILED]` when downloading Liquibase, execute Python's built-in certificate installer command: `/Applications/Python\ 3.14/Install\ Certificates.command` (adjust the version folder name matching your Python version).*

5.  **Seed Database**:
    Ingest sample documents and generate/store vector embeddings into PostgreSQL:
    ```bash
    python3 -m scripts.seed_db
    ```

6.  **Run Pipeline**:
    Execute queries using the hybrid/reranked RAG pipeline:
    ```bash
    python3 -m scripts.run_pipeline "What index does PostgreSQL use for vector similarity search?"
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

