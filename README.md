# Agentic RAG Architectures

This project explores different **agentic Retrieval-Augmented Generation (RAG) architectures** through a set of modular components and experiments.

## Project Structure

### `agents.py`

Contains the different **agents** required for the experiments. These agents implement the various roles and behaviors used across the agentic RAG architectures.

### `system_prompts.py`

Contains the **system prompts** used by the different agents. The prompts define the instructions, roles, and behaviors of each agent.

### `retrievers.py`

Contains the retrieval components used by the experiments:

* **BM25 retriever** — a traditional lexical retrieval method based on term matching.
* **Embedding retriever** — a semantic retrieval method based on vector embeddings.
* **Hybrid search retriever** — combines lexical and semantic retrieval to improve search performance.

### `rags.py`

Combines the agents, prompts, and retrievers defined in the other modules to implement **six different agentic RAG architectures**.

## Goal

The primary goal of this project is to **explore and compare different agentic RAG architectures**, using different combinations of agents and retrieval strategies to understand their behavior and effectiveness across experiments.

