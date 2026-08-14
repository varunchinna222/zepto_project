# Support Assistant

The required baseline is fully deterministic for LLM generation with `MOCK_LLM` unset or `1`. Embeddings are generated locally with `sentence-transformers` `all-MiniLM-L6-v2`; ChromaDB is the vector store.

## Architecture

`docs/*.txt` → **ingestion/chunking** in `common.py` → local **embedding** with SentenceTransformer → ChromaDB collection `zepto_policy` → LangGraph `retrieve_and_answer` performs top-3 cosine retrieval → mock generation returns a grounded snippet. `classify_intent` routes policy questions to retrieval or general questions to `direct_answer`.

Only generation/classification generation behavior is gated by `MOCK_LLM`; retrieval and embeddings remain real in both modes. The default path makes no LLM-provider call. `main.py` contains the structured prompt template with role/context/task/format/length, a negative constraint, and a few-shot example for an optional real-provider extension.

## Run

```bash
python support_assistant/build_index.py
MOCK_LLM=1 uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```

Example retrieval call:

```json
{"query":"What is the delivery fee below INR 149?"}
```

Example general call:

```json
{"query":"What is the capital of France?"}
```

The first routes to `policy_question` → `retrieve_and_answer`; the second routes to `general_question` → `direct_answer`. The response schema is always `answer`, `sources`, `confidence`.

## Docker

From the repository root:

```bash
docker build -t zepto-support-assistant support_assistant
docker run --rm -p 7860:7860 zepto-support-assistant
```
