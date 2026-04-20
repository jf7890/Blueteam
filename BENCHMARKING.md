# BlueAgent Benchmarking

This guide is written for Ubuntu/Linux servers.

There are two benchmark flows:

1. `Classification benchmark`: end-to-end `benign/malicious` evaluation against the live API.
2. `RAG benchmark`: retrieval-only evaluation against the Qdrant knowledge base.

All commands below assume you run them from the repo root:

- `/path/to/Blueteam-Agent`

## 1. Prerequisites

Install Python and virtualenv tooling if needed:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

Create and activate a virtual environment:

```bash
cd /path/to/Blueteam-Agent
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements-ubuntu-cpu.txt
```

Why this file exists:

- `sentence-transformers` pulls `torch`
- on Linux, plain `pip install -r requirements.txt` may resolve a CUDA-enabled PyTorch build
- that downloads several large `nvidia-*` wheels and often fills the disk on smaller servers
- `requirements-ubuntu-cpu.txt` pins a CPU-only PyTorch build instead

Prepare `.env`:

```bash
cp .env.example .env
```

Fill at least:

- `GOOGLE_API_KEY`
- `REDIS_URL`
- `QDRANT_URL`
- `QDRANT_COLLECTION`

Optional benchmark toggle:

- `RAG_ENABLED=false` to disable retrieval cleanly
- `NO_RAG=true` is supported as an inverse alias

When RAG is disabled:

- the pipeline still runs normally
- `rag_context` stays empty by design
- benchmark summaries record `config.rag_enabled=false`
- this is preferred over pointing to a fake collection name because it cleanly separates `no-RAG` from `collection not ready`

Important for Ubuntu host runs:

- if you run `python3 app.py` or `python3 -m workers.batch_processor` directly on the host, do not use `REDIS_URL=redis://redis:6379/0`
- `redis` is only resolvable inside Docker Compose service networking
- for host runs, use a host-accessible Redis endpoint such as `redis://localhost:6379/0`

If you already hit `OSError: [Errno 28] No space left on device`, clear pip cache first:

```bash
rm -rf ~/.cache/pip
```

If `/tmp` is small but another mount has free space, install with a different temporary directory:

```bash
mkdir -p /path/with/free-space/pip-tmp
TMPDIR=/path/with/free-space/pip-tmp pip install --no-cache-dir -r requirements-ubuntu-cpu.txt
```

## 2. Start the system

If Redis, Qdrant, and the app already run on your server, reuse them. Otherwise start them first.

Typical host-run flow:

```bash
cd /path/to/Blueteam-Agent
source .venv/bin/activate
python3 app.py
```

In another shell, start the batch worker:

```bash
cd /path/to/Blueteam-Agent
source .venv/bin/activate
python3 -m workers.batch_processor
```

The classification benchmark uses `http://localhost:5000` by default.

To run a `no-RAG` benchmark, export the flag before starting both the API and the batch worker:

```bash
export RAG_ENABLED=false
```

If you use Docker for Redis, make sure Redis is reachable from the host. Example:

```bash
docker run -d --name blueteam-redis -p 6379:6379 redis:7-alpine
```

Then set:

```bash
export REDIS_URL=redis://localhost:6379/0
```

## 3. Clear cache before each serious run

This matters. If Redis cache is not cleared, old verdicts will appear as cache hits and distort both effectiveness and latency numbers.

Use the helper script:

```bash
cd /path/to/Blueteam-Agent
chmod +x scripts/clear_benchmark_cache.sh
./scripts/clear_benchmark_cache.sh
```

If your Redis is not local, pass `REDIS_URL`:

```bash
REDIS_URL=redis://127.0.0.1:6379/0 ./scripts/clear_benchmark_cache.sh
```

If you prefer manual commands:

```bash
redis-cli --scan --pattern 'waf:verdict:*' | while read -r key; do redis-cli DEL "$key" >/dev/null; done
redis-cli --scan --pattern 'waf:result:*' | while read -r key; do redis-cli DEL "$key" >/dev/null; done
redis-cli DEL waf:queue:llm_analysis >/dev/null
redis-cli DEL waf:queue:llm_analysis_dlq >/dev/null
```

## 4. Dataset format

### Minimum schema for classification benchmark

Use either a Hugging Face dataset or a local `csv/json/jsonl/parquet` file.

Required columns:

- raw request column: one of `requests`, `raw_http`, or specify `--request-column`
- label column: one of `label`, `true_label`, or specify `--label-column`

Optional columns:

- `attack_type`
- `sample_id`

### Minimum schema for RAG benchmark

Required:

- query column: one of `query`, `payload`, `text`, `raw_http`, `requests`, or specify `--query-column`

Recommended:

- `label`
- `attack_type`
- `sample_id`

### Example CSV

```csv
sample_id,raw_http,label,attack_type
1,"GET /search?q=%3Cscript%3Ealert(1)%3C/script%3E HTTP/1.1
Host: example.com

",malicious,XSS
2,"GET /products?id=12 HTTP/1.1
Host: example.com

",benign,
```

### Example JSONL

```json
{"sample_id":"1","raw_http":"GET /test?q=%3Cscript%3E HTTP/1.1\r\nHost: example.com\r\n\r\n","label":"malicious","attack_type":"XSS"}
```

## 5. Classification benchmark

### Run with the original Hugging Face dataset

```bash
cd /path/to/Blueteam-Agent
source .venv/bin/activate
python3 scripts/run_benchmark.py \
  --dataset nquangit/CSIC2010_dataset_classification \
  --split train \
  --benign-samples 300 \
  --malicious-samples 300 \
  --shuffle
```

### Run with a local CSV

```bash
cd /path/to/Blueteam-Agent
source .venv/bin/activate
python scripts/benchmark_rag.py \
  --dataset csv \
  --data-file data/pkdd_request_level_eval.csv \
  --split train \
  --query-column raw_http \
  --label-column label \
  --attack-type-column attack_type \
  --benign-samples 0 \
  --malicious-samples 300 \
  --top-k 3 \
  --shuffle \
  --seed 20260419 \
  --output-dir data/Q1_4
```

### Outputs

The script writes:

- `classification_details_<timestamp>.csv`
- `classification_summary_<timestamp>.json`

Main additions in this benchmark:

- overall confusion matrix and binary metrics
- per-source metrics such as `rule_engine`, `llm_batch`, `cache (...)`
- attack-type evaluation if ground-truth `attack_type` exists

### How to read the new attack-type metrics

- `joint_exact_match_accuracy`: among malicious samples with ground-truth attack type, how often the system both flags malicious and names the correct attack type
- `conditional_exact_match_accuracy`: among samples detected as malicious, how often the predicted attack type matches the ground truth
- `macro_f1`: macro-average across attack classes
- `prediction_coverage`: fraction of eligible malicious samples where the system produced any attack-type label

## 6. RAG benchmark

This benchmark evaluates retrieval quality before the LLM reasons on the retrieved context.

By default, the script follows the runtime retrieval path:

- `preprocess`
- `gatekeeper`
- `RAG`

If `DEBUG=true`, the standalone benchmark also writes the same debug CSV shape
used by the live service to `DEBUG_CSV_PATH`. This does not require `app.py`
or the batch worker.

### Run with a local CSV

```bash
cd /path/to/Blueteam-Agent
source .venv/bin/activate
python3 scripts/benchmark_rag.py \
  --dataset csv \
  --data-file ./data/rag_eval.csv \
  --split train \
  --query-column raw_http \
  --label-column label \
  --attack-type-column attack_type \
  --benign-samples 100 \
  --malicious-samples 100 \
  --top-k 5 \
  --min-score 0.0 \
  --shuffle
```

### Outputs

The script writes:

- `rag_details_<timestamp>.csv`
- `rag_summary_<timestamp>.json`

### Retrieval audit export

Use this when you want a manual-inspection artifact instead of only aggregate metrics.

```bash
cd /path/to/Blueteam-Agent
source .venv/bin/activate
python3 scripts/audit_rag_queries.py \
  --dataset csv \
  --data-file ./data/pkdd_request_level_eval.csv \
  --split train \
  --query-column raw_http \
  --label-column label \
  --attack-type-column attack_type \
  --only-label malicious \
  --top-k 3 \
  --max-samples 50 \
  --max-payloads-per-request 20 \
  --output-dir ./data/rag_audit
```

This script writes:

- `rag_audit_<timestamp>.csv`
- `rag_audit_<timestamp>.jsonl`
- `rag_audit_summary_<timestamp>.json`

Useful fields in the audit export:

- `raw_query_text`: original raw HTTP request or raw query text
- `normalized_payloads`: normalized values extracted from the request
- `normalized_payload_details`: source-aware payload trace (`path/query/header/body`)
- `retrieval_payloads`: payloads actually eligible for retrieval
- `skipped_payloads`: payloads skipped because they were too short or the collection was not ready
- `retrieved_topk_*`: merged top-k retrieval results
- `per_payload_trace` in JSONL: top hits for each individual normalized payload

Useful control flags:

- `--max-samples`: limit how many requests are audited from the dataset
- `--sample-id`: audit only specific requests
- `--only-attack-type`: keep only one or more attack families
- `--max-payloads-per-request`: limit how many normalized payloads are queried inside each request

### Reported RAG metrics

For malicious queries with ground-truth `attack_type`:

- `Hit@k`
- `Precision@k`
- `MRR`
- `nDCG@k`

For benign queries:

- `context_return_rate`
- `mean_top1_score`

This answers two separate questions:

- Does RAG retrieve relevant attack examples for malicious traffic?
- Does RAG inject noisy malicious context into benign traffic?

## 7. Recommended evaluation procedure

Run these three experiments on the same fixed test set:

1. current system with the current RAG knowledge base
2. same system with RAG disabled
3. same system with updated payload knowledge

Keep these fixed:

- API model
- prompt
- concurrency
- test set
- benchmark script settings

Change only one variable at a time.

## 8. Recommended file organization

Keep two local benchmark datasets:

- `data/classification_eval.csv`
- `data/rag_eval.csv`

Suggested `classification_eval.csv` columns:

- `sample_id`
- `raw_http`
- `label`
- `attack_type`
- `source_dataset`

Suggested `rag_eval.csv` columns:

- `sample_id`
- `raw_http` or `query`
- `label`
- `attack_type`

## 9. Rebuilding the Qdrant collection

If you update payload knowledge, rebuild the collection:

```bash
cd /path/to/Blueteam-Agent
source .venv/bin/activate
python3 qdrant_collection_builder/build_collection.py \
  --payload-dir ./payloads \
  --collection waf_payloads \
  --recreate
```

The builder stores extra metadata:

- `category`
- `raw_payload`
- `source_file`
- `line_no`

That makes retrieval audit easier.

## 10. Common mistakes

- Do not compare runs with a dirty Redis cache.
- Do not compare runs where the test set changed.
- Do not use LLM-as-a-judge for the classifier verdict.
- Do not claim RAG is useful unless `no-RAG` vs `with-RAG` has been measured on the same set.
