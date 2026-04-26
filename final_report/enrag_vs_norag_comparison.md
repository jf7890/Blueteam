# `fullbench_enrag` vs `fullbench_norag`

Sources:
- `final_report/fullbench_enrag/classification_summary_20260424_020813.json`
- `final_report/fullbench_norag/classification_summary_20260424_023206.json`

## Compact Summary

Delta = `RAG - no_RAG`

| Metric | RAG | no_RAG | Delta |
|---|---:|---:|---:|
| Accuracy | `0.8838` | `0.8838` | `+0.0000` |
| Precision | `0.9510` | `0.9380` | `+0.0130` |
| Recall | `0.9143` | `0.9286` | `-0.0143` |
| F1-score | `0.9323` | `0.9332` | `-0.0010` |
| FPR | `0.3300` | `0.4300` | `-0.1000` |
| FNR | `0.0857` | `0.0714` | `+0.0143` |
| Joint exact match | `0.7486` | `0.5500` | `+0.1986` |
| Conditional exact | `0.8188` | `0.5923` | `+0.2264` |
| Macro-F1 | `0.7955` | `0.6097` | `+0.1859` |
| Coverage | `0.9143` | `0.9286` | `-0.0143` |

## Detailed Comparison

| Group | Metric | `enrag` | `no_rag` | Better | Notes |
|---|---|---:|---:|---|---|
| Overall | Accuracy | 88.38% | 88.38% | `=` | Overall binary performance is effectively tied |
| Overall | Precision | 95.10% | 93.80% | `enrag` | Higher is better |
| Overall | Recall | 91.43% | 92.86% | `no_rag` | Higher is better |
| Overall | F1-score | 93.23% | 93.32% | `no_rag` | Difference is very small |
| Overall | Specificity | 67.00% | 57.00% | `enrag` | Higher is better |
| Overall | FPR | 33.00% | 43.00% | `enrag` | Lower is better |
| Overall | FNR | 8.57% | 7.14% | `no_rag` | Lower is better |
| Attack Type | Coverage | 91.43% | 92.86% | `no_rag` | Share of malicious samples with a usable attack-type output |
| Attack Type | Joint exact match | 74.86% | 55.00% | `enrag` | Correct on both label and attack type |
| Attack Type | Conditional exact | 81.87% | 59.23% | `enrag` | Correct attack type among samples detected as malicious |
| Attack Type | Macro-F1 | 79.55% | 60.97% | `enrag` | Important for balanced attack-family evaluation |
| Attack Type | Distinct predicted attack labels | 10 | 48 | `enrag` | More stable taxonomy, less label drift |
| LLM Path | Eligible malicious in `llm_batch` | 414 | 440 | - | Q2 sample set differs because routing differs |
| LLM Path | Label accuracy | 91.79% | 88.64% | `enrag` | Computed only on malicious samples routed to `llm_batch` |
| LLM Path | Attack-type accuracy | 89.86% | 52.95% | `enrag` | Large gap |
| LLM Path | True attack type in top-k | 82.61% | 0.00% | `enrag` | `no_rag` has no retrieval |
| LLM Path | Predicted attack type in top-k | 78.99% | 0.00% | `enrag` | Output is supported by retrieved context |
| LLM Path | Grounded correct | 78.26% | 0.00% | `enrag` | Correct and supported by top-k context |
| Performance | Throughput (RPS) | 0.80 | 0.88 | `no_rag` | Higher is better |
| Performance | Avg latency | 348,865 ms | 300,179 ms | `no_rag` | Lower is better |
| Performance | P95 latency | 925,269 ms | 850,306 ms | `no_rag` | Lower is better |
| Performance | Immediate path | 314 | 275 | - | `enrag` includes `gatekeeper_ml` traffic |
| Performance | Queued path | 486 | 525 | - | `no_rag` sends more samples to `llm_batch` |

Quick takeaways:
- If you only care about binary `malicious/benign`, the two runs are nearly tied.
- If you care about attack-type quality, `enrag` is clearly stronger.
- `enrag` reduces false positives better, while `no_rag` is slightly better on false negatives.
- `no_rag` is faster, while `enrag` trades speed for groundedness and better LLM-side classification quality.
- This is not a perfectly clean RAG-only A/B comparison because `enrag` also enables `gatekeeper`, while `no_rag` does not.
