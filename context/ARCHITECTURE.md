# ARCHITECTURE

## Shape

A synchronous Flask API for fast paths, a worker pool for anything involving network fetches or model inference above a latency budget, and a set of independently evaluable analysers whose outputs are composed into an evidence card. No analyser can veto another. The composer aggregates and abstains; it does not decide.

```
Browser extension (MV3)          Web client
        |                            |
        +------------ HTTPS ---------+
                      |
              Caddy (TLS, rate limit L7)
                      |
              Flask API  (gunicorn + gthread)
                      |
     +----------------+------------------+
     |                |                  |
  Redis          Postgres+pgvector   Job queue (RQ)
 (cache,          (registry,              |
  idempotency,     evidence index,        |
  rate limit)      audit log)             |
                                    Worker pool
                                          |
        +---------------+-----------------+----------------+
        |               |                 |                |
   Claim extractor  Retrieval        NLI stance      Rhetoric model
   (LLM, Groq)      (bge embed +     (DeBERTa MNLI    (DeBERTa FT,
                     pgvector +       FEVER ANLI,      multilabel +
                     news API)        ONNX)            span attribution)
        |               |                 |                |
        +---------------+--------+--------+----------------+
                                 |
                          Source registry        Circulation sampler
                          (domain provenance)    (Bluesky, Reddit)
                                 |
                          Evidence composer
                          (aggregate, calibrate, abstain)
```

## Request lifecycle

The client submits either a text selection or a URL. Both paths converge quickly.

For a URL, the API does not fetch it inline. It enqueues a fetch job and returns a job identifier immediately, because remote fetch latency is unbounded and an attacker controlled URL must never hold an API worker thread. The fetch runs in a worker behind the SSRF guarded client described in SECURITY.md, and the extracted text is what enters the pipeline. Raw HTML is discarded after extraction and never persisted.

For a text selection the pipeline starts directly. A content hash becomes the idempotency key, so repeated submission of the same passage returns the cached card rather than reprocessing. The hash is over normalised text, which means whitespace collapsed and Unicode normalised to NFKC, so trivial variations hit the same cache entry.

Claim extraction runs first because everything downstream is claim scoped. It produces zero to five discrete claims with character offsets into the source text, each tagged as factual assertion, opinion, prediction, or attribution of speech. Only factual assertions proceed to retrieval. If extraction produces no factual assertions, the pipeline short circuits and the card says so, which is a legitimate and common outcome for opinion writing.

Retrieval, source resolution, and circulation sampling then run concurrently, since none depends on another's output. Retrieval fans out per claim. NLI stance classification runs on retrieval results as they arrive, batched. Rhetoric analysis runs on the full source text in parallel with all of it.

The composer waits on all branches with a per branch deadline. A branch that misses its deadline is reported as unavailable in the card rather than blocking it. Partial cards are normal and the interface is designed for them.

## Components

### Claim extractor

An LLM call over an OpenAI compatible interface, currently Groq. The prompt is versioned in `prompts/` with a hash recorded in every output, so a card can be traced to the exact prompt that produced it.

This is the one component where an LLM is the right tool: the task is linguistic, low stakes, and the output is structurally validated before use. The output is a JSON schema constrained response, parsed into a Pydantic model, and any claim whose character offsets do not resolve to the actual source text is dropped. That validation matters because it is also the boundary where injected instructions in fetched article text would have to escape, and they cannot escape a schema that only permits offsets and enum tags.

Fallback when the LLM is unavailable: a rule based sentence selector using dependency parse features. Worse, but the pipeline degrades rather than fails.

### Retrieval

Two sources, unified behind one interface.

The local index holds an embedded corpus of reference material, built with `BAAI/bge-base-en-v1.5` at 768 dimensions, stored in pgvector with an HNSW index. Cosine distance. Chunked at roughly 300 tokens with 50 token overlap on sentence boundaries.

Live retrieval hits a news API for recency, because the local index goes stale within days on breaking topics and the whole point of the system is handling current claims. Results are embedded on the fly and scored the same way.

Hybrid scoring: dense similarity combined with BM25 lexical overlap, weighted, because pure dense retrieval loses on proper nouns and numbers, which is precisely the vocabulary claims are made of. The weighting is a config value swept in evaluation, not a magic constant.

### NLI stance classifier

`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` exported to ONNX for CPU inference. Premise is the retrieved passage, hypothesis is the extracted claim, output is entailment, contradiction, or neutral.

This is the component that makes the system more than a search box. Retrieval alone tells you related text exists. Stance classification tells you whether that text agrees with the claim, which is the difference between "here are ten articles about this topic" and "eight retrieved passages contradict this claim and two are neutral".

It is also the component most likely to be wrong in interesting ways, and EVALUATION.md treats it accordingly. Retrieved passages are not clean premises. They contain hedging, attribution to third parties, and partial overlap with the claim. Stance classifiers trained on curated NLI pairs degrade on them, and measuring that degradation is a core deliverable.

### Rhetoric model

`microsoft/deberta-v3-base` fine tuned as a multilabel classifier over sentence spans. Labels: unsourced attribution, emotional intensifier density, manufactured urgency, absolute quantifier use, unnamed expert appeal.

Multilabel and span scoped rather than document level binary, deliberately. A document level truth score is the thing this project exists to avoid. A span level observation that a sentence attributes a claim to unnamed sources is a factual statement about the text, verifiable by the reader looking at the highlighted sentence, and it does not require the model to be right about the world.

Attribution via Integrated Gradients for the demonstration path, though the span scoping means the model already localises without post hoc attribution in normal operation.

### Source registry

Postgres table keyed on registrable domain, resolved through the public suffix list rather than naive string splitting, since subdomain handling is a common source of both bugs and evasion.

Fields carry provenance. Every field records its source and observation date: domain registration age from RDAP, ownership and masthead presence from structured extraction of the about page, corrections policy presence, whether the domain appears in open transparency datasets, and any structured publisher metadata.

The registry never stores a credibility score computed by this project. It stores observations. The interface displays observations. Composing them into a score would recreate the truth verdict problem one level down, and it would also make the registry a defamation surface, which is a real legal exposure and not a hypothetical one.

### Circulation sampler

Bluesky via the public AT Protocol firehose and Reddit via its API. Both chosen for accessibility and terms that permit research use, unlike X, whose current pricing puts it out of reach and whose terms complicate redistribution.

The measure is not volume. Volume is confounded by topic popularity. The measures are account age distribution among posting accounts, text similarity across posts (near identical text across many accounts indicates coordination rather than organic spread), and posting time clustering.

All social data is processed in memory and aggregated. No post text, no handle, no user identifier is persisted. Only the aggregate statistics reach the database. This is a GDPR requirement, not a preference (SECURITY.md, data minimisation).

### Composer and calibration

Takes analyser outputs and produces the card. The critical function is abstention.

Calibration is per component. Each analyser reports a confidence, those confidences are calibrated against held out data using temperature scaling, and the composer applies thresholds below which the component's output is suppressed rather than shown with a hedge. Users do not read hedges; they read the number and discard the qualifier. Suppression is the only honest form of uncertainty communication in a three second interface.

The card shows what was found. It never shows a summary judgement, an overall score, or a colour coded verdict on the claim as a whole.

## Storage

Postgres 16 with pgvector for the evidence index, source registry, and audit log. Redis for idempotency keys, response cache, rate limit counters, and the job queue backend.

Retention: evidence cards 30 days keyed by content hash, then purged. Extracted article text is never persisted, only its hash and the derived claims. Audit log entries retain request metadata and analyser decisions for 90 days for evaluation purposes, with no content.

## Serving and deployment

Docker Compose on a single Hetzner instance. Caddy terminates TLS and applies a first rate limit layer. Flask runs under gunicorn with gthread workers, since the request path is IO bound. RQ workers run as separate containers with independent scaling.

Models serve from ONNX Runtime on CPU. Both the NLI and rhetoric models are base sized and quantised to int8, which keeps single passage inference in the tens of milliseconds and removes the GPU from the deployment entirely. Batch inference in the worker amortises the rest.

The model container is network isolated. It has no egress. It receives text over an internal socket and returns tensors. If a checkpoint were ever compromised, this containment is what prevents it from reaching anything.

## What is deliberately not here

No microservices beyond the API and worker split. No Kubernetes. No feature store. No message broker beyond Redis. Every one of those would be defensible at scale and indefensible here, and the ability to say why an abstraction was rejected is worth more in review than the abstraction itself.
