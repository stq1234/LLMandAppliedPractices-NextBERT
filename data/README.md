# Local assets and prepared data

All source acquisition and tokenization run locally. The remote training server does not download source data. Source repositories and datasets are pinned to immutable Hugging Face revisions in `src/prepare_assets.py`, `src/prepare_data.py`, and the generated manifests.

## Prepare assets

```powershell
python src/prepare_assets.py
```

The model is saved under `artifacts/mmbert-base`. The downloader requests only the pinned PyTorch weight, tokenizer, config, and model card; it skips the duplicate safetensors weight. Large weight downloads use resumable HTTP Range chunks and are accepted only when the final size and SHA-256 match the pinned Hub LFS metadata. Dataset cards are retained under `artifacts/raw/source_cards`, and asset hashes and license metadata are recorded in `artifacts/assets_manifest.json`.

The XNLI Hub card does not declare a license at the pinned revision. Its manifest entry preserves that status rather than assigning a license. Read the captured source card and upstream terms before redistributing XNLI-derived files.

## Prepare data

```powershell
python src/prepare_data.py --profile pilot
python src/prepare_data.py --profile full
```

`pilot` writes the frozen input interfaces directly in `data/prepared/`; it contains at least 32 article-local full windows per Wiki length and at least 64 regular training questions per language for classification, QA, and retrieval. QA pilot data also includes 16 real 4K long-context examples per language when enough pinned Wiki background is available. `full` writes to `data/prepared/full/` so it cannot replace any pilot files or raw source records already used by the four-variant pre-run. Raw selections are profile-separated under `artifacts/raw/pilot/` and `artifacts/raw/full/`. Each profile has independent `manifest_{task}.json` files; run `python src/prepare_data.py --profile PROFILE --merge-manifest` after its task processes finish to create the combined `manifest.json`.

The Wiki training arrays `wiki_{length}.npy` contain only `train` rows, with shape `int32 [N, length]`, BOS and EOS, non-overlapping article-contiguous content windows, and no padding. Held-out rows are stored separately as `wiki_{length}_dev.npy` and `wiki_{length}_test.npy`. Each array has an adjacent JSONL sidecar (`wiki_{length}.jsonl`, `wiki_{length}_dev.jsonl`, or `wiki_{length}_test.jsonl`) that maps rows to source article, language, split, and chunk. An article is assigned to exactly one split by a deterministic hash, and all four variants use the same source windows.

Classification rows are in `classification_{train,dev,test}.jsonl`; QA rows are in `qa_{train,dev,test}.jsonl`. QA rows include `context_mask`, token-to-context `offset_mapping`, original context, question, normalized answers, and `source_answers_original`. If an answer offset is not exact, preparation leaves the answer text unchanged and shifts the offset only to a uniquely nearest exact occurrence. Training questions with no exact occurrence are skipped and recorded under `artifacts/raw/{profile}/qa/offset_rejected_*.jsonl`; unalignable development/public-evaluation questions remain in the question denominator, keep their original gold answers and all overflow windows, and use `answer_alignment_status=unalignable_eval` with span labels `-100`. Other overflow windows can also have `-100` when the aligned answer lies outside that window. Group QA features by `question_id` for question-level evaluation. Long training augmentation rows are also listed in `qa_long_augmentation.jsonl`; pilot uses 16 per language and full uses 10% of regular training questions per language. Their filler backgrounds come only from the Wiki `train` split. The `dev` questions are article-grouped from source training data; public SQuAD/CMRC validation is kept as `test`.

Retrieval training rows are in `retrieval_{train,dev,test}.jsonl`. Each row has token-list fields `query_ids`, `positive_ids`, and `negative_ids`, plus query/document source text and IDs. A stable hash assigns training documents a maximum token-prefix length of 512, 4096, or 8192 (90%, 8%, or 2%); evaluation passages use their first 8192 tokens. Reranking training groups use two deterministic MLDR negatives; evaluation groups use one fixed BM25 top-20 over the reduced corpus. The evaluation interface is `retrieval_corpus.jsonl` (`id`, `input_ids`, `language`) and `retrieval_queries_{dev,test}.jsonl` (`id`, `input_ids`, `language`, `relevant_doc_ids`). Corpus rows are split by language in `retrieval_corpus_{en,zh}.jsonl`. For the full profile, each language corpus includes known dev/test positives and SHA-256-ranked distractors from a bounded prefix of the pinned MLDR corpus; the manifest records the exact prefix read and actual selected counts.

Prepared data is a reproducible, tokenized subset, not a claim that the full upstream dataset snapshot has been downloaded. Raw selected source rows/documents are retained under `artifacts/raw/`. The pinned dataset cards and their hashes are recorded in `artifacts/assets_manifest.json`; licensing information is reported as stated in the Hub metadata, including when it is absent.

`tools/archive_prepared.py` includes files smaller than 96 MiB and records exclusions in `data/archive_manifest.json`. Large MLDR prepared files are excluded from that repository archive, so it does not restore every prepared interface; retain the separately approved MLDR bundles or rebuild the retrieval profile locally from the pinned source and cached prefixes.
