# Hugging Face one-click downloads — research notes (2026-09-22)

How a desktop app (Cerebro, Rust + spawned llama.cpp) should implement LM Studio-style
one-click model downloads from the Hugging Face Hub. All facts verified Sep 2026 from
primary sources linked inline. Focus: what changed since 2026-09-01, plus durable design.

## 1. Hub HTTP API: search, model info, file listing

- Search/list: `GET /api/models` with `search`, `author`, `filter` (tags), `sort`
  (`downloads`/`likes`/`lastModified`), `direction`, `limit`, `full`, `config`. Paginated
  via `Link` header. Documented at https://huggingface.co/docs/huggingface_hub/en/guides/search
  and the OpenAPI spec https://huggingface.co/.well-known/openapi.md
  (docs index moved to the OpenAPI playground: https://huggingface.co/docs/hub/en/api).
- Python equivalent is `HfApi.list_models()`; CLI equivalent is `hf models ls --search llama
  --sort downloads --limit 5`; file listing `hf models ls <repo> -R --tree -h`.
  See https://huggingface.co/docs/huggingface_hub/en/guides/cli
- Per-file metadata comes from the Hub "paths-info" / tree endpoints; `huggingface.js`
  `downloadFileToCacheDir` resolves etag via `pathsInfo({paths:[path], revision, expand:true})`
  reading `info.lfs.oid`, `info.xetHash`, or `info.oid`:
  https://github.com/huggingface/huggingface.js/blob/main/packages/hub/src/lib/download-file-to-cache-dir.ts
- `snapshot_download` pins one commit hash (`repo_info.sha`), stores `refs/<branch>`, and
  caches the per-commit file list in `trees/<commit>.json` so repeat downloads cost one
  network call (branch→hash resolve). Offline + incomplete snapshot raises
  `IncompleteSnapshotError` (with `snapshot_path` attr):
  https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
- GGUF-quant repos: `repo_info.siblings` (rfilename/size/lfs/xet) is how apps enumerate
  `Q4_K_M` etc. variants; LM Studio additionally curates a catalog. `allow_patterns` /
  `ignore_patterns` globs filter snapshot downloads:
  https://huggingface.co/docs/huggingface_hub/en/guides/download

## 2. Resolve URLs + resumable Range downloads (the LFS path)

- Canonical file URL: `https://huggingface.co/{ns}/{repo}/resolve/{rev}/{path}`
  (`hf_hub_url()` builds it). Files > few MB 302-redirect to CloudFront/CDN; client must
  follow redirects. HEAD on the resolve URL returns `ETag` (= git-sha1 for git files,
  sha256 for LFS files), `X-Linked-Size`, `X-Repo-Commit`:
  https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download
  and impl https://github.com/huggingface/huggingface_hub/blob/v1.22.0/src/huggingface_hub/file_download.py
- Resume: `http_get` sends `Range: bytes=<resume_size>-`, writes to `blobs/<etag>.incomplete`,
  atomically renames on completion, guards concurrent same-file downloads with a file lock
  (`WeakFileLock`, `.locks/` dir), retries transient errors ~5x, checks `expected_size`:
  same file_download.py source above.
- Known edge (May 2026, PR #4209): CloudFront may ignore `Range` and return 200 (gzip+chunked,
  no `Content-Length` for small text files); client must truncate + roll back progress:
  https://github.com/huggingface/huggingface_hub/pull/4209
- `hf_transfer` (parallel-range Rust downloader, `HF_HUB_ENABLE_HF_TRANSFER=1`) is
  **deprecated** — Xet backend replaces it; it also never supported resume/proxies:
  https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables

## 3. Xet storage backend — why plain HTTP clients stall at 0 bytes

- Since `huggingface_hub` 0.30–0.32 (2025), large files live on Xet, not LFS. With `hf_xet`
  installed (bundled since 0.32.0) downloads go: auth → CAS `GET /v2/reconstructions/{file_id}`
  → signed multi-range xorb fetches → chunk reassembly:
  https://huggingface.co/docs/xet/download-protocol and https://huggingface.co/docs/xet/api
- Critical interop trap: `GET <resolve-URL>` returns an `X-Xet-Hash` header; following the
  302 downloads via the legacy LFS-compat bridge
  (`cas-bridge.xethub.hf.co`), NOT the Xet protocol — and the bridge can stall/hang at
  0 bytes (matches the observed `hf download` 0-byte stall on this machine). Doc:
  https://github.com/huggingface/hub-docs/blob/main/docs/xet/file-id.md
- CAS endpoint is discovered per-repo via `/xet-read-token` (`casUrl` =
  `https://cas-server.xethub.hf.co/`, plus xorb buckets `transfer.xethub.hf.co`,
  `us.aws.cdn.hf.co` / `us.gcp.cdn.hf.co`); short-lived signed URLs must not be cached:
  https://github.com/huggingface/huggingface_hub/issues/4475
- `HF_HUB_DISABLE_XET=1` forces the legacy path; `HF_XET_CACHE` (`$HF_HOME/xet`) holds
  chunk/shard caches; `HF_XET_HIGH_PERFORMANCE=1` needs ~64 GB RAM — unsuitable for the
  32 GB reference laptop: https://huggingface.co/docs/hub/en/xet/using-xet-storage
- Reference impl is Rust `xet-core` (`hf-xet`, `xet-client`, `xet-data` crates):
  https://github.com/huggingface/xet-core
- Resume caveat (Jun 2025 forum): Xet-enabled repos can restart from 0% after interrupt;
  Xet resume differs from LFS `.incomplete` resume:
  https://discuss.huggingface.co/t/strange-problem-with-huggingface-cli/160083
- Implication for Cerebro: a pure-`reqwest` Range downloader works for small/git files but
  is second-class for multi-GB weights. Options: (a) depend on `hf-xet`/`xet-core` crates,
  (b) shell to `hf download`/`hfrs`, (c) implement CAS reconstruction (spec is open).

## 4. Rust client: `hf-hub` crate 1.0.0 (Jul 2026)

- Official Rust equivalent of `huggingface_hub`: `HFClient` (async) + `HFClientSync`
  (`blocking` feature), cache-compatible with Python, `HFError::RateLimited` (429),
  Xet high-performance transfers, `snapshot_download` with allow/ignore globs, wasm support:
  https://crates.io/crates/hf-hub and https://github.com/huggingface/hf-hub
- Download-builder semantics (range = Rust `start..end`, ETag/`If-None-Match`, `.no_exist`
  markers, transient-error→cache fallback): https://docs.rs/hf-hub/latest/src/hf_hub/repository/download.rs.html
- Ships an `hfrs` CLI (`cargo install --git … hfrs`). Token order: explicit → `HF_TOKEN` →
  token file; `HF_ENDPOINT` override supported (mirror-friendly).

## 5. Gated models + tokens

- Gated repos require login + "Agree" (username+email shared with authors); auto vs manual
  approval; request flow is browser-only, downloads need a token:
  https://huggingface.co/docs/hub/en/models-gated
- Tokens: fine-grained (scoped, preferred for shipping), `read` (download private/gated),
  `write`; pass as `Authorization: Bearer hf_…` or `?token=`; never bake into binaries:
  https://huggingface.co/docs/hub/en/security-tokens
- `hf auth login` (browser device flow or `--token`), token at `$HF_HOME/token`:
  https://huggingface.co/docs/huggingface_hub/en/guides/cli
- Ollama precedent for private HF GGUFs: per-machine SSH key (`~/.ollama/id_ed25519.pub`
  registered in HF settings) — no bearer token in app:
  https://huggingface.co/docs/hub/ollama

## 6. Rate limits (Sep 2025 tiers, still current doc)

- Three buckets over 5-min windows: Hub APIs / **Resolvers** (`/resolve/…`, highest limits)
  / Pages. Free user: 1,000 API + 5,000 resolver reqs; anon (per IP): 500 + 3,000:
  https://huggingface.co/docs/hub/en/rate-limits
- Exceed → 429 with IETF `RateLimit` / `RateLimit-Policy` headers; `huggingface_hub` ≥1.2.0
  auto-waits-until-reset on 429. Guidance: always send `HF_TOKEN` downstream, prefer
  resolver calls over API calls, back off with jitter. Ollama users hit 429 on unauthenticated
  `hf.co` pulls (0.14.x): https://github.com/ollama/ollama/issues/7240

## 7. Checksums

- Cache identity IS the checksum: `blobs/<etag>` where etag = LFS sha256 (or git-sha1);
  `snapshots/<commit>/<path>` symlinks to blobs:
  https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
- `hf cache verify <repo> [--revision …] [--local-dir …]` (added Oct 2025, PR #3461)
  recomputes sha256 (LFS) / git-sha1 and reports mismatches:
  https://github.com/huggingface/huggingface_hub/commit/80025cdf5c3ea7e0739f1233e6b510ff802a0a0c
- Deliberately NO on-download hash check (full re-read of GBs too costly; size check only) —
  Dec 2025 decision after Xet corruption reports:
  https://github.com/huggingface/huggingface_hub/issues/3643
  Cerebro should offer post-download verify as an explicit quality-gate step.

## 8. How LM Studio / Jan / Ollama do it

- **LM Studio**: curated catalog + variant picker (Q3_K_S…Q8), search accepts keywords or full
  HF URLs, nested-subfolder repos supported (0.3.9, Jan 2025), optional HF proxy setting,
  `~/.lmstudio/models/<publisher>/<model>/` layout preserving HF structure, `lms get <url>`
  CLI, `POST /api/v1/models/download` REST:
  https://lmstudio.ai/docs/app/basics/download-model ,
  https://lmstudio.ai/blog/lmstudio-v0.3.9 ,
  https://huggingface.co/docs/hub/en/lmstudio ,
  https://lmstudio.ai/docs/app/advanced/import-model ,
  https://github.com/lmstudio-ai/lms/blob/9902c3a0/src/subcommands/get.ts
- **Jan** (open source, Tauri+Rust): download manager in Rust backend (single-file +
  whole-HF-repo, progress via Tauri events, cancel, `HF_TOKEN` setting, path-traversal
  sanitization) — merged May 2025 for the llama.cpp extension:
  https://github.com/menloresearch/jan/pull/4933 ; models stored as plain `.gguf` files.
- **Ollama**: OCI-style registry (`registry.ollama.ai`): manifest JSON → content-addressed
  `blobs/sha256-<digest>`; chunked parallel pull with per-chunk resume (`-partial-N` state),
  NDJSON progress; `hf.co/<user>/<repo>[:quant]` bridge runs any of ~45k HF GGUFs (server-side
  manifest synthesis, requires GGUF/llama.cpp-compatible):
  https://github.com/ollama/ollama/blob/e434a938/server/images.go ,
  https://huggingface.co/docs/hub/ollama ,
  https://github.com/ollama/ollama/issues/8632 (CDN stall precedent)

## 9. Mirrors

- `HF_ENDPOINT` (read at import/startup) redirects all Hub traffic; community mirror
  `https://hf-mirror.com` documents `export HF_ENDPOINT=https://hf-mirror.com`:
  https://hf-mirror.com/ ; maintainer-confirmed pattern:
  https://github.com/huggingface/huggingface_hub/issues/2830
- Cerebro (offline-first, no mirrors by default) should still accept an endpoint override
  for lab/proxy use; never hardcode third-party mirrors.

## 10. Licence display requirements

- Licence is model-card YAML: `license:` (+ `license_name`/`license_link` for custom) —
  surfaced on the model page and filterable: https://huggingface.co/docs/hub/en/model-cards
- One-click UX must show `license` (+ `gated` state) BEFORE download; fetch via
  `hf models card <repo> --metadata [--format json]`:
  https://huggingface.co/docs/huggingface_hub/en/guides/cli
- Gated "Agree" text (`extra_gated_*` fields) is part of the licence consent surface:
  https://huggingface.co/docs/hub/en/models-gated

## 11. Recommended Cerebro design (Windows-first, offline, Rust)

1. **Search**: `GET {endpoint}/api/models?search=…&filter=gguf&sort=downloads&limit=…`
   (+ `Link` pagination), custom `User-Agent: Cerebro/<ver>`. Cache results; respect 429
   headers with backoff. No network except opt-in update check — gate the whole feature
   behind explicit user action + token entry.
2. **Variant pick**: `GET /api/models/{id}` (siblings: name/size/lfs-sha256/xet flag) →
   filter `*.gguf` → rank by fit (VRAM 8 GB: prefer Q4_K_M of ≤8B; show est. RAM/VRAM).
   Pin `revision` = commit hash; show licence + gated badge first.
3. **Download**: prefer Xet-capable path (depend on `hf-hub` 1.x/`xet-core` or shell
   `hf download --local-dir`); else Range-resume GET to `.part` + atomic rename,
   `Range` re-request on 200-without-`Content-Range`, `HF_HUB_DOWNLOAD_TIMEOUT`-style
   generous timeouts, token from OS credential store (never disk-plaintext).
4. **Verify**: size check always; optional `sha256` vs LFS etag post-pass (quality gate).
5. **Layout**: `models/<publisher>/<repo>/<rev-short>/<file>.gguf` + sidecar JSON
   (repo, rev, etag, size, licence, URL, timestamp) — LM Studio-like, offline-readable.
