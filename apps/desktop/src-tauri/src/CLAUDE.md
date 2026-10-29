# Backend (Tauri commands, engine bridge): reviewer checklist

## Engine bridge (`engine.rs`)
- One worker thread runs every engine job in order (`Job::Load`, `Chat`, `Unload`, `Benchmark`, ...).
  A job that never checks `cancel` / `cerebro_engine::STOP` blocks everything queued behind it: check new
  jobs for stop points. `Engine::unload` sets both flags before queueing; keep that true.
- `load()` -> `with_llama` (AMD/Intel graphics only; NVIDIA always runs Cerebro's engine) -> `assemble` ->
  `place`. A placement or plan call that assumes generic layers must tolerate `model.ds41` (DeepSeek-V4.1 has
  no generic layers; it runs on the processor through `cerebro_engine::ds41`).
- Library scan (`describe`, `walk`): quant names come from the file name first (`quant_from_name`), then
  `general.file_type`; split parts and shared Engram files must not show up as separate models; a part
  missing its siblings sets `missing`.
- Chat (`run_chat`): the prompt is rendered with `cerebro_engine::ChatFormat`; tokens/s and where the reply
  ran must reach the UI after every reply. Look for any path that drops them.

## Security (treat each as high severity)
- `open_external`: https only, no shell. `local_api.rs`, `mcp_server.rs`, `tunnel.rs`, `remote*`: bind to
  127.0.0.1 unless the user turned on LAN access; tokens never logged or echoed in full.
- `secret.rs` / `cloud.rs`: API keys stay in the OS store; never written to logs, configs or crash reports.
- `agent_tools.rs`, `builtin_tools.rs`, `toolcall.rs`: path traversal outside the allowed workspace,
  symlink escapes, command injection, unbounded output read into memory, and tool-call JSON that fails to
  parse (must become a readable error for the model, not a panic).
- `hub.rs`: downloads verify size, resume safely, never write outside Cerebro's models folder.

## Speed
- Model loads: no full-file reads or hashes on the UI path; the library scan must stay fast with dozens
  of multi-GB files (no full-file hashing; `sha256` is a path/size/mtime key on purpose).
- Anything polled by the UI (status, monitor) must be cheap and lock-light.
