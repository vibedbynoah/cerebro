# Shared crates: reviewer checklist

`cerebro-bench`, `-config`, `-flags`, `-job`, `-probe`, `-store`, `-supervisor`, `-winfs` back the desktop app.

- Windows first: long paths (> 260 chars), paths with spaces (a credential-helper path with a space broke
  Git once), OneDrive-redirected folders, and files locked by a running exe (rename before rebuild).
- `cerebro-store` / `cerebro-config`: atomic writes (temp file + rename), no data loss on crash, schema
  changes migrate instead of discarding user data.
- `cerebro-job` / `cerebro-supervisor`: child processes are killed with their whole tree on cancel; no
  orphaned workers after the app closes.
- `cerebro-probe` / `cerebro-bench`: measurements state their conditions (GPU, build, context depth); never
  report one tok/s number without them.
