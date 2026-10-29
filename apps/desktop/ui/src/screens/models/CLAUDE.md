# Models screen: reviewer checklist

- Load, Cancel (while loading) and Unload work at any moment from the library row, the drawer and the
  loaded list; a cancelled load says "Loading stopped", not an error.
- A file that is one part of a model (a split part, or DeepSeek's shared Engram table without its main
  weights) shows "Parts missing" with what to download; architectures the engine cannot run show
  "Not supported yet". Neither may be silently hidden.
- Quant and publisher appear in every label; two copies of the same model must be distinguishable.
- Big downloads (100 GB+) are allowed with a clear warning about disk space, never blocked; deletes need
  confirmation or Undo.
- The Benchmark lab row (monitor/BenchmarkLab.tsx) keeps labels, controls and hints aligned at every width.
