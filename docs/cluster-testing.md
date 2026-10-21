# Testing Cluster on several computers

Cluster runs one model across computers on the same network. Each computer runs a contiguous
range of the model's layers; every token's activation crosses each link once.

## On every computer

1. Install Cerebro (`Cerebro_0.1.0_x64-setup.exe`).
2. Download the **same model file** from the Model Hub on each (Cluster checks that the files
   match before anything runs). Start with `Qwen3-0.6B-Q8_0` for a first test, then a model too big
   for one machine.

## On each helper computer ("worker")

1. Open **Cluster** → **Share this computer** → pick the model → **Start sharing**.
2. Windows asks whether Cerebro may use private networks: **Allow**.
3. Note the 6-digit pairing code and the address shown (for example `192.168.1.34:50080`).

## On the computer you chat from

1. Open **Cluster** → **Use other computers** → **Look for computers** (or type the address).
2. For each worker: type its pairing code, and set its share (give more to computers with more
   free memory).
3. Pick the same model → **Load across N computers**. The table shows each link's round trip and
   speed.
4. Open **Chat** and talk to it. Watch the live panel (Ctrl .) for tokens per second.

## What to record for each setup

| Setup | Link | Round trip (ms) | Link MB/s | Tokens/s alone | Tokens/s split |
|---|---|---|---|---|---|

Links worth trying: Ethernet through a switch, Wi-Fi, and a direct USB4/Thunderbolt or USB
network cable between two computers.

## Known limits of this first version

- Every computer runs its layers on its processor (the graphics card joins in a later version).
- Models with recurrent layers (Qwen3.5 / Qwen3.8 DeltaNet) cannot be split yet.
- Images cannot be read by a split model yet.
- If a link drops mid-reply the reply stops; load again.
