# import yaml
# from pathlib import Path
# from utils.model_factory import build_hyper_grid, PRESETS, HYPER_GRID, BARS

# # Load coins list
# with open(r'C:\Users\jordd\OneDrive\Desktop\Code\StockBot\StockFactory2\config\coins.yaml') as f:
#     coins = yaml.safe_load(f)["coins"]

# print(f"Loaded {len(coins)} coins:")
# for c in coins:
#     print(" -", c["symbol"])

# print("\nTesting all (coin, bar, preset) combinations...")

# failures = []
# tested = 0

# for coin in coins:
#     symbol = coin["symbol"]
#     for bar in BARS:
#         for preset in PRESETS.keys():
#             cfg = {
#                 "symbol": symbol,
#                 "interval_minutes": bar,
#             }
#             try:
#                 grid = build_hyper_grid(cfg, preset=preset, bar=bar)
#                 keys = list(grid)
#                 combos = 1
#                 for v in grid.values():
#                     combos *= len(v)
#                 print(f"OK: {symbol} @ {bar}m [{preset}] grid has {combos} combos")
#             except Exception as e:
#                 print(f"FAIL: {symbol} @ {bar}m [{preset}] → {e}")
#                 failures.append((symbol, bar, preset, str(e)))
#             tested += 1

# print(f"\nTested {tested} (coin, bar, preset) combos")
# if failures:
#     print(f"\n{len(failures)} failures:")
#     for s, b, p, e in failures:
#         print(f" - {s}@{b}m [{p}]: {e}")
#     exit(1)
# # else:
#     print("\nALL GRID BUILDS OK ✅")



import yaml
from pathlib import Path
from utils.model_factory import build_hyper_grid, PRESETS, HYPER_GRID, BARS

# Load coins list
with open(r'C:\Users\jordd\OneDrive\Desktop\Code\StockBot\StockFactory2\config\coins.yaml') as f:
    coins = yaml.safe_load(f)["coins"]

print(f"Loaded {len(coins)} coins:")
for c in coins:
    print(" -", c["symbol"])

print("\nTesting all (coin, bar, preset) combinations...")

failures = []
tested = 0

for coin in coins:
    symbol = coin["symbol"]
    for bar in BARS:
        for preset in PRESETS.keys():
            cfg = {
                "symbol": symbol,
                "interval_minutes": bar,
            }
            try:
                grid = build_hyper_grid(cfg, preset=preset, bar=bar)
                keys = list(grid)
                combos = 1
                for v in grid.values():
                    combos *= len(v)
                print(f"OK: {symbol} @ {bar}m [{preset}] grid has {combos} combos")
            except Exception as e:
                print(f"FAIL: {symbol} @ {bar}m [{preset}] → {e}")
                failures.append((symbol, bar, preset, str(e)))
            tested += 1

print(f"\nTested {tested} (coin, bar, preset) combos")
if failures:
    print(f"\n{len(failures)} failures:")
    for s, b, p, e in failures:
        print(f" - {s}@{b}m [{p}]: {e}")
    exit(1)
# else:
    print("\nALL GRID BUILDS OK ✅")
