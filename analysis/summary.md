# Connect4 Battle Summary

## Per-run table

| file | iters/move | games | LLM wins | Pure wins | draws | LLM win rate | LLM non-draw win rate | LLM calls | cache entries | cache hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| output5.txt | 50 | 50 | 41 | 9 | 0 | 82.00% | 82.00% | 256 | 34622 | 3038 |
| output4.txt | 100 | 50 | 36 | 14 | 0 | 72.00% | 72.00% | 256 | 79045 | 7497 |
| output2.txt | 200 | 50 | 35 | 15 | 0 | 70.00% | 70.00% | 256 | 147961 | 19174 |
| output3.txt | 400 | 50 | 38 | 12 | 0 | 76.00% | 76.00% | 256 | 298440 | 47504 |
| output.txt | 800 | 50 | 32 | 16 | 2 | 64.00% | 66.67% | 256 | 553605 | 105270 |

## Aggregated by iterations

| iters/move | runs | games | LLM wins | Pure wins | draws | LLM win rate | LLM non-draw win rate | LLM first win rate | LLM second win rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1 | 50 | 41 | 9 | 0 | 82.00% | 82.00% | 96.00% | 68.00% |
| 100 | 1 | 50 | 36 | 14 | 0 | 72.00% | 72.00% | 84.00% | 60.00% |
| 200 | 1 | 50 | 35 | 15 | 0 | 70.00% | 70.00% | 72.00% | 68.00% |
| 400 | 1 | 50 | 38 | 12 | 0 | 76.00% | 76.00% | 68.00% | 84.00% |
| 800 | 1 | 50 | 32 | 16 | 2 | 64.00% | 66.67% | 68.00% | 60.00% |

## Notes

- `LLM win rate` uses all games (draws count in denominator).
- `LLM non-draw win rate` excludes draws.
- `LLM first/second win rate` tracks side bias.
