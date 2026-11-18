# PathSnap dataset exploration

From the repository root, run:

```powershell
python -m pip install -r analysis/eda/requirements.txt
python analysis/eda/run.py --zip "C:\Users\dabbu\Desktop\ml_hackathon\Student folder.zip" --out docs/eda
```

The archived data is read directly. The analysis covers all 80 training and 10 student-testing trips. Each contains exactly one track, node, arc and route file (360 files). Trip IDs are disjoint across the two partitions. The evaluator partition is reserved for the final evaluation.

| Measurement | Training | Student testing |
|---|---:|---:|
| GPS readings | 208,417 | 19,160 |
| Readings per trip, min-max | 361-19,313 | 262-3,958 |
| Time steps exactly 1 s apart | 96.50% | 98.62% |
| Time steps exceeding 10 s | 590 | 27 |
| Largest interval | 153,064 s | 210 s |
| Median observed GPS distance per trip | 24.07 km | 20.94 km |
| Nodes per trip, min-max | 299-155,003 | 1,770-26,635 |
| Arcs per trip, min-max | 649-307,392 | 3,577-56,027 |
| Trips with multiple weakly connected components | 79 | 10 |
| Trips with a directed reference-route break | 10 | 1 |

## Interpretation

All four file types passed the expected width, numeric parsing and range checks. All arc endpoints and route IDs resolved. Timestamps increased strictly within every trip. Nevertheless, trip 00000090 contains a 42 h 31 min 4 s gap. A velocity estimate across that interval would describe only the displacement between its endpoints. Flag large gaps before deriving motion features.

No nodes were isolated, yet 89 trip networks contained multiple weak components. Nodes can have neighbours inside their own component while remaining unreachable from another component. Even a pair of geographically close road candidates can therefore have no connecting road path.

There were 13 directed reference breaks in training and one in student testing. A break means the preceding arc's destination differs from the following arc's origin. It is a diagnostic flag. The reference path alone does not reveal which arc corresponds to a particular GPS timestamp.

The command above generates the full local outputs, including maps and coordinate bounds. The committed trip table omits coordinate bounds.

## Outputs and assumptions

- `file_inventory.csv` gives filenames, uncompressed sizes, expected column names and numeric types, observed row counts, and validation issues. The inputs are headerless and whitespace-delimited. Node and arc IDs are zero-based row positions.
- `trip_quality.csv` gives trip IDs, coordinate bounds, timing, network connectivity, route validity and observed GPS distances. Preserve trip IDs as text when opening the CSV in a spreadsheet.
- `partition_summary.csv` and `partition_distributions.png` summarize the two partitions separately. Interval percentages pool adjacent observations within each partition. Median distances weight trips equally.
- Distance is the sum of spherical great-circle distances between consecutive valid GPS readings. GPS noise, indirect travel, and long gaps affect the estimate. The reported distance is an observed trajectory statistic.
- `training_trip.png` and `training_trip.html` show training trip 00000001 by default. Grey lines show the network, green lines show the supplied reference, and blue points show observations. Use the HTML layer controls to inspect overlaps. The optional `--trip` argument selects another training trip.
- `quality_notes.txt` records thresholds, the largest gaps and map generation mode. Ten seconds is a descriptive review threshold. No measurements were deleted or imputed.
