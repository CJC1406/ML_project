# Dataset Exploration Findings

**Inspection type:** read-only  
**Dataset inspected:** local training partition  
**Path:** configured locally only; raw data is not part of the repository  
**Models trained:** none

## Directory structure

The training data is organised as one directory per trip:

```text
train/
  00000002/
    00000002.track
    00000002.nodes
    00000002.arcs
    00000002.route
  ...
```

The inspected training partition contains 80 trip directories. Every directory is correctly paired by the six-digit trip identifier and contains exactly one file of each required type. There are no files directly in the `train` root and no missing or extra trip files were found.

## File schemas

Representative trips inspected: `00000002`, `00000034`, and `00000093`.

All representative files are headerless and whitespace-separated. No quoted fields or delimiters other than whitespace were observed.

| File | Observed columns | Interpretation |
|---|---:|---|
| `.track` | 3 | longitude, latitude, elapsed timestamp |
| `.nodes` | 2 | longitude, latitude; node ID is the zero-based row index |
| `.arcs` | 2 | from-node row index, to-node row index; arc ID is the zero-based row index |
| `.route` | 1 | chronological arc row-index sequence |

Track timestamps are numeric elapsed seconds. They begin at approximately 1 and are not Unix epoch timestamps. The normal interval is one second, but gaps occur. Node and arc identifiers are represented by row position rather than an explicit ID column; the canonical loader must preserve those row-index identities and must not renumber rows after parsing.

The inspected route entries were integer-valued and all referenced valid arc row indices for the three representative trips. This was also checked across all 80 training trips.

## Measured training-partition inventory

| Quantity | Measured value |
|---|---:|
| Trip directories | 80 |
| `.track` files | 80 |
| `.nodes` files | 80 |
| `.arcs` files | 80 |
| `.route` files | 80 |
| Total GPS observations | 208,417 |
| Total road nodes | 1,933,485 |
| Total road arcs | 3,876,904 |
| Total route entries | 31,336 |
| Trips with missing files | 0 |
| Trips with malformed track rows | 0 |
| Invalid node references | 0 |
| Invalid arc references in routes | 0 |
| Trips with duplicate GPS rows | 0 |

Per-trip ranges were: 361-19,313 GPS observations, 299-155,003 nodes, 649-307,392 arcs, and 59-2,266 route entries. These are descriptive measurements, not model results.

## Representative trip statistics

| Trip | Track | Nodes | Arcs | Route | GPS bounds (min lon, min lat, max lon, max lat) | Duration (s) | Max gap (s) | Weak components | Parallel directed pairs |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 00000002 | 7,243 | 7,936 | 16,139 | 262 | 10.999961, 47.658344, 11.050902, 47.695459 | 8,933.156 | 160 | 14 | 0 |
| 00000034 | 3,614 | 19,597 | 40,971 | 454 | 44.260728, 40.188342, 44.510744, 40.364683 | 3,855 | 15 | 23 | 4 |
| 00000093 | 1,211 | 45,161 | 85,122 | 590 | -3.840512, 43.349849, -3.703300, 43.444055 | 1,243 | 6 | 32 | 17 |

Across all trips, the median node count is 9,608, median arc count is 19,526.5, and median route length is 280. Every inspected graph had zero isolated nodes. Weakly connected component counts ranged from 1 to 421, so graph connectivity cannot be assumed globally.

## GPS timing and movement

The usual track interval is one second. Across trips, maximum gaps ranged from 1 to 153,064 seconds; 590 gaps across the partition exceeded ten seconds. Large gaps must be represented as input features or decoder conditions rather than silently dropped. Consecutive-point distances and speed estimates were calculated using the Haversine formula and positive time deltas; outliers were reported, not removed.

Representative median consecutive distances were 7.18 m (`00000002`), 12.61 m (`00000034`), and 17.27 m (`00000093`). These values matter for candidate radius selection and should not be treated as universal thresholds without checking the full distribution.

## Road-network observations

The road networks are directed and contain parallel directed source/destination pairs in at least the representative trips `00000034` and `00000093`. A simple `DiGraph` would collapse these arcs and lose label identity, so `MultiDiGraph` and explicit arc-row IDs are required. In/out-degree distributions are bounded and non-trivial; representative maximum in/out degrees were 5/5, 5/6, and 6/6 respectively.

## Validation status

The read-only inspection found no missing files, empty representative files, malformed rows, invalid coordinates, invalid timestamps, out-of-order timestamps, duplicate GPS observations, invalid node references, or invalid route references in the training partition. The existing pandas/networkx validator could not be executed in this environment because those dependencies are not installed locally; the independent standard-library inspection produced the measurements above.

## Partition status and leakage boundary

The inspected directory is the 80-trip training partition specified by the guidelines. Student-testing and evaluator-testing partitions were not inferred from this directory and were not accessed. Evaluator labels were not used. The absolute source path remains local and is intentionally absent from tracked configuration.

## Visualisation status

No Folium maps were generated in this inspection because Folium, Shapely, pandas, and NetworkX are not installed in the current environment. Generating placeholder or synthetic maps would violate the project rules. Once dependencies are installed, maps should be generated from the actual trip data and stored under ignored `data/processed/` or a private report location.

## Why these findings matter

- Headerless row-index IDs make row order part of the data contract.
- Numeric elapsed seconds must not be parsed as nanoseconds or Unix time.
- Parallel directed arcs require a multigraph and stable arc identity.
- Disconnected components mean transitions need explicit network checks.
- Large GPS gaps can invalidate naive speed, bearing, and transition features.
- The wide variation in network size requires bounded, indexed candidate queries.
- The clean 80-trip inventory establishes the training boundary without exposing evaluator data.

## Unresolved questions

- Exact formats and locations of the student-testing and evaluator-testing partitions remain to be inspected separately.
- The route-to-GPS alignment is not one-to-one by row count; candidate recall must not be computed until the reference alignment method is established.
- A region-specific projected CRS should be selected before final metric spatial operations; the three representative trips span different geographic regions.
- Folium/geometry outputs and canonical-loader validation remain to be rerun after installing declared dependencies.
