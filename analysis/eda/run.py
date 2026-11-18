"""PathSnap EDA. Install numpy matplotlib networkx folium, then run this file.

Defaults to the supplied Desktop archive and writes results under outputs/eda.
Override with --zip PATH --out DIRECTORY. Use --trip TRAIN_ID to select a map.
"""
import argparse
import csv
import math
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np

SCHEMA = {"track": ("longitude latitude time_s", float), "nodes": ("longitude latitude", float),
          "arcs": ("from_node_id to_node_id", int), "route": ("arc_id", int)}
PARTITIONS = {"train": 80, "student_test": 10}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def parse(z, member, kind):
    columns, dtype = SCHEMA[kind]
    rows, issues = [], Counter()
    with z.open(member) as stream:
        for raw in stream:
            fields = raw.split()
            try:
                if len(fields) != len(columns.split()):
                    raise ValueError("missing_fields" if len(fields) < len(columns.split()) else "extra_fields")
                if any(f.lower() in (b"na", b"nan", b"null", b"none", b"?") for f in fields):
                    raise ValueError("missing_value")
                row = tuple(map(dtype, fields))
                if dtype is float and not all(map(math.isfinite, row)):
                    raise ValueError("nonfinite")
                if kind in ("nodes", "track") and not (-180 <= row[0] <= 180 and -90 <= row[1] <= 90):
                    raise ValueError("coordinate_range")
            except (ValueError, OverflowError) as exc:
                reason = str(exc)
                issues[reason if reason in ("missing_fields", "extra_fields", "missing_value",
                                            "nonfinite", "coordinate_range") else "invalid_numeric"] += 1
                row = None
            rows.append(row)  # Preserve zero-based IDs, even for malformed records.
    return rows, issues


def km(a, b):
    lon1, lat1, lon2, lat2 = map(math.radians, (*a[:2], *b[:2]))
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 12742.0176 * math.asin(math.sqrt(min(1, h)))


def analyse(data):
    track, nodes, arcs, route = (data[k] for k in ("track", "nodes", "arcs", "route"))
    gps = [p for p in track if p]
    pairs = [(a,b) for a,b in zip(track, track[1:]) if a and b]
    gaps = np.array([b[2]-a[2] for a,b in pairs])
    graph = nx.DiGraph()
    graph.add_nodes_from(i for i,p in enumerate(nodes) if p)
    edges = {i:e for i,e in enumerate(arcs) if e and all(v in graph for v in e)}
    graph.add_edges_from(edges.values())
    sequence = [edges.get(r[0]) if r else None for r in route]
    components = list(nx.weakly_connected_components(graph))
    stats = dict(valid_gps=len(gps), invalid_gps=len(track)-len(gps), valid_time_steps=len(gaps),
                 one_second_steps=int(sum(gaps == 1)), duplicate_adjacent_times=int(sum(gaps == 0)),
                 backward_steps=int(sum(gaps < 0)), gaps_over_1s=int(sum(gaps > 1)),
                 gaps_over_10s=int(sum(gaps > 10)), max_gap_s=max(gaps, default=None),
                 median_gap_s=np.median(gaps) if len(gaps) else None,
                 observed_gps_km=sum(km(a,b) for a,b in pairs) if pairs else None,
                 valid_nodes=len(graph), usable_arcs=len(edges), unusable_arcs=len(arcs)-len(edges),
                 isolated_nodes=nx.number_of_isolates(graph), weak_components=len(components),
                 largest_component_nodes=max(map(len,components), default=0),
                 strong_components=nx.number_strongly_connected_components(graph),
                 reference_steps=len(route), unusable_reference_steps=sequence.count(None),
                 reference_breaks=sum(a[1]!=b[0] for a,b in zip(sequence,sequence[1:]) if a and b))
    for i,axis in enumerate(("lon","lat")):
        stats["min_"+axis] = min((p[i] for p in gps), default=None)
        stats["max_"+axis] = max((p[i] for p in gps), default=None)
    return stats, gaps, edges, sequence


def draw(out, trip, data, edges, seq):
    nodes = data["nodes"]
    gps = [p[:2] for p in data["track"] if p]
    roads = [[nodes[a],nodes[b]] for a,b in edges.values()]
    route = [[nodes[e[0]],nodes[e[1]]] for e in seq if e]
    fig, ax = plt.subplots()
    for lines,color,width,label in [(roads,"silver",0.4,"Road network"),(route,"green",2,"Reference route")]:
        ax.add_collection(LineCollection(lines,colors=color,linewidths=width,label=label))
    ax.scatter(*zip(*gps),s=3,label="GPS",color="royalblue")
    ax.autoscale()
    ax.set(xlabel="Longitude (degrees)",ylabel="Latitude (degrees)",title=f"Training trip {trip}")
    ax.legend()
    fig.savefig(out/"training_trip.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    points = [(lat,lon) for lon,lat in gps]
    try:
        import folium
    except ImportError:
        # Keep one map implementation. Pyodide runs the same Folium function in the browser.
        import inspect
        import base64
        import gzip
        payload = base64.b64encode(gzip.compress(json.dumps([points,roads,route]).encode())).decode()
        code = inspect.getsource(build_map) + "\nimport folium, json, gzip, base64\n" + (
            "points, roads, route = json.loads(gzip.decompress(base64.b64decode(payload)))\n"
            "build_map(folium, points, roads, route).get_root().render()"
        )
        html = """<!doctype html><meta charset=utf-8><title>PathSnap training trip</title>
<style>body{font:16px system-ui;margin:0}iframe{border:0;width:100%;height:98vh}</style>
<p id="status">Loading the training map. Internet access is required.</p>
<iframe id="map" hidden title="Training GPS observations, road network and reference route"></iframe>
<script src="https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js"></script><script>
async function start(){
  const py = await loadPyodide();
  await py.loadPackage('micropip');
  await py.runPythonAsync("import micropip\\nawait micropip.install('folium==0.19.7')");
  py.globals.set('payload', PAYLOAD);
  const frame = document.getElementById('map');
  frame.srcdoc = await py.runPythonAsync(CODE);
  frame.hidden = false;
  document.getElementById('status').hidden = true;
}
start().catch(e => document.getElementById('status').textContent =
  'Map dependencies could not load. Install folium locally and rerun the analysis. ' + e);
</script>"""
        (out/"training_trip.html").write_text(html.replace("PAYLOAD",json.dumps(payload)).replace("CODE",json.dumps(code)),encoding="utf-8")
        return "Browser-rendered Folium map generated (requires internet on first open)."
    build_map(folium,points,roads,route).save(str(out/"training_trip.html"))
    return "Interactive Folium map generated."


def build_map(folium, points, roads, route):
    m = folium.Map(location=points[0],prefer_canvas=True)
    for lines,color,name,width in [(roads,"gray","Road network",1),(route,"green","Reference route",4)]:
        if lines:
            folium.GeoJson({"type":"MultiLineString","coordinates":lines},name=name,
                           style_function=lambda _,c=color,w=width:dict(color=c,weight=w)).add_to(m)
    layer = folium.FeatureGroup(name="GPS observations").add_to(m)
    for point in points:
        folium.CircleMarker(point,radius=2,color="blue",weight=1,fill=True).add_to(layer)
    m.fit_bounds([[min(p[i] for p in points) for i in (0,1)],
                  [max(p[i] for p in points) for i in (0,1)]])
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip",type=Path,default=Path.home()/"Desktop/ml_hackathon/Student folder.zip")
    parser.add_argument("--out",type=Path,default=Path("docs/eda"))
    parser.add_argument("--trip",help="Training trip ID including leading zeroes")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    inventory, trips, summaries = [], [], []
    groups = defaultdict(lambda: defaultdict(list))
    intervals = defaultdict(list)
    map_status = "No eligible training trip selected."
    mapped = False
    with ZipFile(args.zip) as z:
        for item in z.infolist():
            path = PurePosixPath(item.filename)
            for i,part in enumerate(path.parts[:-1]):
                if part in PARTITIONS:
                    key = part,path.parts[i+1]
                    groups[key]
                    if not item.is_dir():
                        groups[key][path.suffix[1:]].append(item)
                    break
        if not groups:
            parser.error("No train or student_test trip folders found.")
        if args.trip and ("train",args.trip) not in groups:
            parser.error("Requested training trip absent.")
        for (part,trip),files in sorted(groups.items()):
            data = {}
            for kind,(columns,dtype) in SCHEMA.items():
                members = files.get(kind,[])
                rows,issues = [],Counter()
                if len(members)==1:
                    rows,issues = parse(z,members[0],kind)
                data[kind] = rows
                inventory.append(dict(partition=part,trip=trip,kind=kind,files=len(members),
                                      names=" | ".join(x.filename for x in members),
                                      bytes=sum(x.file_size for x in members),rows=len(rows),
                                      columns=columns,validated_type=dtype.__name__,invalid_rows=sum(issues.values()),
                                      missing_fields=issues["missing_fields"],missing_values=issues["missing_value"],
                                      issues=str(dict(issues))))
            stats,gaps,edges,seq = analyse(data)
            trips.append(dict(partition=part,trip=trip,**stats))
            intervals[part].extend(gaps.tolist())
            if part=="train" and not mapped and (args.trip is None or args.trip==trip) and stats["valid_gps"]:
                map_status = draw(out,trip,data,edges,seq)
                mapped = True
    write_csv(out/"file_inventory.csv",inventory)
    write_csv(out/"trip_quality.csv",trips)
    fig,axes = plt.subplots(2,2,figsize=(10,7))
    for col,(part,expected) in enumerate(PARTITIONS.items()):
        rows = [r for r in trips if r["partition"]==part]
        g = np.array(intervals[part])
        lengths = [r["observed_gps_km"] for r in rows if r["observed_gps_km"] is not None]
        summaries.append(dict(partition=part,expected_trips=expected,trips=len(rows),
                              gps_rows=sum(r["valid_gps"]+r["invalid_gps"] for r in rows),
                              invalid_gps=sum(r["invalid_gps"] for r in rows),
                              one_second_pct=100*np.mean(g==1) if len(g) else None,
                              max_gap_s=max(g,default=None),median_observed_km=np.median(lengths) if lengths else None))
        axes[0,col].hist(lengths,bins=15)
        axes[0,col].set(title=part,xlabel="Observed GPS distance (km)",ylabel="Trips")
        counts = [sum(g<=0),sum((g>0)&(g<1)),sum(g==1),sum((g>1)&(g<=10)),sum(g>10)]
        axes[1,col].bar(["<=0","(0,1)","1","(1,10]",">10"],counts)
        axes[1,col].set(xlabel="Adjacent time interval (s)",ylabel="Intervals (symlog)",yscale="symlog")
    fig.tight_layout()
    fig.savefig(out/"partition_distributions.png",dpi=150)
    plt.close(fig)
    write_csv(out/"partition_summary.csv",summaries)
    overlap = sorted({t for p,t in groups if p=="train"} & {t for p,t in groups if p=="student_test"})
    notes = ["Only train and student_test inspected.",f"Trip IDs shared between partitions: {overlap}",
             "Raw row order preserved. Missing or ambiguous file types are flagged by files != 1 in inventory.",
             "GPS distance sums spherical great-circle steps (mean Earth radius 6371.0088 km). Noise and gaps affect this estimate.",
             "Malformed rows interrupt adjacency. A gap above 10 s is a descriptive review flag.",
             "Arc direction follows the two endpoint columns. Reference breaks count adjacent usable arcs with mismatched endpoints.",
             "Reference arcs describe traversal order; no per-observation labels inferred or independently verified.",map_status]
    for s in summaries:
        ranked = sorted((r for r in trips if r["partition"]==s["partition"] and r["max_gap_s"] is not None),
                        key=lambda r:r["max_gap_s"],reverse=True)
        notes.append(s["partition"]+" largest gaps: "+", ".join(f"{r['trip']} ({r['max_gap_s']:g} s)" for r in ranked[:3]))
    (out/"quality_notes.txt").write_text("\n".join(notes)+"\n",encoding="utf-8")
    print(*summaries,map_status,sep="\n")


if __name__=="__main__":
    main()
