import networkx as nx
import pandas as pd


def build_graph(nodes: pd.DataFrame, arcs: pd.DataFrame) -> nx.MultiDiGraph:
    """Build a directed multigraph so parallel arcs and original IDs survive."""
    graph = nx.MultiDiGraph()
    node_ids = nodes["node_id"] if "node_id" in nodes else range(len(nodes))
    graph.add_nodes_from(int(x) for x in node_ids)
    for arc_id, row in arcs.reset_index(drop=True).iterrows():
        original_id = int(row.arc_id) if "arc_id" in arcs else int(arc_id)
        graph.add_edge(int(row.from_node_id), int(row.to_node_id), key=original_id, arc_id=original_id)
    return graph


def graph_statistics(graph: nx.DiGraph) -> dict[str, int]:
    return {"nodes": graph.number_of_nodes(), "arcs": graph.number_of_edges(), "isolated_nodes": len(list(nx.isolates(graph))), "weak_components": nx.number_weakly_connected_components(graph)}


def neighbouring_nodes(graph: nx.DiGraph, node_id: int) -> list[int]:
    return sorted(graph.successors(node_id))
