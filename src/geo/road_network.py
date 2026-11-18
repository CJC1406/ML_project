import networkx as nx
import pandas as pd


def build_graph(nodes: pd.DataFrame, arcs: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(nodes)))
    for arc_id, row in arcs.reset_index(drop=True).iterrows():
        graph.add_edge(int(row.from_node_id), int(row.to_node_id), arc_id=int(arc_id))
    return graph


def graph_statistics(graph: nx.DiGraph) -> dict[str, int]:
    return {"nodes": graph.number_of_nodes(), "arcs": graph.number_of_edges(), "isolated_nodes": len(list(nx.isolates(graph))), "weak_components": nx.number_weakly_connected_components(graph)}


def neighbouring_nodes(graph: nx.DiGraph, node_id: int) -> list[int]:
    return sorted(graph.successors(node_id))
