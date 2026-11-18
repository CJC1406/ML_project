# Graph Theory

Each trip is a directed road multigraph. Nodes are row-index identities from `.nodes`; arcs are row-index identities from `.arcs`. A `MultiDiGraph` is required because parallel directed arcs are distinct candidates. Weak components describe undirected reachability; directed predecessor/successor checks govern legal transitions. Isolated nodes and invalid endpoints are validation findings, not silently removed data.
