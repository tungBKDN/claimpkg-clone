def plot_entity_graph(data: dict):
    """
    Plot a directed graph of an entity and its direct connections.
    Supports multiple relations between the same nodes by merging labels.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.DiGraph()

    # map identity -> display name
    id_to_name = {}
    current = data["current_node"]
    id_to_name[current["identity"]] = current["properties"].get("name", str(current["identity"]))
    for dn in data["direct_node"]:
        id_to_name[dn["identity"]] = dn["properties"].get("name", str(dn["identity"]))

    # add nodes
    for nid, name in id_to_name.items():
        G.add_node(nid, label=name)

    # merge multiple relations between same start/end
    edge_labels = {}
    for rel in data["relations"]:
        start_id = rel["start"]
        end_id = rel["end"]
        key = (start_id, end_id)
        if key not in edge_labels:
            edge_labels[key] = []
        edge_labels[key].append(rel["relation_name"])
        G.add_edge(start_id, end_id)  # ensure edge exists

    # layout
    pos = nx.spring_layout(G, seed=42)

    # draw nodes
    nx.draw(
        G,
        pos,
        labels={nid: G.nodes[nid]["label"] for nid in G.nodes()},
        node_color="lightblue",
        node_size=2000,
        font_size=10,
        arrows=True,
        arrowstyle="->",
        arrowsize=15,
    )

    # draw edge labels (merged)
    merged_labels = {k: ", ".join(v) for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=merged_labels, font_size=8)

    plt.title(f"Entity Graph: {current['properties'].get('name', current['identity'])}")
    plt.axis("off")
    plt.show()

# plot_entity_graph(entity_data)