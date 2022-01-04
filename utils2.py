

def overlap(s1, e1, s2, e2):
    return s2 <= s1 <= e2 or s1 <= s2 <= e1


def merge_nodes(G, nodes, new_node):
    """
    Merges the selected `nodes` of the graph G into one `new_node`,
    meaning that all the edges that pointed to or from one of these
    `nodes` will point to or from the `new_node`.
    attr_dict and **attr are defined as in `G.add_node`.
    """

    G.add_node(new_node)  # Add the 'merged' node

    edge_list = list(G.edges(data=True))
    for n1, n2, data in edge_list:
        # For all edges related to one of the nodes to merge,
        # make an edge going to or coming from the `new gene`.
        if n1 in nodes:
            G.add_edge(new_node, n2)
        elif n2 in nodes:
            G.add_edge(n1, new_node)

    graph_nodes = G.nodes(data=True)
    for n in nodes:  # remove the merged nodes
        if n in graph_nodes:
            G.remove_node(n)


def levenshtein(a, b):
    """Calculates the Levenshtein distance between a and b."""
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n
    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]