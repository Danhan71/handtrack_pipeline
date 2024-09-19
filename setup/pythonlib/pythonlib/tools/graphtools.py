""" for use with networkx"""
import networkx as nx

def path_between_nodepair(graph, nodes, fail_if_multiple=True, maxn=10):
    """ find list of edges between nodes
    will be in directed format (but actually undir)
    e.g., [(0,1,0), (1,2,0)]..
    - fail_if_multiple, then makes sure the nodes
    you gave me define a single path
    IN:
    - nodes, list of ints, [0,1], len 2

    OUT:
    - list of edges
    """

    if nodes[1]==nodes[0]:
        # Then this is a loop. get all edges between
        path = [ed for ed in graph.edges if ed[0]==nodes[0] and ed[1]==nodes[0]]
        if fail_if_multiple and len(path)>1:
            print(path)
            print(len(path))
            print(graph.edges)
            print(graph.nodes)
            print(nodes)
            assert False, "mutlipel paths"
    else:
        paths = list(nx.all_simple_edge_paths(graph, nodes[0], nodes[1], maxn))
        if fail_if_multiple and len(paths)>1:
            print(path)
            print(len(path))
            print(graph.edges)
            print(graph.nodes)
            print(nodes)
            assert False, "mutlipel paths"
        elif len(paths)==0:
            assert False, "no path found"
        
        path = [pp for p in paths for pp in p]

    return path


def path_through_list_nodes(graph, list_ni, inputted_all_nodes=True, maxn=10):
    """ given ordered nodes, get the corresponding directed edges (but actually undir)
     key is that between each pair should be unambiguous path.
     e..g, list_ni = [0,5,2]
     --> [(0, 5, 0), (5, 2, 0)]
     - inputted_all_nodes, then all nodes along traj are inputted. e.g., kif trajectory passes
     thru 0,1,2, you dont just give me 0, 2.
     """
    if inputted_all_nodes:
        # all are 1 edge
        maxn=1
    path = []
    for i in range(len(list_ni)-1):
        pair = [list_ni[i], list_ni[i+1]]
        list_of_edges = path_between_nodepair(graph, pair, maxn=maxn)
        # print("HERE", pair, list_of_edges)
        path.extend(list_of_edges)

    return path


def find_all_cycles_nodes(G, source=None, cycle_length_limit=None):
    """forked from networkx dfs_edges function. Assumes nodes are integers, or at least
    types which work with min() and > .
    LT:
    From : https://gist.github.com/joe-jordan/6548029
    Modified to get also cycles with 2 nodes.
    - Previously this sdidnt work since ignored any 2-node, since avoiding going back on oneself (same edge)
    and calling that a cycle. Doesnt work for multi-fgraphs
    - Doesnt allow traverse back an edge. This important, since dont have to use the trick (above), and so works
    on multi-graphs.
    INPUT:
    - source, list of nodes, which will find all cycles when sourcing from those nodes.
    OUTPUT:
    - list of paths, each a list of nodes in a traversed order (does not repeat node1 and nodelast)
    """
    if source is None:
        # produce edges for all components
        nodes=[list(i)[0] for i in nx.connected_components(G)]
    else:
        # produce edges for components with source
        nodes=[source]
    # extra variables for cycle detection:
    cycle_stack = []
    output_cycles = set()
    edge_stack = []
    
    def get_hashable_cycle(cycle):
        """cycle as a tuple in a deterministic order."""
        m = min(cycle)
        mi = cycle.index(m)
        mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
        if cycle[mi-1] > cycle[mi_plus_1]:
            result = cycle[mi:] + cycle[:mi]
        else:
            result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
        return tuple(result)
    
    for start in nodes:
        edges_done = []
        
        if start in cycle_stack:
            continue
        cycle_stack.append(start)
#         stack = [(start,iter(G[start]))]
        stack = [(start,iter(G.edges(start, keys=True)))]
#         stack = [(start,iter(P.find_edges_connected_to_this_node(start)))]
        while stack:
            parent,children = stack[-1]
            try:
                edge = next(children)
                
                # Make sure don't go back down previuoslty done edge
                edge_hash = (set(edge[:2]), edge[2])
                if edge_hash in edges_done:
                    continue
                else:
                    edges_done.append(edge_hash)
                    
                _, child, key = edge
                
                if child not in cycle_stack:
                    # not yet a cycle.
                    cycle_stack.append(child)
#                     stack.append((child,iter(G[child])))
                    stack.append((child,iter(G.edges(child, keys=True))))
                    edge_stack.append(edge)
                else:
                    # You made a cycle. save it.
                    i = cycle_stack.index(child)
                    if False:
                        print("cycle stack", "child:", cycle_stack, child, i)
#                     if i < len(cycle_stack) - 1: 
#                       output_cycles.add(get_hashable_cycle(cycle_stack[i:]))
                    output_cycles.add(get_hashable_cycle(cycle_stack[i:]))
                
            except StopIteration:
                if False:
                    print("---")
                    print(cycle_stack)
                    print(output_cycles)
                stack.pop()
                cycle_stack.pop()
                
    return [list(i) for i in output_cycles]


def find_all_cycles_edges(G, source=None, cycle_length_limit=None):
    """forked from networkx dfs_edges function. Assumes nodes are integers, or at least
    types which work with min() and > .
    LT:
    From : https://gist.github.com/joe-jordan/6548029
    Modified to get also cycles with 2 nodes.
    - Previously this sdidnt work since ignored any 2-node, since avoiding going back on oneself (same edge)
    and calling that a cycle. Doesnt work for multi-fgraphs
    - Doesnt allow traverse back an edge. This important, since dont have to use the trick (above), and so works
    on multi-graphs.
    INPUT:
    - source, list of nodes, which will find all cycles when sourcing from those nodes.
    OUTPUT:
    - list of edges
    """
    if source is None:
        # produce edges for all components
        nodes=[list(i)[0] for i in nx.connected_components(G)]
    else:
        # produce edges for components with source
        nodes=[source]
    # extra variables for cycle detection:
    cycle_stack = []
    output_cycles = set()
    
    edge_stack = []
    output_cycles_edges = []

    def get_hashable_cycle(cycle):
        """cycle as a tuple in a deterministic order.
        """
        m = min(cycle)
        mi = cycle.index(m)
        mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
        if cycle[mi-1] > cycle[mi_plus_1]:
            result = cycle[mi:] + cycle[:mi]
        else:
            result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
        return tuple(result)
    
    for start in nodes:
        edges_done = []
        
        if start in cycle_stack:
            continue
        cycle_stack.append(start)
#         stack = [(start,iter(G[start]))]
        stack = [(start,iter(G.edges(start, keys=True)))]
#         stack = [(start,iter(P.find_edges_connected_to_this_node(start)))]
        while stack:
            parent,children = stack[-1]
            try:
                edge = next(children)
                
                # Make sure don't go back down previuoslty done edge
                edge_hash = (set(edge[:2]), edge[2])
                if edge_hash in edges_done:
                    continue
                else:
                    edges_done.append(edge_hash)
                    
                _, child, key = edge
                if child not in cycle_stack:
                    # not yet a cycle.
                    cycle_stack.append(child)
#                     stack.append((child,iter(G[child])))
                    stack.append((child,iter(G.edges(child, keys=True))))
                    edge_stack.append(edge)
                else:
                    # You made a cycle. save it.
                    i = cycle_stack.index(child)
                    # edge_stack.append(edge)
                    if False:
                        print("cycle stack", "child:", cycle_stack, child, i)
#                     if i < len(cycle_stack) - 1: 
#                       output_cycles.add(get_hashable_cycle(cycle_stack[i:]))
                    output_cycles.add(get_hashable_cycle(cycle_stack[i:]))
                    output_cycles_edges.append(edge_stack[i:] + [edge])
                
            except StopIteration:
                if False:
                    print("---")
                    print(cycle_stack)
                    print(output_cycles)
                stack.pop()
                if len(edge_stack)>0:
                    edge_stack.pop()
                cycle_stack.pop()

    output_cycles = [list(i) for i in output_cycles]
    output_cycles_edges = [list(i) for i in output_cycles_edges]
    return output_cycles_edges
