from logging import getLogger
from typing import Self

logger = getLogger(__name__)

class Graph(dict[int, set[int]]):
    def __init__(self, n: int = 0):
        super().__init__([
            (i, set()) \
            for i in range(n)
        ])
        self.is_circular: bool = False

    def __str__(self) -> str:
        lines = [
            f"{orig}: {str(dests)}"\
            for (orig, dests) in self.items()
        ]
        lines.insert(0, f"Graph at {hex(id(self))}:")

        return '\n'.join(lines)
    
    def copy(self) -> Self:
        g = Graph()
        g.update(self)
        return g
    
    def expand(self, inplace: bool = True) -> Self:
        """
        This method expands all sets such that all downstream nodes are 
        shown.
        """
        if inplace: new_graph: Graph = self
        else:       new_graph: Graph = self.copy()

        check: bool = (len(self) > 1)
        for orig, dests in sorted(
            new_graph.items(),
            key = lambda item: len(item[1]),
        ):
            if check and (len(dests) > 0):
                self.is_circular = True
                logger.warning("'Graph' is circular!")
                return new_graph
            
            new_dests = dests.copy()
            for dest in dests:
                new_dests = new_dests.union(new_graph[dest])

            new_graph[orig] = new_dests
            check = False

        return new_graph

    def isolateRoot(self) -> tuple[int | None, Self]:
        if self.is_circular:
            return None, self

        possible_roots: set[int] = set(self.keys())
        for dests in self.values():
            # Remove common values
            possible_roots -= dests

        if len(possible_roots) == 0:
            self.is_circular = True
            logger.warning("'Graph' is circular!")
            return None, self
        
        root = sorted(possible_roots)[0]
        graph = Graph()
        graph.update(dict([
            item \
            for item in self.items() \
            if item[0] != root
        ]))

        return root, graph
    
    def createChain(self) -> list[int]:
        if self.is_circular:
            return list(range(len(self)))
        
        chain: list[int] = []
        graph = self
        while True:
            root, graph = graph.isolateRoot()
            chain.append(root)

            if len(graph) == 0:
                break

        return chain