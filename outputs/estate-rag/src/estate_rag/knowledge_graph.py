"""A knowledge graph is data; the LangGraph graph is the execution workflow."""

import re
from collections import deque
import networkx as nx
from .models import Evidence


def normalize(value):
    return " ".join(re.findall(r"\w+", value.casefold()))


class KnowledgeGraph:
    def __init__(self, payload: dict, tenant: str):
        self.tenant = tenant
        self.graph = nx.MultiDiGraph()
        for node in payload["nodes"]:
            if node["tenant"] == tenant:
                self.graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
        for edge in payload["edges"]:
            if edge["tenant"] == tenant and edge["source"] in self.graph and edge["target"] in self.graph:
                self.graph.add_edge(edge["source"], edge["target"], **edge)

    def resolve(self, mentions: list[str]) -> tuple[list[str], list[str]]:
        resolved, ambiguous = [], []
        for mention in mentions[:5]:
            matches = [
                node
                for node, data in self.graph.nodes(data=True)
                if normalize(mention)
                in {
                    normalize(node),
                    normalize(data["name"]),
                    *[normalize(a) for a in data.get("aliases", [])],
                }
            ]
            if len(matches) == 1:
                resolved.extend(matches)
            elif len(matches) > 1:
                ambiguous.append(mention)
        return list(dict.fromkeys(resolved)), ambiguous

    def mentions_in(self, question: str):
        # Deterministic alias fallback, word boundaries avoid L-001 matching L-0010.
        text = f" {normalize(question)} "
        return [
            node
            for node, data in self.graph.nodes(data=True)
            if any(
                f" {normalize(alias)} " in text for alias in [node, data["name"], *data.get("aliases", [])]
            )
        ][:5]

    def traverse(self, seeds: list[str], hops=2, max_edges=30) -> list[Evidence]:
        if not 1 <= hops <= 2:
            raise ValueError("Traversal limited to two hops")
        queue = deque((seed, 0) for seed in seeds if seed in self.graph)
        visited, seen_edges, evidence = set(), set(), []
        while queue and len(evidence) < max_edges:
            node, depth = queue.popleft()
            if node in visited or depth >= hops:
                continue
            visited.add(node)
            edges = list(self.graph.out_edges(node, keys=True, data=True)) + list(
                self.graph.in_edges(node, keys=True, data=True)
            )
            for source, target, key, data in edges:
                edge_id = data["id"]
                if edge_id not in seen_edges:
                    seen_edges.add(edge_id)
                    text = f"{source} ({self.graph.nodes[source]['name']}) --{data['relation']}--> "
                    text += f"{target} ({self.graph.nodes[target]['name']}); evidence: {data['evidence']}"
                    evidence.append(
                        Evidence(id=f"graph:{edge_id}", text=text, source=data["document"], kind="graph")
                    )
                    if len(evidence) >= max_edges:
                        break
                queue.append((target if source == node else source, depth + 1))
        return evidence

    def listing_ids(self, seeds):
        # Live status tools only operate on directly resolved listings; no broad autonomous lookups.
        return [s for s in seeds if self.graph.nodes[s]["type"] == "Listing"][:3]
