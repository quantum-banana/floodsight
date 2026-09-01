import heapq
import math

from app.intelligence.contracts import (
    AccessibilityGraph,
    GridCellEvidence,
    RoadEdgeEvidence,
)
from app.schemas.live_result import DataOrigin, Point, Road, RoadState, Route, RouteStatus


class AccessibilityEngine:
    def build(self, cells: list[GridCellEvidence]) -> AccessibilityGraph:
        by_id = {cell.cell_id: cell for cell in cells}
        nodes = {
            cell_id: Point(x=(cell.column + 0.5) / 4, y=(cell.row + 0.5) / 4)
            for cell_id, cell in by_id.items()
        }
        edges: list[RoadEdgeEvidence] = []
        for cell_id in sorted(by_id):
            row = ord(cell_id[0]) - ord("A")
            column = int(cell_id[1:]) - 1
            for next_row, next_column in ((row + 1, column), (row, column + 1)):
                if next_row >= 4 or next_column >= 4:
                    continue
                other_id = f"{chr(ord('A') + next_row)}{next_column + 1}"
                current, other = by_id[cell_id], by_id[other_id]
                state = max((current.road_state, other.road_state), key=_road_priority)
                confidence = max(current.confidence, other.confidence)
                uncertainty = 1.0 - confidence
                enabled = state is not RoadState.BLOCKED
                penalty = {
                    RoadState.CLEAR: 0.0,
                    RoadState.UNKNOWN: 1.5,
                    RoadState.FLOODED: 4.0,
                    RoadState.BLOCKED: 100.0,
                }[state]
                edges.append(
                    RoadEdgeEvidence(
                        edge_id=f"EDGE-{cell_id}-{other_id}",
                        start_node=cell_id,
                        end_node=other_id,
                        geometry=[nodes[cell_id], nodes[other_id]],
                        state=state,
                        severity={
                            RoadState.CLEAR: 0.0,
                            RoadState.UNKNOWN: 0.25,
                            RoadState.FLOODED: 0.65,
                            RoadState.BLOCKED: 1.0,
                        }[state],
                        uncertainty=round(uncertainty, 4),
                        travel_cost=round(1.0 + penalty + uncertainty * 2.0, 4),
                        enabled=enabled,
                    )
                )
        return AccessibilityGraph(nodes=nodes, edges=edges)

    @staticmethod
    def as_live_roads(graph: AccessibilityGraph) -> list[Road]:
        return [
            Road(
                road_id=edge.edge_id,
                label="Relative image-space accessibility edge",
                state=edge.state,
                access_status=(
                    "BLOCKED"
                    if not edge.enabled
                    else "DEGRADED"
                    if edge.state in {RoadState.FLOODED, RoadState.UNKNOWN}
                    else "ACCESSIBLE"
                ),
                geometry=edge.geometry,
                confidence=round(1.0 - edge.uncertainty, 4),
                data_origin=DataOrigin.DERIVED_ANALYTIC,
                travel_cost=edge.travel_cost,
                enabled=edge.enabled,
                uncertainty=edge.uncertainty,
            )
            for edge in graph.edges
        ]


class RoutingEngine:
    def __init__(self, *, base_node: str = "D1") -> None:
        self.base_node = base_node
        self._previous_edges: dict[str, tuple[str, ...]] = {}

    def route(
        self, graph: AccessibilityGraph, *, zone_id: str, target_cells: list[str]
    ) -> tuple[Route, list[Route]]:
        candidates = [
            self._shortest(graph, self.base_node, target, excluded=set())
            for target in target_cells
            if target in graph.nodes
        ]
        available = [item for item in candidates if item is not None]
        if not available:
            return (
                Route(
                    route_id=f"ROUTE-{zone_id}",
                    status=RouteStatus.UNAVAILABLE,
                    target_zone_id=zone_id,
                    label="No enabled relative route",
                    waypoints=[],
                    distance_m=None,
                    access_summary="No image-space path avoids the currently blocked edges.",
                    data_origin=DataOrigin.DERIVED_ANALYTIC,
                    edge_ids=[],
                    route_cost=None,
                    changed_reason="Target is inaccessible in the current relative graph.",
                ),
                [],
            )
        cost, nodes, edge_ids = min(available, key=lambda item: (item[0], item[1]))
        previous = self._previous_edges.get(zone_id)
        current = tuple(edge_ids)
        changed_reason = None
        if previous is not None and previous != current:
            changed_reason = "Route changed because accessibility evidence changed."
        self._previous_edges[zone_id] = current
        route = self._route_model(
            zone_id, graph, cost, nodes, edge_ids, changed_reason=changed_reason
        )
        alternatives = self._alternatives(graph, zone_id, nodes[-1], edge_ids)
        return route, alternatives

    def _alternatives(
        self, graph: AccessibilityGraph, zone_id: str, target: str, primary_edges: list[str]
    ) -> list[Route]:
        alternatives = []
        for excluded_edge in primary_edges:
            candidate = self._shortest(graph, self.base_node, target, {excluded_edge})
            if candidate is not None and candidate[2] != primary_edges:
                alternatives.append(candidate)
        if not alternatives:
            return []
        cost, nodes, edges = min(alternatives, key=lambda item: (item[0], item[1]))
        return [
            self._route_model(
                zone_id,
                graph,
                cost,
                nodes,
                edges,
                route_id=f"ROUTE-{zone_id}-ALT",
                label="Alternative relative route",
            )
        ]

    def _route_model(
        self,
        zone_id: str,
        graph: AccessibilityGraph,
        cost: float,
        nodes: list[str],
        edge_ids: list[str],
        *,
        route_id: str | None = None,
        label: str = "Recommended relative route",
        changed_reason: str | None = None,
    ) -> Route:
        return Route(
            route_id=route_id or f"ROUTE-{zone_id}",
            status=RouteStatus.RECOMMENDED,
            target_zone_id=zone_id,
            label=label,
            waypoints=[graph.nodes[node] for node in nodes],
            distance_m=None,
            access_summary=(
                "Image-space tactical route; no GIS distance or real-world traversability claimed."
            ),
            data_origin=DataOrigin.DERIVED_ANALYTIC,
            edge_ids=edge_ids,
            route_cost=round(cost, 4),
            changed_reason=changed_reason,
        )

    @staticmethod
    def _shortest(
        graph: AccessibilityGraph, start: str, target: str, excluded: set[str]
    ) -> tuple[float, list[str], list[str]] | None:
        adjacency: dict[str, list[tuple[str, RoadEdgeEvidence]]] = {
            node: [] for node in graph.nodes
        }
        for edge in graph.edges:
            if not edge.enabled or edge.edge_id in excluded:
                continue
            adjacency[edge.start_node].append((edge.end_node, edge))
            adjacency[edge.end_node].append((edge.start_node, edge))
        queue: list[tuple[float, str, list[str], list[str]]] = [(0.0, start, [start], [])]
        best = {start: 0.0}
        while queue:
            cost, node, nodes, edges = heapq.heappop(queue)
            if node == target:
                return cost, nodes, edges
            if cost > best.get(node, math.inf):
                continue
            for neighbor, edge in sorted(adjacency[node], key=lambda item: item[1].edge_id):
                next_cost = cost + edge.travel_cost
                if next_cost < best.get(neighbor, math.inf):
                    best[neighbor] = next_cost
                    heapq.heappush(
                        queue,
                        (next_cost, neighbor, [*nodes, neighbor], [*edges, edge.edge_id]),
                    )
        return None


def _road_priority(state: RoadState) -> int:
    return {RoadState.CLEAR: 0, RoadState.UNKNOWN: 1, RoadState.FLOODED: 2, RoadState.BLOCKED: 3}[
        state
    ]
