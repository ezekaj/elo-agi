"""
Neuro Module 10: Spatial Cognition and Cognitive Maps

Implementation of spatial navigation cells and cognitive maps based on 2025 research:
- Place Cells (Hippocampus) - "Where am I?"
- Grid Cells (Entorhinal cortex) - Hexagonal lattice for distance/displacement
- Head Direction Cells - Compass heading
- Border Cells - Environment edge detection
- Cognitive Maps - Unified spatial representation
- Conceptual Space - Same cells encode conceptual space (2025 discovery)
"""

from .place_cells import PlaceField, PlaceCell, PlaceCellPopulation
from .grid_cells import GridParameters, GridCell, GridCellModule, GridCellPopulation
from .head_direction_cells import HeadDirectionCell, HeadDirectionSystem
from .border_cells import BorderCell, BorderCellPopulation, WallDirection, Wall
from .cognitive_map import CognitiveMap, Environment, Landmark, CognitiveMapState
from .path_integration import PathIntegrator
from .conceptual_space import ConceptCell, ConceptualGrid, SocialDistanceGrid, ConceptualMap

__all__ = [
    "PlaceField",
    "PlaceCell",
    "PlaceCellPopulation",
    "GridParameters",
    "GridCell",
    "GridCellModule",
    "GridCellPopulation",
    "HeadDirectionCell",
    "HeadDirectionSystem",
    "BorderCell",
    "BorderCellPopulation",
    "CognitiveMap",
    "Environment",
    "Wall",
    "Landmark",
    "PathIntegrator",
    "ConceptCell",
    "ConceptualGrid",
    "SocialDistanceGrid",
    "ConceptualMap",
]
