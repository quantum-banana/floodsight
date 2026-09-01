from app.inference.contracts import (
    DetectionProvenance,
    DetectionResult,
    EvidenceSource,
    FusedScene,
    SegmentationProvenance,
    SegmentationResult,
    SemanticEvidence,
)
from app.inference.taxonomy import Taxonomy

FLOOD_EVIDENCE_CLASSES = frozenset({"water", "road_flooded", "building_flooded"})
POOL_CLASS = "pool"


class SceneFusionEngine:
    def __init__(self, segmentation_taxonomy: Taxonomy) -> None:
        self.taxonomy = segmentation_taxonomy
        if POOL_CLASS in self.taxonomy.by_name and POOL_CLASS in FLOOD_EVIDENCE_CLASSES:
            raise ValueError("pool cannot be a flood evidence class")

    def fuse(
        self,
        *,
        frame_id: int,
        timestamp_ms: int,
        source_width: int,
        source_height: int,
        segmentation: SegmentationResult | None,
        detection: DetectionResult | None,
    ) -> FusedScene:
        for result in (segmentation, detection):
            if result is None:
                continue
            if (
                result.frame_id != frame_id
                or result.source_width != source_width
                or result.source_height != source_height
            ):
                raise ValueError("model result does not match the source frame")
        evidence = []
        if segmentation is not None:
            source = (
                EvidenceSource.SIMULATED
                if segmentation.provenance_mode is SegmentationProvenance.SIMULATED
                else EvidenceSource.SEGMENTATION
            )
            evidence = [
                SemanticEvidence(
                    class_id=item.class_id,
                    class_name=item.class_name,
                    coverage_percent=item.coverage_percent,
                    confidence=item.mean_confidence,
                    source=source,
                )
                for item in segmentation.class_statistics
            ]
        provenance: list[EvidenceSource] = []
        if segmentation is not None:
            provenance.append(
                EvidenceSource.SIMULATED
                if segmentation.provenance_mode is SegmentationProvenance.SIMULATED
                else EvidenceSource.SEGMENTATION
            )
        if detection is not None:
            provenance.append(
                EvidenceSource.SIMULATED
                if detection.provenance_mode is DetectionProvenance.SIMULATED
                else EvidenceSource.DETECTION
            )
        flood_ids = [
            item.class_id for item in self.taxonomy.classes if item.name in FLOOD_EVIDENCE_CLASSES
        ]
        pool = self.taxonomy.by_name.get(POOL_CLASS)
        return FusedScene(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            source_width=source_width,
            source_height=source_height,
            taxonomy_version=self.taxonomy.version,
            semantic_mask=segmentation.mask if segmentation else None,
            semantic_evidence=evidence,
            detections=detection.detections if detection else [],
            flood_class_ids=flood_ids,
            pool_class_id=pool.class_id if pool else None,
            provenance=list(dict.fromkeys(provenance)),
        )
