from app.schemas.live_result import BoundingBox, Point


def normalize_point(x: float, y: float, width: int, height: int) -> Point:
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    return Point(x=min(1.0, max(0.0, x / width)), y=min(1.0, max(0.0, y / height)))


def normalize_bbox(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> BoundingBox:
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bounding box must have positive area")
    start = normalize_point(x1, y1, width, height)
    end = normalize_point(x2, y2, width, height)
    box_width = end.x - start.x
    box_height = end.y - start.y
    if box_width <= 0 or box_height <= 0:
        raise ValueError("clipped bounding box has no area")
    return BoundingBox(x=start.x, y=start.y, width=box_width, height=box_height)


def bbox_center(box: BoundingBox) -> Point:
    return Point(x=box.x + box.width / 2, y=box.y + box.height / 2)


def grid_cell_for_point(point: Point, *, rows: int = 4, columns: int = 4) -> str:
    if rows < 1 or columns < 1 or rows > 26:
        raise ValueError("grid dimensions are invalid")
    row = min(rows - 1, int(point.y * rows))
    column = min(columns - 1, int(point.x * columns))
    return f"{chr(ord('A') + row)}{column + 1}"


def grid_cell_polygon(cell_id: str, *, rows: int = 4, columns: int = 4) -> list[Point]:
    if len(cell_id) < 2:
        raise ValueError("invalid grid cell ID")
    row = ord(cell_id[0].upper()) - ord("A")
    column = int(cell_id[1:]) - 1
    if not (0 <= row < rows and 0 <= column < columns):
        raise ValueError("grid cell is outside the configured grid")
    x1, x2 = column / columns, (column + 1) / columns
    y1, y2 = row / rows, (row + 1) / rows
    return [Point(x=x1, y=y1), Point(x=x2, y=y1), Point(x=x2, y=y2), Point(x=x1, y=y2)]
