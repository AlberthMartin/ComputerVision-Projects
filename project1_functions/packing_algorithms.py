# %%
# Container dimensions
W1, L1 = 56.5, 135
W2, L2 = 135, 56.5

TRUCK_WIDTH = 250

# Standardized orientations
RECTS = [
    {"w": W1, "h": L1},  # normal
    {"w": W2, "h": L2},  # rotated
]

def skyline_packing(N):
    skyline = [(0, 0, TRUCK_WIDTH)]  # (x, y, width)
    placements = []

    for _ in range(N):
        best = None

        for j, (sx, sy, sw) in enumerate(skyline):
            for rect in RECTS:
                w = rect["w"]  # width (Y)
                h = rect["h"]  # length (X)

                if w <= sw and sy + w <= TRUCK_WIDTH:
                    if best is None or sy < best[0]:
                        best = (sy, j, w, h, sx, sy)

        if best is None:
            break

        _, j, w, h, x, y = best

        placements.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })

        sx, sy, sw = skyline[j]
        skyline.pop(j)

        # split skyline properly
        skyline.insert(j, (sx + h, sy, sw))
        skyline.insert(j, (sx, sy + w, w))

    L = max(p["x"] + p["h"] for p in placements) if placements else 0
    return L, placements

# Solver CP-SAT
from ortools.sat.python import cp_model

def solve_packing_cpsat(N):
    model = cp_model.CpModel()

    # Dimensions (scaled to integers to avoid floats)
    SCALE = 10
    W1, L1 = int(56.5*SCALE), int(135*SCALE)
    W2, L2 = int(135*SCALE), int(56.5*SCALE)

    TRUCK_WIDTH = int(250*SCALE)

    # Variables
    x = [model.NewIntVar(0, 20000, f"x_{i}") for i in range(N)]
    y = [model.NewIntVar(0, TRUCK_WIDTH, f"y_{i}") for i in range(N)]

    r = [model.NewBoolVar(f"r_{i}") for i in range(N)]

    # Width/height depending on rotation
    w = []
    h = []

    for i in range(N):
        wi = model.NewIntVar(min(W1, W2), max(W1, W2), f"w_{i}")
        hi = model.NewIntVar(min(L1, L2), max(L1, L2), f"h_{i}")

        model.Add(wi == W1).OnlyEnforceIf(r[i].Not())
        model.Add(wi == W2).OnlyEnforceIf(r[i])

        model.Add(hi == L1).OnlyEnforceIf(r[i].Not())
        model.Add(hi == L2).OnlyEnforceIf(r[i])

        w.append(wi)
        h.append(hi)

    # Length variable
    L = model.NewIntVar(0, 20000, "L")

    # Boundary constraints
    for i in range(N):
        model.Add(y[i] + w[i] <= TRUCK_WIDTH)
        model.Add(x[i] + h[i] <= L)

    # No-overlap using intervals
    x_intervals = []
    y_intervals = []

    for i in range(N):
        x_end = model.NewIntVar(0, 20000, f"x_end_{i}")
        model.Add(x_end == x[i] + h[i])

        xi = model.NewIntervalVar(x[i], h[i], x_end, f"x_int_{i}")

        y_end = model.NewIntVar(0, TRUCK_WIDTH, f"y_end_{i}")
        model.Add(y_end == y[i] + w[i])

        yi = model.NewIntervalVar(y[i], w[i], y_end, f"y_int_{i}")

        x_intervals.append(xi)
        y_intervals.append(yi)

    model.AddNoOverlap2D(x_intervals, y_intervals)

    # Objective
    model.Minimize(L)

    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None

    solution = []
    for i in range(N):
        solution.append({
            "x": solver.Value(x[i]) / SCALE,
            "y": solver.Value(y[i]) / SCALE,
            "w": solver.Value(w[i]) / SCALE,
            "h": solver.Value(h[i]) / SCALE,
        })

    return solver.Value(L) / SCALE, solution

# SELF PACKING
def shelf_packing(N):
    placements = []
    shelves = []

    current_y = 0

    for i in range(N):
        placed = False

        for shelf in shelves:
            shelf_y, shelf_height, remaining_width = shelf

            for rect in RECTS:
                w = rect["w"]
                h = rect["h"]

                if w <= remaining_width and h <= shelf_height:
                    x = TRUCK_WIDTH - remaining_width
                    y = shelf_y

                    placements.append({
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h
                    })

                    shelf[2] -= w
                    placed = True
                    break

            if placed:
                break

        if not placed:
            # create new shelf
            rect = RECTS[0]  # pick default orientation
            w, h = rect["w"], rect["h"]

            placements.append({
                "x": 0,
                "y": current_y,
                "w": w,
                "h": h
            })

            shelves.append([current_y, h, TRUCK_WIDTH - w])
            current_y += h

    L = max(p["x"] + p["h"] for p in placements) if placements else 0
    return L, placements

## Guillotine Packing
class Node:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.used = False
        self.right = None
        self.down = None


def guillotine_packing(N):
    root = Node(0, 0, TRUCK_WIDTH, 1330)
    placements = []

    def insert(node, w, h):
        if node.used:
            return insert(node.right, w, h) or insert(node.down, w, h)

        if w <= node.w and h <= node.h:
            node.used = True
            node.right = Node(node.x + h, node.y, node.w - w, h)
            node.down = Node(node.x, node.y + w, node.w, node.h - h)
            return node

        return None

    for i in range(N):
        placed = False

        for rect in RECTS:
            w = rect["w"]
            h = rect["h"]

            node = insert(root, w, h)

            if node:
                placements.append({
                    "x": node.x,
                    "y": node.y,
                    "w": w,
                    "h": h
                })
                placed = True
                break

        if not placed:
            break

    L = max(p["x"] + p["h"] for p in placements) if placements else 0
    return L, placements
