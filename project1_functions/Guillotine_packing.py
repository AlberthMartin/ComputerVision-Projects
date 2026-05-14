class Node:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.used = False
        self.right = None
        self.down = None


def guillotine_packing(N):
    root = Node(0, 0, 250, 1330)
    rects = [(135, 56.5)] * N

    placements = []

    def insert(node, w, h):
        if node.used:
            return insert(node.right, w, h) or insert(node.down, w, h)

        elif w <= node.w and h <= node.h:
            node.used = True
            node.right = Node(node.x + w, node.y, node.w - w, h)
            node.down = Node(node.x, node.y + h, node.w, node.h - h)
            return node
        else:
            return None

    for i in range(N):
        placed = False
        for (w, h) in [(135, 56.5), (56.5, 135)]:
            node = insert(root, w, h)
            if node:
                placements.append({
                    "x": node.x,
                    "y": node.y,
                    "w": h,
                    "h": w
                })
                placed = True
                break

        if not placed:
            break

    max_length = max(p["x"] + p["h"] for p in placements)
    return max_length, placements