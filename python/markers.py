"""Two-layer "ring + dot" markers used in all figures: a big hollow shape
(`marker_inner_style`) with a small solid dot of the same shape on top
(`marker_outer_style`), both drawn at the same data point. Run this file
directly (``python markers.py``) to preview them.

Note: `unit_regular_polygon(4)` is a square inscribed in the unit circle
with a vertex pointing straight up, which renders as a diamond (rotated
square); rotating that by 45 degrees swings the vertices onto the axes,
which renders as an axis-aligned square. So, confusingly, `square` below
is the shape that looks like a diamond, and `diamond` is the shape that
looks like a square -- the names describe how each Path is built, not how
it looks on screen.
"""

import matplotlib.path as mpath
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

triangle = mpath.Path.unit_regular_polygon(3)
square = mpath.Path.unit_regular_polygon(4)
circle = mpath.Path.unit_circle()
diamond = MarkerStyle(square, transform=Affine2D().rotate_deg(45))
pentagon = mpath.Path.unit_regular_polygon(5)
hexagon = mpath.Path.unit_regular_polygon(6)


def styled(shape):
    """Wrap a shape with the crisp-cornered join/cap style used throughout."""
    return MarkerStyle(shape, joinstyle="miter", capstyle="projecting")


def marker_inner_style(color, size=35, edgewidth=4):
    return dict(markersize=size, markerfacecolor="white", markeredgecolor=color, markeredgewidth=edgewidth)


def marker_outer_style(color, size=15):
    return dict(markersize=size, markerfacecolor=color, markeredgecolor="white", markeredgewidth=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.05)

    shapes = [diamond, square, triangle, circle, pentagon, hexagon]
    for y, shape in enumerate(shapes):
        m = styled(shape)
        for x in range(5):
            ax.plot(x, y, marker=m, **marker_inner_style("blue"))
            ax.plot(x, y, marker=m, **marker_outer_style("blue"))

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, len(shapes) - 0.5)
    plt.show()
