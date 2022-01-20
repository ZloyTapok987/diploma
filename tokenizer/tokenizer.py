import numpy as np
from svgelements import *
import matplotlib.colors
import drawSvg as draw

from tokenizer import svg
from tokenizer.svg import MoveTo, Segment


class SvgTokenizer:
    visible_idx = 0
    color_r_idx = 1
    color_g_idx = 2
    color_b_idx = 3
    last = 4

    op_visible = 0
    operation_M = 1
    operation_L = 2
    x_idx = 3
    y_idx = 4

    op_size = 5

    class Operation:
        M, L = range(2)

    # token path layout:
    # visible | color_r | color_g | color_b |  max_dots_in_path_count * [operation (-1 - undefined, 0 - M, 1 - L) | x | y]

    def __init__(self, max_paths_count=600, max_dots_in_path_count=400):
        self.max_paths_count = max_paths_count
        self.max_dots_in_path_count = max_dots_in_path_count
        self.path_size = max_dots_in_path_count * self.op_size + self.last

    # self.header = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\" width=\"{0}\" height=\"{1}\">\n"

    def saveSvg(self, tensor, scale=1.0, filename="out.svg"):
        tensor = copy(tensor)
        max_height = 0
        max_width = 0

        for path in tensor:
            if path[SvgTokenizer.visible_idx] <= 0:
                continue
            for i in range(self.max_dots_in_path_count):
                if path[self.last + i * self.op_size + self.op_visible] > 0:
                    path[self.last + i * self.op_size + self.x_idx] = path[
                                                                          self.last + i * self.op_size + self.x_idx] * scale
                    path[self.last + i * self.op_size + self.y_idx] = path[
                                                                          self.last + i * self.op_size + self.y_idx] * scale
                    max_height = max(max_height, abs(path[self.last + i * self.op_size + self.y_idx]))
                    max_width = max(max_width, abs(path[self.last + i * self.op_size + self.x_idx]))

        d = draw.Drawing(max_width, max_height)
        for path in tensor:
            if path[SvgTokenizer.visible_idx] <= 0:
                continue
            r = path[self.color_r_idx]
            g = path[self.color_g_idx]
            b = path[self.color_b_idx]

            try:
                p = draw.Path(fill=matplotlib.colors.to_hex((r, g, b)))
            except Exception as e:
                continue

            for i in range(self.max_dots_in_path_count):
                if path[self.last + i * self.op_size + self.op_visible] < 0:
                    continue

                opM = path[self.last + i * self.op_size + self.operation_M]
                opL = path[self.last + i * self.op_size + self.operation_L]
                x = path[self.last + i * self.op_size + self.x_idx]
                y = path[self.last + i * self.op_size + self.y_idx]
                if opM > opL:
                    p = p.M(x, max_height - y)
                else:
                    p = p.L(x, max_height - y)
            d.append(p)
        return d.saveSvg(filename)

    def parseSvg(self, filename, simplify=True):
        geometry_parser = svg.Svg(filename)
        color_parser = SVG.parse(filename)
        svg_tensor = np.full((self.max_paths_count, self.path_size), -1, dtype=float)
        i = 0
        max_height = 0
        max_width = 0
        for path in geometry_parser.items[0].items:
            if i >= self.max_paths_count: break
            row = svg_tensor[i]
            i = i + 1
            j = 0
            if simplify:
                segments = path.simplify(3)
            else:
                segments = path.simplify(0)
            for segment in segments:
                row[SvgTokenizer.visible_idx] = 1
                k = 0
                for elem in segment:
                    if k >= self.max_dots_in_path_count: break
                    row[self.last + k * self.op_size + self.op_visible] = 1
                    if k == 0:
                        row[self.last + k * self.op_size + self.operation_M] = 1
                        row[self.last + k * self.op_size + self.operation_L] = 0
                        row[self.last + k * self.op_size + SvgTokenizer.x_idx] = elem.x
                        row[self.last + k * self.op_size + SvgTokenizer.y_idx] = elem.y
                        max_height = max(max_height, abs(elem.y))
                        max_width = max(max_width, abs(elem.x))
                    else:
                        row[self.last + k * self.op_size + self.operation_M] = 0
                        row[self.last + k * self.op_size + self.operation_L] = 1
                        row[self.last + k * self.op_size + SvgTokenizer.x_idx] = elem.x
                        row[self.last + k * self.op_size + SvgTokenizer.y_idx] = elem.y
                        max_height = max(max_height, abs(elem.y))
                        max_width = max(max_width, abs(elem.x))
                    k = k + 1
                j = j + 1

        for path in svg_tensor:
            for i in range(self.max_dots_in_path_count):
                if path[self.last + i * self.op_size + self.op_visible] > 0:
                    path[self.last + i * self.op_size + self.x_idx] = path[
                                                                        self.last + i * self.op_size + self.x_idx] / max_width
                    path[self.last + i * self.op_size + self.y_idx] = path[
                                                                          self.last + i * self.op_size + self.y_idx] / max_height

        i = 0
        for path in color_parser:
            if i >= self.max_paths_count: break
            row = svg_tensor[i]
            i = i + 1
            color = path.values['fill']
            (r, g, b) = matplotlib.colors.to_rgb(color)
            row[SvgTokenizer.color_r_idx] = r
            row[SvgTokenizer.color_g_idx] = g
            row[SvgTokenizer.color_b_idx] = b
        return svg_tensor

    def parseSvgArr(self, svgArr):
        res = []
        for svg in svgArr:
            res.append(self.parseSvg(svg))
        return np.array(res)
