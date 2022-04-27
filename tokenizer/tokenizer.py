import os.path

import numpy as np
from svgelements import *
import matplotlib.colors
import drawSvg as draw
from svgpathtools import svg2paths, wsvg, Arc
from io import StringIO


from tokenizer import svg
from tokenizer.svg import MoveTo, Segment


class SvgTokenizer:
    visible_idx = 0
    color_r_idx = 1
    color_g_idx = 2
    color_b_idx = 3
    is_stroke_black = 4
    opacity = 5
    stroke_width = 6
    last = 7

    op_visible = 0
    operation_M = 1
    operation_L = 2
    operation_C = 3
    operation_CLOSE = 4
    x_idx = 5
    y_idx = 6
    cx1_idx = 7
    cy1_idx = 8
    cx2_idx = 9
    cy2_idx = 10

    op_size = 11

    # token path layout:
    # visible | color_r | color_g | color_b | is_stroke_black | max_dots_in_path_count * [operation ( 0 - M, 1 - L, 2 - C) | x | y | cx1 | cy1 | cx2 | cy2

    def __init__(self, max_paths_count=600, max_dots_in_path_count=400):
        self.max_paths_count = max_paths_count
        self.max_dots_in_path_count = max_dots_in_path_count
        self.path_size = max_dots_in_path_count * self.op_size + self.last

    def getOperation(self, *operation_prob):
        ind = np.argmax(operation_prob)
        return ind + 1

    def get_max_width(self, tensor, scale = 1.0):
        max_height = -1e9
        max_width = -1e9
        min_height = 1e9
        min_width = 1e9
        for path in tensor:
            if path[SvgTokenizer.visible_idx] <= 0:
                continue

            for i in range(self.max_dots_in_path_count):
                if path[self.last + i * self.op_size + self.op_visible] <= 0:
                    continue

                opM = path[self.last + i * self.op_size + self.operation_M]
                opL = path[self.last + i * self.op_size + self.operation_L]
                opC = path[self.last + i * self.op_size + self.operation_C]
                opZ = path[self.last + i * self.op_size + self.operation_CLOSE]

                max_op = self.getOperation(opM, opL, opC, opZ)

                if max_op == self.operation_M or max_op == self.operation_L:
                    path[self.last + i * self.op_size + self.x_idx] = path[
                                                                          self.last + i * self.op_size + self.x_idx] * scale
                    path[self.last + i * self.op_size + self.y_idx] = path[
                                                                          self.last + i * self.op_size + self.y_idx] * scale
                    max_height = max(max_height, path[self.last + i * self.op_size + self.y_idx])
                    max_width = max(max_width, path[self.last + i * self.op_size + self.x_idx])
                    min_height = min(min_height, path[self.last + i * self.op_size + self.y_idx])
                    min_width = min(min_width, path[self.last + i * self.op_size + self.x_idx])
                elif max_op == self.operation_C:
                    path[self.last + i * self.op_size + self.x_idx] = path[
                                                                          self.last + i * self.op_size + self.x_idx] * scale
                    path[self.last + i * self.op_size + self.y_idx] = path[
                                                                          self.last + i * self.op_size + self.y_idx] * scale
                    max_height = max(max_height, path[self.last + i * self.op_size + self.y_idx])
                    max_width = max(max_width, path[self.last + i * self.op_size + self.x_idx])
                    min_height = min(min_height, path[self.last + i * self.op_size + self.y_idx])
                    min_width = min(min_width, path[self.last + i * self.op_size + self.x_idx])

                    path[self.last + i * self.op_size + self.cx1_idx] = path[
                                                                            self.last + i * self.op_size + self.cx1_idx] * scale
                    path[self.last + i * self.op_size + self.cy1_idx] = path[
                                                                            self.last + i * self.op_size + self.cy1_idx] * scale
                    max_height = max(max_height, path[self.last + i * self.op_size + self.cy1_idx])
                    max_width = max(max_width, path[self.last + i * self.op_size + self.cx1_idx])
                    min_height = min(min_height, path[self.last + i * self.op_size + self.cy1_idx])
                    min_width = min(min_width, path[self.last + i * self.op_size + self.cx1_idx])

                    path[self.last + i * self.op_size + self.cx2_idx] = path[
                                                                            self.last + i * self.op_size + self.cx2_idx] * scale
                    path[self.last + i * self.op_size + self.cy2_idx] = path[
                                                                            self.last + i * self.op_size + self.cy2_idx] * scale
                    max_height = max(max_height, path[self.last + i * self.op_size + self.cy2_idx])
                    max_width = max(max_width, path[self.last + i * self.op_size + self.cx2_idx])
                    min_height = min(min_height, path[self.last + i * self.op_size + self.cy2_idx])
                    min_width = min(min_width, path[self.last + i * self.op_size + self.cx2_idx])

                elif max_op == self.operation_CLOSE:
                    pass
                else:
                    raise RuntimeError("Unknown op")
        return (max_width - min_width, max_height-min_height)

    def saveSvg(self, tensor, scale=1.0, filename="out.svg"):
        tensor = copy(tensor)
        max_height = -1e9
        max_width = -1e9
        min_height = 1e9
        min_width = 1e9


        for path in tensor:
            if path[SvgTokenizer.visible_idx] <= 0:
                continue

            for i in range(self.max_dots_in_path_count):
                if path[self.last + i * self.op_size + self.op_visible] <= 0:
                    continue

                opM = path[self.last + i * self.op_size + self.operation_M]
                opL = path[self.last + i * self.op_size + self.operation_L]
                opC = path[self.last + i * self.op_size + self.operation_C]
                opZ = path[self.last + i * self.op_size + self.operation_CLOSE]

                max_op = self.getOperation(opM, opL, opC, opZ)

                if max_op == self.operation_M or max_op == self.operation_L:
                    path[self.last + i * self.op_size + self.x_idx] = path[
                                                                          self.last + i * self.op_size + self.x_idx] * scale
                    path[self.last + i * self.op_size + self.y_idx] = path[
                                                                          self.last + i * self.op_size + self.y_idx] * scale
                    max_height = max(max_height, path[self.last + i * self.op_size + self.y_idx])
                    max_width = max(max_width, path[self.last + i * self.op_size + self.x_idx])
                    min_height = min(min_height, path[self.last + i * self.op_size + self.y_idx])
                    min_width = min(min_width, path[self.last + i * self.op_size + self.x_idx])
                elif max_op == self.operation_C:
                    path[self.last + i * self.op_size + self.x_idx] = path[
                                                                          self.last + i * self.op_size + self.x_idx] * scale
                    path[self.last + i * self.op_size + self.y_idx] = path[
                                                                          self.last + i * self.op_size + self.y_idx] * scale
                    max_height = max(max_height, path[self.last + i * self.op_size + self.y_idx])
                    max_width = max(max_width, path[self.last + i * self.op_size + self.x_idx])
                    min_height = min(min_height, path[self.last + i * self.op_size + self.y_idx])
                    min_width = min(min_width, path[self.last + i * self.op_size + self.x_idx])

                    path[self.last + i * self.op_size + self.cx1_idx] = path[
                                                                            self.last + i * self.op_size + self.cx1_idx] * scale
                    path[self.last + i * self.op_size + self.cy1_idx] = path[
                                                                            self.last + i * self.op_size + self.cy1_idx] * scale
                    max_height = max(max_height, path[self.last + i * self.op_size + self.cy1_idx])
                    max_width = max(max_width, path[self.last + i * self.op_size + self.cx1_idx])
                    min_height = min(min_height, path[self.last + i * self.op_size + self.cy1_idx])
                    min_width = min(min_width, path[self.last + i * self.op_size + self.cx1_idx])

                    path[self.last + i * self.op_size + self.cx2_idx] = path[
                                                                            self.last + i * self.op_size + self.cx2_idx] * scale
                    path[self.last + i * self.op_size + self.cy2_idx] = path[
                                                                            self.last + i * self.op_size + self.cy2_idx] * scale
                    max_height = max(max_height, path[self.last + i * self.op_size + self.cy2_idx])
                    max_width = max(max_width, path[self.last + i * self.op_size + self.cx2_idx])
                    min_height = min(min_height, path[self.last + i * self.op_size + self.cy2_idx])
                    min_width = min(min_width, path[self.last + i * self.op_size + self.cx2_idx])

                elif max_op == self.operation_CLOSE:
                    pass
                else:
                    raise RuntimeError("Unknown op")

        right = max(max_width, max_height)
        left = min(min_width, min_height)
        res = right - left
        d = draw.Drawing(res, res, origin=(left, -right))
        for path in tensor:
            if path[SvgTokenizer.visible_idx] <= 0:
                continue
            r = path[self.color_r_idx]
            g = path[self.color_g_idx]
            b = path[self.color_b_idx]

            try:
                if r == -1 and g == -1 and b == -1:
                    p = draw.Path(fill="none", stroke='black')
                else:
                    if path[SvgTokenizer.opacity] != -1:
                        if path[SvgTokenizer.stroke_width] != -1:
                            p = draw.Path(fill=matplotlib.colors.to_hex((r, g, b)), opacity=path[SvgTokenizer.opacity], stroke_width=path[SvgTokenizer.stroke_width])
                        else:
                            p = draw.Path(fill=matplotlib.colors.to_hex((r, g, b)), opacity=path[SvgTokenizer.opacity])
                    else:
                        p = draw.Path(fill=matplotlib.colors.to_hex((r, g, b)))
            except Exception as e:
                print(e)
                continue

            for i in range(self.max_dots_in_path_count):
                if path[self.last + i * self.op_size + self.op_visible] < 0:
                    continue

                opM = path[self.last + i * self.op_size + self.operation_M]
                opL = path[self.last + i * self.op_size + self.operation_L]
                opC = path[self.last + i * self.op_size + self.operation_C]
                opZ = path[self.last + i * self.op_size + self.operation_CLOSE]

                max_op = self.getOperation(opM, opL, opC, opZ)

                if max_op == self.operation_M:
                    x = path[self.last + i * self.op_size + self.x_idx]
                    y = path[self.last + i * self.op_size + self.y_idx]
                    p = p.M(x, - y)
                elif max_op == self.operation_L:
                    x = path[self.last + i * self.op_size + self.x_idx]
                    y = path[self.last + i * self.op_size + self.y_idx]
                    p = p.L(x, -y)
                elif max_op == self.operation_C:
                    c1x = path[self.last + i * self.op_size + self.cx1_idx]
                    c1y = path[self.last + i * self.op_size + self.cy1_idx]
                    c2x = path[self.last + i * self.op_size + self.cx2_idx]
                    c2y = path[self.last + i * self.op_size + self.cy2_idx]
                    ex = path[self.last + i * self.op_size + self.x_idx]
                    ey = path[self.last + i * self.op_size + self.y_idx]
                    p = p.C(c1x, -c1y, c2x, -c2y, ex, -ey)
                elif max_op == self.operation_CLOSE:
                    p = p.Z()
                else:
                    raise RuntimeError("Unknown op")
            d.append(p)
        return d.saveSvg(filename)

    def parsePath(self, path, row, k, need_split=False):
        j = 0
        max_height = -1e9
        max_width = -1e9
        min_height = 1e9
        min_width = 1e9
        while k < len(path):
            segment = path[k]
            if j >= self.max_dots_in_path_count:
                print("max_dots_in_path_count")
                raise RuntimeError("OLOLO")
            row[self.last + j * self.op_size + self.op_visible] = 1
            if isinstance(segment, Move):
                if k + 1 < len(path) and isinstance(path[k + 1], Close):
                    k = k + 2
                    continue

                if k + 1 == len(path):
                    k = k + 1
                    continue

                row[self.last + j * self.op_size + self.operation_M] = 1
                row[self.last + j * self.op_size + self.operation_L] = 0
                row[self.last + j * self.op_size + self.operation_C] = 0
                row[self.last + j * self.op_size + self.operation_CLOSE] = 0
                row[self.last + j * self.op_size + SvgTokenizer.x_idx] = segment.end.x
                row[self.last + j * self.op_size + SvgTokenizer.y_idx] = segment.end.y
                max_height = max(max_height, segment.end.y)
                max_width = max(max_width, segment.end.x)
                min_height = min(min_height, segment.end.y)
                min_width = min(min_width, segment.end.x)
            elif isinstance(segment, CubicBezier):
                row[self.last + j * self.op_size + self.operation_M] = 0
                row[self.last + j * self.op_size + self.operation_L] = 0
                row[self.last + j * self.op_size + self.operation_C] = 1
                row[self.last + j * self.op_size + self.operation_CLOSE] = 0
                row[self.last + j * self.op_size + SvgTokenizer.x_idx] = segment.end.x
                row[self.last + j * self.op_size + SvgTokenizer.y_idx] = segment.end.y
                row[self.last + j * self.op_size + SvgTokenizer.cx1_idx] = segment.control1.x
                row[self.last + j * self.op_size + SvgTokenizer.cy1_idx] = segment.control1.y
                row[self.last + j * self.op_size + SvgTokenizer.cx2_idx] = segment.control2.x
                row[self.last + j * self.op_size + SvgTokenizer.cy2_idx] = segment.control2.y
                max_height = max(max_height, segment.end.y)
                max_width = max(max_width, segment.end.x)
                max_height = max(max_height, segment.control1.y)
                max_width = max(max_width, segment.control1.x)
                max_height = max(max_height, segment.control2.y)
                max_width = max(max_width, segment.control2.x)

                min_height = min(min_height, segment.end.y)
                min_width = min(min_width, segment.end.x)
                min_height = min(min_height, segment.control1.y)
                min_width = min(min_width, segment.control1.x)
                min_height = min(min_height, segment.control2.y)
                min_width = min(min_width, segment.control2.x)
            elif isinstance(segment, Line):
                row[self.last + j * self.op_size + self.operation_M] = 0
                row[self.last + j * self.op_size + self.operation_L] = 1
                row[self.last + j * self.op_size + self.operation_C] = 0
                row[self.last + j * self.op_size + self.operation_CLOSE] = 0
                row[self.last + j * self.op_size + SvgTokenizer.x_idx] = segment.end.x
                row[self.last + j * self.op_size + SvgTokenizer.y_idx] = segment.end.y
                max_height = max(max_height, segment.end.y)
                max_width = max(max_width, segment.end.x)
                min_height = min(min_height, segment.end.y)
                min_width = min(min_width, segment.end.x)
            elif isinstance(segment, Close):
                row[self.last + j * self.op_size + self.operation_M] = 0
                row[self.last + j * self.op_size + self.operation_L] = 0
                row[self.last + j * self.op_size + self.operation_C] = 0
                row[self.last + j * self.op_size + self.operation_CLOSE] = 1
                if need_split:
                    return k + 1
            else:
                raise RuntimeError("STRANGE SEGMENT IN PATH")
            j = j + 1
            k = k + 1

        return k, max_height, max_width, min_height, min_width

    def parseSvg(self, filename, normalize=True):
        color_parser = SVG.parse(filename)
        atrs = None
        if not isinstance(filename, StringIO):
            _, atrs = svg2paths(filename)
        svg_tensor = np.full((self.max_paths_count, self.path_size), -1, dtype=float)

        i = 0
        max_height = 0
        max_width = 0
        min_height = 1e9
        min_width = 1e9
        for element in color_parser.elements():
            if i >= self.max_paths_count:
                print("MAX_PATH_COUNT")
                raise RuntimeError("BLALBLABLA")
            try:
                if element.values['visibility'] == 'hidden':
                    continue
            except (KeyError, AttributeError):
                pass
            if isinstance(element, Path):
                row = svg_tensor[i]
                if len(element) == 1:
                    continue
                row[self.visible_idx] = 1
                color = element.values['fill']
                if color.startswith('rgb('):
                   s = color.removeprefix('rgb(')[:-1]
                   arr = s.split(', ')
                   r = float(arr[0])/255.0
                   g = float(arr[1])/255.0
                   b = float(arr[2])/255.0
                else:
                    (r, g, b) = matplotlib.colors.to_rgb(color)
                if color == 'none':
                    r = -1.0
                    g = -1.0
                    b = -1.0
                    row[SvgTokenizer.is_stroke_black] = 1
                row[SvgTokenizer.color_r_idx] = r
                row[SvgTokenizer.color_g_idx] = g
                row[SvgTokenizer.color_b_idx] = b

                if atrs is not None:
                    try:
                        row[SvgTokenizer.opacity] = float(atrs[i]['opacity'])
                        row[SvgTokenizer.stroke_width] = float(atrs[i]['stroke-width'])
                    except (KeyError, AttributeError):
                        pass
                j, mh, mw, mih, miw = self.parsePath(element, row, 0)
                max_height = max(max_height, mh)
                max_width = max(max_width, mw)
                min_width = min(min_width, miw)
                min_height = min(min_height, mih)
                while j < len(element):
                    i = i + 1
                    if i >= self.max_paths_count:
                        print("MAX_PATH_COUNT")
                        raise RuntimeError("BLALBLABLA")
                    row = svg_tensor[i]
                    j, mh, mw, mih, miw = self.parsePath(element, row, j)
                    max_height = max(max_height, mh)
                    max_width = max(max_width, mw)
                    min_width = min(min_width, miw)
                    min_height = min(min_height, mih)
                i = i + 1
            elif isinstance(element, Shape):
                e = Path(element)
                e.reify()  # In some cases the shape could not have reified, the path must.
                if len(e) == 1:
                    continue
                row = svg_tensor[i]
                row[self.visible_idx] = 1
                color = e.values['fill']
                if color == "none":
                    continue
                (r, g, b) = matplotlib.colors.to_rgb(color)
                row[SvgTokenizer.color_r_idx] = r
                row[SvgTokenizer.color_g_idx] = g
                row[SvgTokenizer.color_b_idx] = b
                j, mh, mw, mih, miw = self.parsePath(e, row, 0)
                max_height = max(max_height, mh)
                max_width = max(max_width, mw)
                min_width = min(min_width, miw)
                min_height = min(min_height, mih)
                while j < len(e):
                    i = i + 1
                    row = svg_tensor[i]
                    row[self.visible_idx] = 1
                    row[SvgTokenizer.color_r_idx] = r
                    row[SvgTokenizer.color_g_idx] = g
                    row[SvgTokenizer.color_b_idx] = b
                    if i >= self.max_paths_count:
                        print("MAX_PATH_COUNT")
                        raise RuntimeError("BLALBLABLA")
                    j, mh, mw, mih, miw = self.parsePath(e, row, 0)
                    max_height = max(max_height, mh)
                    max_width = max(max_width, mw)
                    min_width = min(min_width, miw)
                    min_height = min(min_height, mih)
                i = i + 1

        # max_width = max(max_width, max_height)
        # max_height = max(max_width, max_height)
        if normalize:
            delit = max((max_width - min_width), (max_height - min_height))
            for path in svg_tensor:
                for i in range(self.max_dots_in_path_count):
                    if path[self.last + i * self.op_size + self.op_visible] <= 0:
                        continue

                    opM = path[self.last + i * self.op_size + self.operation_M]
                    opL = path[self.last + i * self.op_size + self.operation_L]
                    opC = path[self.last + i * self.op_size + self.operation_C]
                    opZ = path[self.last + i * self.op_size + self.operation_CLOSE]

                    max_op = self.getOperation(opM, opL, opC, opZ)

                    if max_op == self.operation_M or max_op == self.operation_L:
                        path[self.last + i * self.op_size + self.x_idx] = (path[
                                                                               self.last + i * self.op_size + self.x_idx] - min_width) / delit
                        path[self.last + i * self.op_size + self.y_idx] = (path[
                                                                               self.last + i * self.op_size + self.y_idx] - min_height) / delit
                    elif max_op == self.operation_C:
                        path[self.last + i * self.op_size + self.cx1_idx] = (path[
                                                                                 self.last + i * self.op_size + self.cx1_idx] - min_width) / delit
                        path[self.last + i * self.op_size + self.cy1_idx] = (path[
                                                                                 self.last + i * self.op_size + self.cy1_idx] - min_height) / delit
                        path[self.last + i * self.op_size + self.cx2_idx] = (path[
                                                                                 self.last + i * self.op_size + self.cx2_idx] - min_width) / delit
                        path[self.last + i * self.op_size + self.cy2_idx] = (path[
                                                                                 self.last + i * self.op_size + self.cy2_idx] - min_height) / delit
                        path[self.last + i * self.op_size + self.x_idx] = (path[
                                                                               self.last + i * self.op_size + self.x_idx] - min_width) / delit
                        path[self.last + i * self.op_size + self.y_idx] = (path[
                                                                               self.last + i * self.op_size + self.y_idx] - min_height) / delit
                    elif max_op == self.operation_CLOSE:
                        pass
                    else:
                        raise RuntimeError("Unknown op")
        return svg_tensor

    def parseSvgArr(self, svgArr):
        res = []
        for svg in svgArr:
            res.append(self.parseSvg(svg))
        return np.array(res)

    def cut(self, tensor, min_width, max_width):
        for path in tensor:
            for i in range(self.max_dots_in_path_count):
                if path[self.last + i * self.op_size + self.op_visible] <= 0:
                    continue

                opM = path[self.last + i * self.op_size + self.operation_M]
                opL = path[self.last + i * self.op_size + self.operation_L]
                opC = path[self.last + i * self.op_size + self.operation_C]
                opZ = path[self.last + i * self.op_size + self.operation_CLOSE]

                max_op = self.getOperation(opM, opL, opC, opZ)

                if max_op == self.operation_M or max_op == self.operation_L:
                    path[self.last + i * self.op_size + self.x_idx] = min(path[
                                                                           self.last + i * self.op_size + self.x_idx], max_width)
                    path[self.last + i * self.op_size + self.y_idx] = min(path[
                                                                           self.last + i * self.op_size + self.y_idx], max_width)
                    path[self.last + i * self.op_size + self.x_idx] = max(path[
                                                                              self.last + i * self.op_size + self.x_idx],
                                                                          min_width)
                    path[self.last + i * self.op_size + self.y_idx] = max(path[
                                                                              self.last + i * self.op_size + self.y_idx],
                                                                          min_width)
                elif max_op == self.operation_C:
                    path[self.last + i * self.op_size + self.cx1_idx] = min(path[
                                                                             self.last + i * self.op_size + self.cx1_idx], max_width)
                    path[self.last + i * self.op_size + self.cy1_idx] = min(path[
                                                                             self.last + i * self.op_size + self.cy1_idx], max_width)
                    path[self.last + i * self.op_size + self.cx2_idx] = min(path[
                                                                             self.last + i * self.op_size + self.cx2_idx], max_width)
                    path[self.last + i * self.op_size + self.cy2_idx] = min(path[
                                                                             self.last + i * self.op_size + self.cy2_idx], max_width)
                    path[self.last + i * self.op_size + self.x_idx] = min(path[
                                                                           self.last + i * self.op_size + self.x_idx], max_width)
                    path[self.last + i * self.op_size + self.y_idx] = min(path[
                                                                           self.last + i * self.op_size + self.y_idx], max_width)

                    path[self.last + i * self.op_size + self.cx1_idx] = max(path[
                                                                                self.last + i * self.op_size + self.cx1_idx],
                                                                            min_width)
                    path[self.last + i * self.op_size + self.cy1_idx] = max(path[
                                                                                self.last + i * self.op_size + self.cy1_idx],
                                                                            min_width)
                    path[self.last + i * self.op_size + self.cx2_idx] = max(path[
                                                                                self.last + i * self.op_size + self.cx2_idx],
                                                                            min_width)
                    path[self.last + i * self.op_size + self.cy2_idx] = max(path[
                                                                                self.last + i * self.op_size + self.cy2_idx],
                                                                            min_width)
                    path[self.last + i * self.op_size + self.x_idx] = max(path[
                                                                              self.last + i * self.op_size + self.x_idx],
                                                                          min_width)
                    path[self.last + i * self.op_size + self.y_idx] = max(path[
                                                                              self.last + i * self.op_size + self.y_idx],
                                                                          min_width)
                elif max_op == self.operation_CLOSE:
                    pass
                else:
                    raise RuntimeError("Unknown op")

    def empty_svg_tensor(self):
        return np.full((self.max_paths_count, self.path_size), -1, dtype=float)

    def concat_svg_tensors(self, l, r):
        len_path_l = 0
        svg_tensor = np.full((self.max_paths_count, self.path_size), -1, dtype=float)
        i = 0
        for path in l:
            if path[SvgTokenizer.visible_idx] <= 0:
                continue
            svg_tensor[i] = path
            i = i + 1

        for path in r:
            if i >= self.max_paths_count:
                raise RuntimeError("Max path count reached")

            if path[SvgTokenizer.visible_idx] <= 0:
                continue
            svg_tensor[i] = path
            i = i + 1

        return svg_tensor

    def tranlsate(self, tensor, x, y):
        for path in tensor:
            if path[SvgTokenizer.visible_idx] <= 0:
                continue

            for i in range(self.max_dots_in_path_count):
                if path[self.last + i * self.op_size + self.op_visible] <= 0:
                    continue

                opM = path[self.last + i * self.op_size + self.operation_M]
                opL = path[self.last + i * self.op_size + self.operation_L]
                opC = path[self.last + i * self.op_size + self.operation_C]
                opZ = path[self.last + i * self.op_size + self.operation_CLOSE]

                max_op = self.getOperation(opM, opL, opC, opZ)

                if max_op == self.operation_M or max_op == self.operation_L:
                    path[self.last + i * self.op_size + self.x_idx] = path[
                                                                          self.last + i * self.op_size + self.x_idx] + x
                    path[self.last + i * self.op_size + self.y_idx] = path[
                                                                          self.last + i * self.op_size + self.y_idx] + y
                elif max_op == self.operation_C:
                    path[self.last + i * self.op_size + self.x_idx] = path[
                                                                          self.last + i * self.op_size + self.x_idx] + x
                    path[self.last + i * self.op_size + self.y_idx] = path[
                                                                          self.last + i * self.op_size + self.y_idx] + y

                    path[self.last + i * self.op_size + self.cx1_idx] = path[
                                                                            self.last + i * self.op_size + self.cx1_idx] + x
                    path[self.last + i * self.op_size + self.cy1_idx] = path[
                                                                            self.last + i * self.op_size + self.cy1_idx] + y

                    path[self.last + i * self.op_size + self.cx2_idx] = path[
                                                                            self.last + i * self.op_size + self.cx2_idx] + x
                    path[self.last + i * self.op_size + self.cy2_idx] = path[
                                                                            self.last + i * self.op_size + self.cy2_idx] + y

                elif max_op == self.operation_CLOSE:
                    pass
                else:
                    raise RuntimeError("Unknown op")

    def scale(self, tensor, scale):
        for path in tensor:
            if path[SvgTokenizer.visible_idx] <= 0:
                continue

            for i in range(self.max_dots_in_path_count):
                if path[self.last + i * self.op_size + self.op_visible] <= 0:
                    continue

                opM = path[self.last + i * self.op_size + self.operation_M]
                opL = path[self.last + i * self.op_size + self.operation_L]
                opC = path[self.last + i * self.op_size + self.operation_C]
                opZ = path[self.last + i * self.op_size + self.operation_CLOSE]

                max_op = self.getOperation(opM, opL, opC, opZ)

                if max_op == self.operation_M or max_op == self.operation_L:
                    path[self.last + i * self.op_size + self.x_idx] = path[
                                                                          self.last + i * self.op_size + self.x_idx] * scale
                    path[self.last + i * self.op_size + self.y_idx] = path[
                                                                          self.last + i * self.op_size + self.y_idx] * scale
                elif max_op == self.operation_C:
                    path[self.last + i * self.op_size + self.x_idx] = path[
                                                                          self.last + i * self.op_size + self.x_idx] * scale
                    path[self.last + i * self.op_size + self.y_idx] = path[
                                                                          self.last + i * self.op_size + self.y_idx] * scale

                    path[self.last + i * self.op_size + self.cx1_idx] = path[
                                                                            self.last + i * self.op_size + self.cx1_idx] * scale
                    path[self.last + i * self.op_size + self.cy1_idx] = path[
                                                                            self.last + i * self.op_size + self.cy1_idx] * scale

                    path[self.last + i * self.op_size + self.cx2_idx] = path[
                                                                            self.last + i * self.op_size + self.cx2_idx] * scale
                    path[self.last + i * self.op_size + self.cy2_idx] = path[
                                                                            self.last + i * self.op_size + self.cy2_idx] * scale

                elif max_op == self.operation_CLOSE:
                    pass
                else:
                    raise RuntimeError("Unknown op")
