from cairosvg import svg2png
import os

dirname = "5"

outputdirname = "output/" + dirname

if not os.path.exists(outputdirname):
    os.makedirs(outputdirname)


for file in os.listdir(dirname):
    with open(os.path.join(dirname, file), 'r') as f:
        svgStr = f.read()
    svg2png(svgStr, write_to=os.path.join(outputdirname, os.path.splitext(file)[0] + ".png"))