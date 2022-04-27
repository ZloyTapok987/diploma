import os

from tokenizer import *

svg_dir = "output_cats"

PATH_SIZE = 700
DOT_SIZE = 150

tokenizer = SvgTokenizer(PATH_SIZE, DOT_SIZE)
count = 0
for file in os.listdir(svg_dir):
    count = count + 1
    filename = os.path.join(svg_dir, file)
    tensor = tokenizer.parseSvg(filename)
    tensor.tofile('output_encoded_labels/image_{0}.csv'.format(count), sep=',', format='%10.5f')
