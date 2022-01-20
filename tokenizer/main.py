from tokenizer.tokenizer import *

filename = "/Users/dkochergin/diploma/bert/output_flowers/image_00001.svg"
tokenizer = SvgTokenizer()
tensor = tokenizer.parseSvg(filename)
tokenizer.saveSvg(tensor)
