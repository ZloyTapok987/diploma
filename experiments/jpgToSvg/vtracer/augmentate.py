import Augmentor

p = Augmentor.Pipeline('/Users/dkochergin/deepsvg/text_png')

p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.5, min_factor=1.02, max_factor=1.1)

p.sample(70000)