import subprocess
import os
from multiprocessing.dummy import Pool as ThreadPool

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    datasetDir = 'sample'
    outputDir = "/Users/dkochergin/deepsvg/logo_generation/gan_generated_logo"
    arr = []
    for file in os.listdir(datasetDir):
        filename = os.path.join(datasetDir, file)
        out_filename = os.path.join(outputDir, os.path.splitext(file)[0] + ".svg")
        cmd = './vtracer-mac --color_precision 6 --mode polygon --filter_speckle 4  --gradient_step 64 --input ' \
              '{} --output {}'.format(filename, out_filename)
        arr.append(cmd)

    pool = ThreadPool(16)
    pool.map(os.system, arr)
    pool.close()
    pool.join()


#./vtracer-mac --color_precision 6 --mode polygon --filter_speckle 10  --gradient_step 64 --input flowers/image_00001.jpg --output output_flowers/image_00001.jpg