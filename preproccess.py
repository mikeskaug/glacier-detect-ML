import os
from subprocess import run

BOUNDS = (454155, 9117345, 550155, 8955345)  # ulx, uly, lrx, lry in UTM


def crop(image, out_dir, bounds):
    ulx, uly, lrx, lry = bounds
    filename_w_extension = os.path.split(image)[1]
    filename = os.path.splitext(filename_w_extension)[0]
    out_path = os.path.join(os.path.abspath(out_dir), filename + '_clipped.tif')

    command_args = ['rio', 'clip', image, out_path, '--bounds', str(ulx), str(lry), str(lrx), str(uly)]
    run(command_args)


def prepare_images(in_dir, out_dir, crop_bounds):
    for path, dirs, files in os.walk(in_dir):
        for filename in files:
            if '.TIF' in filename:
                crop(os.path.join(path, filename), out_dir, crop_bounds)
