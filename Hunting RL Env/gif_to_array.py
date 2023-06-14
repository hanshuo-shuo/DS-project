import PIL
import numpy as np
from PIL import Image
import os

def readGif(filename, asNumpy=True):
    """ readGif(filename, asNumpy=True)
    Read images from an animated GIF file.  Returns a list of numpy
    arrays, or, if asNumpy is false, a list if PIL images.
    """

    # Check PIL
    if PIL is None:
        raise RuntimeError("Need PIL to read animated gif files.")

    # Check Numpy
    if np is None:
        raise RuntimeError("Need Numpy to read animated gif files.")

    # Check whether it exists
    if not os.path.isfile(filename):
        raise IOError('File not found: ' + str(filename))

    # Load file using PIL
    pilIm = PIL.Image.open(filename)
    pilIm.seek(0)

    # Read all images inside
    images = []
    try:
        while True:
            # Get image as numpy array
            tmp = pilIm.convert()  # Make without palette
            a = np.asarray(tmp)
            if len(a.shape) == 0:
                raise MemoryError("Too little memory to convert PIL image to array")
            # Store, and next
            images.append(a)
            pilIm.seek(pilIm.tell() + 1)
    except EOFError:
        pass

    # Convert to normal PIL images if needed
    if not asNumpy:
        images2 = images
        images = []
        for index, im in enumerate(images2):
            tmp = PIL.Image.fromarray(im)
            images.append(tmp)

    images = np.array(images)
    images = images[:,:,:,0]

    # Done
    return images

def main():
    num = 2
    gif_array = readGif(f"gif_real/test_{num}.gif")
    print(gif_array)


if __name__ == "__main__":
    main()