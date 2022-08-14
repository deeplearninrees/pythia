from PIL import Image
import glob
import random
w = glob.glob("../data/*")
random.shuffle(w)

for p, i in enumerate(w):
    f = glob.glob(i+"/*")
    print("Verifying images in {}".format(i), "Only {} folders to go.".format(len(w)-p))
    for j in f:
        m = Image.open(j)
        m.verify()
        