from PIL import Image
from JPEGRestorator import JPEGRestorator

Restorator = JPEGRestorator("checkpoints/SNet-epoch20-jpeg20.pth")
im = Image.open("images/monarch_jpeg_q20.png")
restored_im = Restorator.restore(im)
restored_im.save("images/monarch_restored.png")


