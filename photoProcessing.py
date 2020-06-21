import matplotlib.pyplot as plt

img = plt.imread('/home/anaconda/data/ch3/sample.jpg')
print(img.shape)

plt.imshow(img)
