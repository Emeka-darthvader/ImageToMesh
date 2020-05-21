import pix2vertex as p2v
from imageio import imread
import matplotlib
import matplotlib.pyplot as plt

detector = p2v.Detector()
reconstructor = p2v.Reconstructor(detector=detector)

image = imread("img/pic.jpeg")
#image = imread("img/pic2.jpg")
# image = imread("img/pic3.jpg")
fig = plt.figure()
plt.imshow(image)
plt.show()

img_crop = detector.detect_and_crop(image)
fig = plt.figure()
plt.imshow(img_crop)
plt.show()



net_res = reconstructor.run_net(img_crop)
p2v.vis_net_result(img_crop,net_res)
final_res = reconstructor.post_process(net_res)

#interactive visualization
plot = p2v.vis_depth_interactive(final_res['Z_surface'])

plot = p2v.vis_pcloud_interactive(final_res,img_crop)

p2v.vis_depth_matplotlib(img_crop,final_res['Z_surface'])


#******DEfault*******
# result, crop = p2v.reconstruct(image)

# p2v.vis_depth_interactive(result['Z_surface'])