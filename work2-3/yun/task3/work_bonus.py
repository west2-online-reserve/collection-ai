import numpy as np
import matplotlib.pyplot as plt
grayscale_image=np.random.randint(0, 256, size=(200, 300))

origin_color_image=np.stack((grayscale_image, grayscale_image, grayscale_image), axis=2).astype(np.uint8)  #将灰度图像复制到三个通道上，形成一个RGB图像

sepia_matrix = np.array([
    [0.393, 0.769, 0.189],
    [0.349, 0.686, 0.168],
    [0.272, 0.534, 0.131]
])

finally_color_image=np.clip(origin_color_image @ sepia_matrix.T, 0, 255)

L=0.299*finally_color_image[:,:,0] + 0.587*finally_color_image[:,:,1] + 0.114*finally_color_image[:,:,2]
alpha=1.5
L=L.reshape(200,300,1)
finally_color_image=np.clip(L+alpha*(finally_color_image-L),0,255)
board=20
#处理左边
t_left=np.linspace(0,1,board).reshape(1,board,1)#变化系数
finally_color_image[:,0:20,:]=finally_color_image[:,0:20,:]*t_left
#处理右边
t_right=np.linspace(1,0,board).reshape(1,board,1)
finally_color_image[:,-1:-21:-1,:]=finally_color_image[:,-1:-21:-1,:]*(1-t_right)+t_right*255

finally_color_image=np.clip(finally_color_image,0,255).astype(np.uint8)             #将像素值限制在0-255范围内，并转换为uint8类型

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.imshow(origin_color_image)
plt.title("Original Grayscale Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(finally_color_image.astype(np.uint8))
plt.title("Sepia-toned Image with Gradient Borders")
plt.axis('off')
plt.tight_layout()
plt.show()

