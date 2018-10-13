# Qt-Image-Hash-with-multiple-methods

A Qt-based image hash system that implements 5 methods for efficient detection of similar images from an image database, given a host image.

You have to include opencv_world340.dll to start the project, which also requires support of opencv-contrib.

_____________________________________________________________________________________________________________________

How to use:

Select Folder(database that contains images) and host image -> Select Hash Method(at the bottom of GUI, or use the default pHash) -> Generate Hash Code for all images in the folder(press F5) -> Get the most similar images by comparison of Hash Code(press F6).

_____________________________________________________________________________________________________________________

References for the methods implemented in this program:

Perceptual Image Hash(pHash) -> "A Visual Model-Based Perceptual Image Hash for Content Authentication", https://ieeexplore.ieee.org/document/7050251

Simplified Image Hash -> https://blog.csdn.net/sinat_26917383/article/details/78582064

Otsu's Method -> https://www.aliyun.com/zixun/wenji/1305502.html

Structual Similarity(SSIM) -> "Image quality assessment: from error visibility to structural similarity", https://ieeexplore.ieee.org/document/1284395

Local HSV Histogram -> https://blog.csdn.net/zhuiqiuk/article/details/54945624

_____________________________________________________________________________________________________________________

Kindly contact me if have problem using this system or any suggestions via shinydotcom@163.com, Thanks!
