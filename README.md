# pca_minibatchkmeans

This code is specifically developed for change detection in both SAR and optical images. It is capable of handling all types of image formats, such as .jpeg, .jpg, .png, .bmp, .tiff, .tif, .ntf, and .nitf.

Optionally, we have included a pre-processing step to eliminate vegetation/non-vegetation, water, and shadow changes in the final change map

The following libraries need to be imported:
1. time
2. os
3. numpy
4. cv2
5. rasterio
6. tifffile
7. sklearn
8. gdal
