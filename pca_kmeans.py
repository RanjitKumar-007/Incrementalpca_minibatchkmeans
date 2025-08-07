"""pca_kmeans - Version 1.0.5"""
"""Importing libraries"""
import time
import os
from collections import Counter
import numpy as np
import cv2
import rasterio # pylint: disable=import-error
import tifffile # pylint: disable=import-error
from sklearn.cluster import KMeans, MiniBatchKMeans # pylint: disable=import-error
from sklearn.decomposition import PCA, IncrementalPCA # pylint: disable=import-error
from osgeo import gdal
from pre_post_processing import load_images
img_obj = load_images()
def extract_patches_strided_batch(image, patch_size=5, batch_size=2000):
    """Generator that yields strided 5x5 patches in batches"""
    h, w = image.shape
    for i in range(0, h - patch_size + 1, batch_size):
        h_end = min(i + batch_size, h - patch_size + 1)
        batch_shape = (h_end - i, w - patch_size + 1, patch_size, patch_size)
        batch_strides = (image.strides[0], image.strides[1],
                         image.strides[0], image.strides[1])
        patches = np.lib.stride_tricks.as_strided(
            image[i:], shape=batch_shape, strides=batch_strides)
        yield patches.reshape(-1, patch_size * patch_size)
def find_vector_set_small(diff_image, nw_size):
    """Finding vector set for smaller images"""
    i = 0
    j = 0
    vector_set = np.zeros((int(nw_size[0] * nw_size[1] / 25), 25))
    while i < vector_set.shape[0]:
        while j < nw_size[1]:
            k = 0
            while k < nw_size[0]:
                block = diff_image[j:j + 5, k:k + 5]
                feature = block.ravel()
                vector_set[i, :] = feature
                k = k + 5
            j = j + 5
        i = i + 1
    mean_vec = np.mean(vector_set, axis=0)
    vector_set = vector_set - mean_vec
    return vector_set, mean_vec
def find_vector_set_large(diff_image, nw_size, batch_size=2000):
    """Finding vector set for larger images,
    and streaming mean vector computation with patch batches"""
    total_sum = np.zeros((25,), dtype=np.float64)
    total_count = 0
    for batch in extract_patches_strided_batch(diff_image,
                                               patch_size=5, batch_size=batch_size):
        total_sum += batch.sum(axis=0)
        total_count += batch.shape[0]
    mean_vec = total_sum / total_count
    return None, mean_vec
def find_fvs_small(evs, diff_image, mean_vec, new):
    """Finding feature vector set for smaller images"""
    i = 2
    feature_vector_set = []
    while i < new[1] - 2:
        j = 2
        while j < new[0] - 2:
            block = diff_image[i - 2:i + 3, j - 2:j + 3]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j + 1
        i = i + 1
    fvs = np.dot(feature_vector_set, evs)
    fvs = fvs - mean_vec
    return fvs
def find_fvs_large(evs, diff_image, mean_vec, new, batch_size=2000):
    """Finding feature vector set for larger images with batched projection,
    writing to memmap"""
    total_patches = (new[1] - 4) * (new[0] - 4)
    fvs = np.memmap('/tmp/fvs_projected.dat', dtype='float32',
                    mode='w+', shape=(total_patches, evs.shape[0]))
    index = 0
    for batch in extract_patches_strided_batch(diff_image,
                                               patch_size=5, batch_size=batch_size):
        batch = batch.astype(np.float32)
        batch -= mean_vec
        projected = np.dot(batch, evs.T)
        fvs[index:index + batch.shape[0]] = projected
        index += batch.shape[0]
    return fvs
def clustering(fvs, components, new, method='minibatchkmeans'):
    """Perform clustering using MiniBatchKMeans or KMeans"""
    if method == 'minibatchkmeans':
        kmeans = MiniBatchKMeans(n_clusters=components, batch_size=1024, n_init=10,
                              init='k-means++', random_state=42, verbose=0)
    elif method == 'kmeans':
        kmeans = KMeans(components, verbose=0)
    kmeans.fit(fvs)
    output = kmeans.predict(fvs)
    count = Counter(output)
    small_index = min(count, key=count.get)
    change_mask = np.reshape(output, (new[1] - 4, new[0] - 4))
    return small_index, change_mask
def run_pca_kmeans(image_path1, image_path2, ndvi_threshold,
                 ndwi_threshold, ndsi_threshold, savi_threshold):
    """Main execution block"""
    start_time = time.time()
    print("[0%] starting pca_kmeans")
    file_ext1 = os.path.splitext(image_path1)[-1].lower()
    file_ext2 = os.path.splitext(image_path2)[-1].lower()
    if file_ext1 != file_ext2:
        print("Error: Image formats do not match")
        return
    if file_ext1 in ['.jpg', '.jpeg', '.bmp', '.png']:
        image1, image2 = img_obj.load_sar_rgb_images(image_path1, image_path2)
    elif file_ext1 in ['.tif', '.tiff', '.ntf', '.nitf']:
        if file_ext1 in ['.tif', '.tiff']:
            with rasterio.open(image_path1) as src1:
                num_bands1 = src1.count
            with rasterio.open(image_path2) as src2:
                num_bands2 = src2.count
        else:
            dataset_1 = gdal.Open(image_path1)
            dataset_2 = gdal.Open(image_path2)
            num_bands1 = dataset_1.RasterCount
            num_bands2 = dataset_2.RasterCount
        if num_bands1 in (1, 2) and num_bands2 in (1, 2):
            image1, image2 = img_obj.load_sar_rgb_images(image_path1, image_path2)
            image1 = img_obj.norm_img(image1)
            image2 = img_obj.norm_img(image2)
        else:
            user_input = int(input("1. Without preprocessing, 2. With preprocessing: "))
            if user_input == 1:
                image1, image2 = img_obj.load_optical_tif_ntf_images(
                image_path1, image_path2, ndvi_threshold, ndwi_threshold,
                ndsi_threshold, savi_threshold, apply_preprocessing=False)
            elif user_input == 2:
                image1, image2 = img_obj.load_optical_tif_ntf_images(
                image_path1, image_path2, ndvi_threshold, ndwi_threshold,
                ndsi_threshold, savi_threshold, apply_preprocessing=True)
            else:
                raise ValueError("Give valid input")
    else:
        raise ValueError("Unsupported file formats")
    new_size = np.asarray(image1.shape) // 5
    new_size = new_size.astype(int) * 5
    image1_resized = cv2.resize(image1, (new_size[0], new_size[1])).astype(int) # pylint: disable=no-member
    image2_resized = cv2.resize(image2, (new_size[0], new_size[1])).astype(int) # pylint: disable=no-member
    difference_image = abs(image1_resized - image2_resized)
    difference_image = difference_image[:, :, 1]
    print("[30%] image loading, resizing, and differencing completed")
    height, width = difference_image.shape
    if height <= 5000 and width <= 5000:
        vector_sets, mean_vector = find_vector_set_small(difference_image, new_size)
        pca = PCA()
        pca.fit(vector_sets)
        eigen_vector = pca.components_
        feature_vector = find_fvs_small(eigen_vector, difference_image, mean_vector, new_size)
        components = 3
        least_index, change_map = clustering(feature_vector, components,
                                        new_size, method='kmeans')
    elif height > 20000 and width > 20000:
        raise ValueError("This particular model is not ideal for very larger images")
    else:
        _, mean_vector = find_vector_set_large(difference_image, new_size)
        ipca = IncrementalPCA(n_components=5, batch_size=2000)
        for batch in extract_patches_strided_batch(difference_image, patch_size=5,
                                              batch_size=2000):
            batch = batch.astype(np.float32)
            batch -= mean_vector
            ipca.partial_fit(batch)
        eigen_vector = ipca.components_
        feature_vector = find_fvs_large(eigen_vector, difference_image,
                                  mean_vector, new_size)
        components = 3
        least_index, change_map = clustering(feature_vector, components,
                                        new_size, method='minibatchkmeans')
    print("[80%] feature extraction and clustering completed")
    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0
    change_map = change_map.astype(np.uint8)
    original_height, original_width = image1.shape[:2]
    change_map_resized = cv2.resize(change_map, (original_width, original_height)) # pylint: disable=no-member
    print("[90%] binary change map generated")
    if all(path.lower().endswith(('.tif', '.tiff',
                                  '.ntf', '.nitf')) for path in (image_path1, image_path2)):
        file_ext = os.path.splitext(image_path1)[-1].lower()
        if file_ext in ['.tif', '.tiff']:
            with rasterio.open(image_path1) as src:
                num_bands = src.count
        else:
            dataset = gdal.Open(image_path1, gdal.GA_ReadOnly)
            num_bands = dataset.RasterCount
        output_path = 'Output_image.tif'
        if num_bands > 3:
            rgb_image = img_obj.process_tif_ntf_unsupervised(image_path1)
            overlay = img_obj.overlay_image(change_map_resized, rgb_image)
        else:
            overlay = img_obj.overlay_image(change_map_resized, image1)
        tifffile.imwrite(output_path, overlay)
    else:
        overlay = img_obj.overlay_image(change_map_resized, image1)
        output_path = 'Output_image.png'
        cv2.imwrite(output_path, overlay) # pylint: disable=no-member
    img_obj.run_georef(image_path1, output_path)
    try:
        os.remove('/tmp/fvs_projected.dat')
    except FileNotFoundError:
        pass
    print("[100%] completed")
    end_time = time.time()
    print("Computational time in seconds:", end_time - start_time)
if __name__ == "__main__":
    image_path1 = 'pisa_1.tif'
    image_path2 = 'pisa_2.tif'
    ndvi_threshold = 0.1 #Value ranges between 0 to 1
    ndwi_threshold = 1 #Value ranges between 0 to 1
    ndsi_threshold = 1 #Value ranges between 0 to 1
    savi_threshold = 1 #Value ranges between 0 to 1
    run_pca_kmeans(image_path1, image_path2, ndvi_threshold,
                     ndwi_threshold, ndsi_threshold, savi_threshold)
