"""Importing libraries"""
# import os
# from osgeo import gdal, osr
import warnings
from osgeo import gdal
import numpy as np
import rasterio # pylint: disable=import-error
import cv2
import tifffile # pylint: disable=import-error
warnings.filterwarnings("ignore")
class load_images():
    """Functions that belongs to pre and post processing"""
    def convert_gray_to_rgb(self, image):
        """Conversion of 1-band or 2-band image to 3-band image"""
        if image.ndim == 2:
            return np.broadcast_to(image[:, :, np.newaxis], (*image.shape, 3))
        elif image.ndim == 3 and image.shape[0] == 2:
            third_band = np.mean(image, axis=0, keepdims=True).astype(image.dtype)
            return np.concatenate([image, third_band], axis=0).transpose(1, 2, 0)
        return image
    def calculate_index(self, band1, band2, formula):
        """Generalizing index calculation function"""
        b_1 = band1.astype('float64')
        b_2 = band2.astype('float64')
        return formula(b_1,b_2)
    def process_image(self, bands):
        """NDVI, NDWI, NDSI, and SAVI calculation"""
        ndvi = self.calculate_index(bands['nir'], bands['red'],
        lambda nir, red: np.where((nir + red) == 0., 0, (nir - red) / (nir + red)))
        ndwi = self.calculate_index(bands['green'], bands['nir'],
        lambda green, nir: np.where((green + nir) == 0., 0, (green - nir) / (green + nir)))
        ndsi = self.calculate_index(bands['swir1'], bands['nir'],
        lambda swir1, nir: np.where((swir1 + nir) == 0., 0, (swir1 - nir) / (swir1 + nir)))
        savi = self.calculate_index(bands['nir'], bands['red'],
        lambda nir, red: np.where((nir + red + 0.5) == 0., 0,
                                  (nir - red)*(1 + 0.5) / (nir + red + 0.5)))
        return ndvi, ndwi, ndsi, savi
    def create_masks(self, ndvi, ndwi, ndsi, savi, ndvi_threshold,
                     ndwi_threshold, ndsi_threshold, savi_threshold):
        """Creating masks based on thresholds for both images"""
        vegetation_mask = ndvi > ndvi_threshold
        water_mask = ndwi > ndwi_threshold
        shadow_mask = ndsi > ndsi_threshold
        soil_mask = savi > savi_threshold
        combined_mask = np.logical_or(np.logical_or
                                      (vegetation_mask, water_mask), shadow_mask, soil_mask)
        return combined_mask
    def apply_mask(self, image, mask):
        """Applying combined mask to each image"""
        mask = mask[:, :, np.newaxis]
        return np.where(mask, np.nan, image.astype('float64'))
    def load_optical_tif_ntf_images(self, image_path1, image_path2, ndvi_threshold, 
                                    ndwi_threshold, ndsi_threshold, savi_threshold,
                                    apply_preprocessing=False):
        """Load optical TIFF, TIF, NITF, and NTF images"""
        if image_path1.lower().endswith(('.tif', '.tiff')):
            def read_image(image_path):
                with rasterio.open(image_path) as src:
                    bands = src.read()
                return bands.transpose(1, 2, 0)
            image1 = read_image(image_path1)
            image2 = read_image(image_path2)
        elif image_path1.lower().endswith(('.nitf', '.ntf')):
            dataset_1 = gdal.Open(image_path1)
            dataset_2 = gdal.Open(image_path2)
            image1 = dataset_1.ReadAsArray().transpose(1, 2, 0)
            image2 = dataset_2.ReadAsArray().transpose(1, 2, 0)
        if image1 is None or image2 is None:
            print(f"Error: Unable to load images {image_path1} or {image_path2}")
            return None, None
        if np.array_equal(image1, image2):
            raise ValueError("Given two input images are similar, no difference")
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0])) # pylint: disable=no-member
        if not apply_preprocessing:
            return image1, image2
        bands1 = {
            'red': image1[:, :, 0], 'green': image1[:, :, 1], 'blue': image1[:, :, 2],
            'nir': image1[:, :, 3], 'swir1': image1[:, :, 4], 'swir2': image1[:, :, 5]
        }
        bands2 = {
            'red': image2[:, :, 0], 'green': image2[:, :, 1], 'blue': image2[:, :, 2],
            'nir': image2[:, :, 3], 'swir1': image2[:, :, 4], 'swir2': image2[:, :, 5]
        }
        ndvi1, ndwi1, ndsi1, savi1 = self.process_image(bands1)
        ndvi2, ndwi2, ndsi2, savi2 = self.process_image(bands2)
        mask1 = self.create_masks(ndvi1, ndwi1, ndsi1, savi1,
                                  ndvi_threshold, ndwi_threshold, ndsi_threshold,
                                  savi_threshold)
        mask2 = self.create_masks(ndvi2, ndwi2, ndsi2, savi2,
                                  ndvi_threshold, ndwi_threshold, ndsi_threshold,
                                  savi_threshold)
        masked_image1 = self.apply_mask(image1, mask1)
        masked_image2 = self.apply_mask(image2, mask2)
        masked_image1 = np.nan_to_num(masked_image1, nan=0, posinf=0, neginf=0)
        masked_image2 = np.nan_to_num(masked_image2, nan=0, posinf=0, neginf=0)
        return masked_image1, masked_image2
    def load_sar_rgb_images(self, image_path1, image_path2):
        """Load one and three channel images"""
        if image_path1.lower().endswith(('.tif', '.tiff')):
            image1 = tifffile.imread(image_path1)
            image2 = tifffile.imread(image_path2)
        elif image_path1.lower().endswith(('.nitf', '.ntf')):
            dataset_1 = gdal.Open(image_path1)
            dataset_2 = gdal.Open(image_path2)
            image1 = dataset_1.ReadAsArray()
            image2 = dataset_2.ReadAsArray()
        elif image_path1.lower().endswith(('.jpg', '.jpeg', '.bmp', '.png')):
            image1 = cv2.imread(image_path1) # pylint: disable=no-member
            image2 = cv2.imread(image_path2) # pylint: disable=no-member
        if image1 is None or image2 is None:
            print(f"Error: Unable to load images {image_path1} or {image_path2}")
            return None, None
        if np.array_equal(image1, image2):
            raise ValueError("Given two input images are similar, no difference")
        image1 = self.convert_gray_to_rgb(image1)
        image2 = self.convert_gray_to_rgb(image2)
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0])) # pylint: disable=no-member
        return image1, image2
    def process_tif_ntf_classical(self, input_data, method='contrast'):
        """ Process TIFF, TIF, NITF, and NTF image array, return it as an RGB array"""
        if not isinstance(input_data, np.ndarray):
            raise ValueError("Input must be a numpy array representing the image")
        red = np.nan_to_num(input_data[:, :, 0], nan=0)
        green = np.nan_to_num(input_data[:, :, 1], nan=0)
        blue = np.nan_to_num(input_data[:, :, 2], nan=0)
        if method == 'contrast':
            def contrast_stretch(band, pmin=2, pmax=98):
                lower = np.percentile(band, pmin)
                upper = np.percentile(band, pmax)
                band = np.clip(band, lower, upper)
                return ((band - lower) / (upper - lower) * 255).astype(np.uint8)
            red = contrast_stretch(red)
            green = contrast_stretch(green)
            blue = contrast_stretch(blue)
        elif method == 'minmax':
            min_val = min(red.min(), green.min(), blue.min())
            max_val = max(red.max(), green.max(), blue.max())
            scale = lambda band: ((band - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            red, green, blue = scale(red), scale(green), scale(blue)
        return np.dstack((red, green, blue))
    def process_tif_ntf_unsupervised(self, input_path):
        """Process TIF, TIFF, NITF and NTF optical images, return it as an RGB array"""
        if input_path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(input_path) as src:
                red = np.nan_to_num(src.read(1), nan=0)
                green = np.nan_to_num(src.read(2), nan=0)
                blue = np.nan_to_num(src.read(3), nan=0)
        elif input_path.lower().endswith(('.nitf', '.ntf')):
            dataset = gdal.Open(input_path)
            red = dataset.GetRasterBand(1).ReadAsArray()
            green = dataset.GetRasterBand(2).ReadAsArray()
            blue = dataset.GetRasterBand(3).ReadAsArray()
            red = np.nan_to_num(red, nan=0)
            green = np.nan_to_num(green, nan=0)
            blue = np.nan_to_num(blue, nan=0)
        def contrast_stretch(band, pmin=2, pmax=98):
            lower = np.percentile(band, pmin)
            upper = np.percentile(band, pmax)
            band = np.clip(band, lower, upper)
            band_scaled = (band - lower) / (upper - lower) * 255
            return band_scaled.astype(np.uint8)
        red =  contrast_stretch(red)
        green =  contrast_stretch(green)
        blue =  contrast_stretch(blue)
        return np.dstack((red, green, blue))
    def norm_img(self, img):
        """Image normalization by computing minimum and maximum pixel values"""
        img = img.astype(np.float32)
        min_val = np.min(img)
        max_val = np.max(img)
        normalized_img = (img - min_val) / (max_val - min_val) * 255
        return normalized_img.astype(np.uint8)
    def automatic_brightness_contrast(self, image, clip_hist_percent=2.5):
        """Automatically adjusts brightness and contrast to preserve color properties"""
        zero_pixels = np.count_nonzero(np.all(image == 0, axis=2))
        total_pixels = image.shape[0] * image.shape[1]
        zero_percentage = zero_pixels / total_pixels * 100
        if zero_percentage < 10:
            clip_hist_percent = 2.0
            contrast_boost = 1.0
            brightness_adjustment = 0.0
        elif 10 <= zero_percentage < 30:
            clip_hist_percent = 2.3
            contrast_boost = 1.0
            brightness_adjustment = 0.0
        elif 30 <= zero_percentage < 50:
            clip_hist_percent = 1.8
            contrast_boost = 1.0
            brightness_adjustment = 0.0
        else:
            clip_hist_percent = 1.8
            contrast_boost = 1.0
            brightness_adjustment = 0.0
        process_image = image.copy().astype(np.float32)
        global_alpha, global_beta = 1.0, 0.0
        for i in range(3):
            channel = process_image[:, :, i]
            non_zero_mask = channel > 0
            hist, _ = np.histogram(channel[non_zero_mask].flatten(), bins=256, range=[0, 256])
            cdf = hist.cumsum()
            total_non_zero_pixels = channel[non_zero_mask].size
            clip_value = clip_hist_percent * total_non_zero_pixels / 100.0
            minimum = np.argmax(cdf > clip_value)
            maximum = np.argmax(cdf > cdf[-1] - clip_value)
            if maximum > minimum:
                alpha = (255.0 / (maximum - minimum)) * contrast_boost
                beta = -minimum * alpha + brightness_adjustment
                channel_adjusted = channel * alpha + beta
                channel[non_zero_mask] = np.clip(channel_adjusted[non_zero_mask], 0, 255)
                global_alpha *= alpha
                global_beta += beta
                darken_threshold = 50
                lighten_threshold = 200
                channel[non_zero_mask & (channel < darken_threshold)] *= 0.9
                mask = non_zero_mask & (channel > lighten_threshold)
                channel[mask] += (255 - channel[mask]) * 0.2
                zero_pixel_factor = 0.7 + (0.1 * (brightness_adjustment / 20))
                channel[~non_zero_mask] *= zero_pixel_factor
                process_image[:, :, i] = np.clip(channel, 0, 255)
        enhanced_image = np.clip(process_image, 0, 255).astype(np.uint8)
        return enhanced_image, global_alpha, global_beta
    def overlay_image(self, change_map_resized, image):
        """Overlay change map on input images"""
        _, binary_mask = cv2.threshold(change_map_resized, 200, 255, cv2.THRESH_BINARY)  # pylint: disable=no-member
        green_mask = np.zeros_like(image)
        indices = np.where(binary_mask == 255)
        green_mask[indices[0], indices[1], :] = [0, 255, 0]
        overlay = cv2.add(image, green_mask)  # pylint: disable=no-member
        return overlay
    '''def extract_georeference(self, input_data):
        """Extract GCPs, projection, geotransform, and CRS from a georeferenced image"""
        dataset = gdal.Open(input_data, gdal.GA_ReadOnly)
        if dataset is None:
            print("Failed to open the input file")
            return None, None, None, None, None
        projection = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
        gcps = dataset.GetGCPs()
        gcp_projection = dataset.GetGCPProjection()
        spatial_ref = osr.SpatialReference()
        if projection:
            spatial_ref.ImportFromWkt(projection)
        crs = spatial_ref.ExportToWkt() if spatial_ref else None
        dataset = None
        return projection, geotransform, gcps, gcp_projection, crs
    def embed_georeference(self, output_data, projection, geotransform, gcps, gcp_projection, crs):
        """Embeds extracted georeference details into a non-georeferenced image"""
        dataset = gdal.Open(output_data, gdal.GA_Update)
        if dataset is None:
            print("Failed to open the output file for updating")
            return
        if projection:
            dataset.SetProjection(projection)
        if geotransform:
            dataset.SetGeoTransform(geotransform)
        if gcps and gcp_projection:
            dataset.SetGCPs(gcps, gcp_projection)
        if crs:
            spatial_ref = osr.SpatialReference()
            spatial_ref.ImportFromWkt(crs)
            dataset.SetProjection(spatial_ref.ExportToWkt())
        dataset = None
    def add_alpha_channel(self, output_data, input_data):
        """Adds an alpha channel to the output image while preserving georeference"""
        src_input = gdal.Open(input_data, gdal.GA_ReadOnly)
        if src_input is None:
            raise FileNotFoundError(f"Failed to open input image: {input_data}")
        projection = src_input.GetProjection()
        geotransform = src_input.GetGeoTransform()
        nodata_value = src_input.GetRasterBand(1).GetNoDataValue()
        input_array = src_input.ReadAsArray()
        if nodata_value is not None:
            nodata_mask = np.all(input_array == nodata_value, axis=0)
        else:
            nodata_mask = np.zeros((src_input.RasterYSize, src_input.RasterXSize), dtype=bool)
        src_input = None
        src_rgb = gdal.Open(output_data, gdal.GA_ReadOnly)
        if src_rgb is None:
            raise FileNotFoundError(f"Failed to open output image: {output_data}")
        output_alpha = output_data.replace(".tif", "_alpha.tif")
        extension = os.path.splitext(output_data)[-1].lower()
        driver_mapping = {".tif": "GTiff"}
        driver_name = driver_mapping.get(extension, None)
        driver = gdal.GetDriverByName(driver_name)
        dst = driver.Create(output_alpha, src_rgb.RasterXSize, src_rgb.RasterYSize, 4, gdal.GDT_Byte)
        if dst is None:
            raise RuntimeError("Failed to create output image with alpha channel")
        for i in range(3):
            band_data = src_rgb.GetRasterBand(i + 1).ReadAsArray()
            band_data[nodata_mask] = 0
            dst.GetRasterBand(i + 1).WriteArray(band_data)
        alpha_channel = np.where(nodata_mask, 0, 255).astype(np.uint8)
        dst.GetRasterBand(4).WriteArray(alpha_channel)
        dst.SetProjection(projection)
        dst.SetGeoTransform(geotransform)
        dst.FlushCache()
        dst = None
        src_rgb = None
        os.remove(output_data)
        return output_alpha
    def run_georef(self, input_data, output_data):
        """Extracts georeference details, embeds them, and adds alpha channel"""
        projection, geotransform, gcps, gcp_projection, crs = self.extract_georeference(input_data)
        if projection and geotransform:
            self.embed_georeference(output_data, projection, geotransform, gcps, gcp_projection, crs)
            return self.add_alpha_channel(output_data, input_data)
        else:
            print("Georeference details could not be extracted")'''
    def extract_georeference(self, input_data):
        """Extract GCPs, projection, and geotransform from a georeferenced image"""
        dataset = gdal.Open(input_data, gdal.GA_ReadOnly)
        if dataset is None:
            print("Failed to open the input file")
        projection = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
        gcps = dataset.GetGCPs()
        gcp_projection = dataset.GetGCPProjection()
        dataset = None
        return projection, geotransform, gcps, gcp_projection
    def embed_georeference(self, output_data, projection, geotransform, gcps, gcp_projection):
        """Embeds extracted georeference details into a non-georeferenced image"""
        dataset = gdal.Open(output_data, gdal.GA_Update)
        if dataset is None:
            print("Failed to open the output file for updating")
            return
        if projection:
            dataset.SetProjection(projection)
        if geotransform:
            dataset.SetGeoTransform(geotransform)
        if gcps and gcp_projection:
            dataset.SetGCPs(gcps, gcp_projection)
            dataset = None
    def run_georef(self, input_data, output_data):
        """Main function of georeferencing"""
        projection, geotransform, gcps, gcp_projection = self.extract_georeference(input_data)
        if projection and geotransform:
            self.embed_georeference(output_data, projection, geotransform, gcps, gcp_projection)
        else:
            print("Georeference details could not be extracted")

