import argparse
import csv
import json
from multiprocessing.dummy import Pool, Lock
import os
from collections import OrderedDict
import time
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore', UserWarning)

import ee
import numpy as np
import rasterio
import urllib3
from rasterio.transform import Affine
from skimage.exposure import rescale_intensity
from torchvision.datasets.utils import download_and_extract_archive
import shapefile
from shapely.geometry import shape, Point

from datasets.seco_dataset import RGB_BANDS, ALL_BANDS


class GeoSampler:

    def sample_point(self):
        raise NotImplementedError()


class UniformSampler(GeoSampler):

    def sample_point(self):
        lon = np.random.uniform(-180, 180)
        lat = np.random.uniform(-90, 90)
        return [lon, lat]


class GaussianSampler(GeoSampler):

    def __init__(self, interest_points=None, std=50):
        if interest_points is None:
            cities = self.get_world_cities()
            self.interest_points = self.get_interest_points(cities)
        else:
            self.interest_points = interest_points
        self.std = std

    def sample_point(self):
        rng = np.random.default_rng()
        point = rng.choice(self.interest_points)
        std = self.km2deg(self.std)
        lon, lat = np.random.normal(loc=point, scale=[std, std])
        return [lon, lat]

    @staticmethod
    def get_world_cities(download_root=os.path.expanduser('~/.cache/simplemaps')):
        url = 'https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.71.zip'
        filename = 'worldcities.csv'
        if not os.path.exists(os.path.join(download_root, os.path.basename(url))):
            download_and_extract_archive(url, download_root)
        with open(os.path.join(download_root, filename)) as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            cities = []
            for row in reader:
                row['population'] = row['population'].replace('.', '') if row['population'] else '0'
                cities.append(row)
        return cities

    @staticmethod
    def get_interest_points(cities, size=10000):
        cities = sorted(cities, key=lambda c: int(c['population']), reverse=True)[:size]
        points = [[float(c['lng']), float(c['lat'])] for c in cities]
        return points

    @staticmethod
    def km2deg(kms, radius=6371):
        return kms / (2.0 * radius * np.pi / 360.0)


class BoundedUniformSampler(GeoSampler):

    def __init__(self, boundaries=None):
        if boundaries is None:
            self.boundaries = self.get_country_boundaries()
        else:
            self.boundaries = boundaries

    def sample_point(self):
        minx, miny, maxx, maxy = self.boundaries.bounds
        lon = np.random.uniform(minx, maxx)
        lat = np.random.uniform(miny, maxy)
        p = Point(lon, lat)
        if self.boundaries.contains(p):
            return [p.x, p.y]
        else:
            return self.sample_point()

    @staticmethod
    def get_country_boundaries(download_root=os.path.expanduser('~/.cache/naturalearth')):
        url = 'https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/cultural/ne_110m_admin_0_countries.zip'
        filename = 'ne_110m_admin_0_countries.shp'
        if not os.path.exists(os.path.join(download_root, os.path.basename(url))):
            download_and_extract_archive(url, download_root)
        sf = shapefile.Reader(os.path.join(download_root, filename))
        return shape(sf.shapes().__geo_interface__)


def maskS2clouds(image):
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    mask = mask.bitwiseAnd(cirrusBitMask).eq(0)

    return image.updateMask(mask)


def get_collection(cloud_pct=20):
    collection = ee.ImageCollection('COPERNICUS/S2_SR')
    # collection = collection.filterDate('2017-03-28', datetime.today().strftime('%Y-%m-%d'))
    collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
    collection = collection.map(maskS2clouds)
    return collection


def filter_collection(collection, coords, period=None):
    filtered = collection
    if period is not None:
        filtered = filtered.filterDate(*period)  # filter time
    filtered = filtered.filterBounds(ee.Geometry.Point(coords))  # filter region
    if filtered.size().getInfo() == 0:
        raise ee.EEException(
            f'ImageCollection.filter: No suitable images found in ({coords[1]:.4f}, {coords[0]:.4f}) between {period[0]} and {period[1]}.')
    return filtered


def center_crop(img, out_size):
    image_height, image_width = img.shape[:2]
    crop_height, crop_width = out_size
    crop_top = int((image_height - crop_height + 1) * 0.5)
    crop_left = int((image_width - crop_width + 1) * 0.5)
    return img[crop_top:crop_top+crop_height, crop_left:crop_left+crop_width]


def adjust_coords(coords, old_size, new_size):
    xres = (coords[1][0] - coords[0][0]) / old_size[1]
    yres = (coords[0][1] - coords[1][1]) / old_size[0]
    xoff = int((old_size[1] - new_size[1] + 1) * 0.5)
    yoff = int((old_size[0] - new_size[0] + 1) * 0.5)
    return [
        [coords[0][0] + (xoff * xres), coords[0][1] - (yoff * yres)],
        [coords[0][0] + ((xoff + new_size[1]) * xres), coords[0][1] - ((yoff + new_size[0]) * yres)]
    ]


def get_properties(image):
    properties = {}
    for property in image.propertyNames().getInfo():
        properties[property] = image.get(property)
    return ee.Dictionary(properties).getInfo()


def get_patch(collection, coords, radius, bands=None, crop=None):
    if bands is None:
        bands = RGB_BANDS

    image = collection.sort('system:time_start', False).first()  # get most recent
    region = ee.Geometry.Point(coords).buffer(radius).bounds()
    patch = image.divide(10000).select(*bands).sampleRectangle(region)
    features = patch.getInfo()  # the actual download

    raster = OrderedDict()
    for band in bands:
        img = np.atleast_3d(features['properties'][band])
        if crop is not None:
            img = center_crop(img, out_size=crop[band])
        img = rescale_intensity(img, in_range=(0, 1), out_range=np.uint8)
        raster[band] = img

    coords = np.array(features['geometry']['coordinates'][0])
    coords = [
        [coords[:, 0].min(), coords[:, 1].max()],
        [coords[:, 0].max(), coords[:, 1].min()]
    ]
    if crop is not None:
        band = bands[0]
        old_size = (len(features['properties'][band]), len(features['properties'][band][0]))
        new_size = raster[band].shape[:2]
        coords = adjust_coords(coords, old_size, new_size)

    return OrderedDict({
        'raster': raster,
        'coords': coords,
        'metadata': get_properties(image)
    })


def get_random_patch(collection, sampler, debug=False, **kwargs):
    coords = sampler.sample_point()
    try:
        patch = get_patch(filter_collection(collection, coords), coords, **kwargs)
    except (ee.EEException, urllib3.exceptions.HTTPError) as e:
        if debug:
            print(e)
        patch = get_random_patch(collection, sampler, debug, **kwargs)
    return patch


def date2str(date):
    return date.strftime('%Y-%m-%d')


def get_period(date, days=5):
    date1 = date - timedelta(days=days / 2)
    date2 = date + timedelta(days=days / 2)
    return date2str(date1), date2str(date2)


def get_random_patches(collection, sampler, dates, debug=False, **kwargs):
    coords = sampler.sample_point()
    delta = timedelta(days=np.random.randint(365))
    periods = [get_period(date - delta, days=15) for date in dates]
    try:
        filtered_collections = [filter_collection(collection, coords, p) for p in periods]
        patches = [get_patch(c, coords, **kwargs) for c in filtered_collections]
    except (ee.EEException, urllib3.exceptions.HTTPError) as e:
        if debug:
            print(e)
        patches = get_random_patches(collection, sampler, dates, debug, **kwargs)
    return patches


def save_geotiff(img, coords, filename):
    height, width, channels = img.shape
    xres = (coords[1][0] - coords[0][0]) / width
    yres = (coords[0][1] - coords[1][1]) / height
    transform = Affine.translation(coords[0][0] - xres / 2, coords[0][1] + yres / 2) * Affine.scale(xres, -yres)
    profile = {
        'driver': 'GTiff',
        'width': width,
        'height': height,
        'count': channels,
        'crs': '+proj=latlong',
        'transform': transform,
        'dtype': img.dtype,
        'compress': 'JPEG'
    }
    with rasterio.open(filename, 'w', **profile) as f:
        f.write(img.transpose(2, 0, 1))


def save_patch(raster, coords, metadata, path, preview=False):
    patch_id = metadata['system:index']
    patch_path = os.path.join(path, patch_id)
    os.makedirs(patch_path, exist_ok=True)

    for band, img in raster.items():
        save_geotiff(img, coords, os.path.join(patch_path, f'{band}.tif'))

    if preview:
        rgb = np.dstack([raster[band] for band in RGB_BANDS])
        rgb = rescale_intensity(rgb, in_range=(0, 255 * 0.3), out_range=(0, 255)).astype(np.uint8)
        save_geotiff(rgb, coords, os.path.join(path, f'{patch_id}.tif'))

    with open(os.path.join(patch_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)


class Counter:

    def __init__(self, start=0):
        self.value = start
        self.lock = Lock()

    def update(self, delta=1):
        with self.lock:
            self.value += delta
            return self.value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_locations', type=int, default=200000)
    parser.add_argument('--cloud_pct', type=int, default=10)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--indices_file', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    ee.Initialize()
    collection = get_collection(cloud_pct=args.cloud_pct)

    sampler = GaussianSampler()

    # sampler = BoundedUniformSampler()
    # worker = lambda x: sampler.sample_point()
    # with Pool(processes=args.num_workers) as p:
    #     points = p.map(worker, range(10000))
    # sampler = GaussianSampler(interest_points=points, std=500)

    reference = datetime.today() - timedelta(weeks=4)
    date1 = reference - timedelta(weeks=3 * 4)  # 3 months ago
    date2 = reference - timedelta(weeks=6 * 4)  # 6 months ago
    date3 = reference - timedelta(weeks=9 * 4)  # 9 months ago
    date4 = reference - timedelta(weeks=12 * 4)  # 1 year ago
    dates = [reference, date1, date2, date3, date4]

    crop10 = (264, 264)
    crop20 = (132, 132)
    crop60 = (44, 44)
    crop = {'B1': crop60, 'B2': crop10, 'B3': crop10, 'B4': crop10, 'B5': crop20, 'B6': crop20, 'B7': crop20,
            'B8': crop10, 'B8A': crop20, 'B9': crop60, 'B11': crop20, 'B12': crop20}

    start_time = time.time()
    counter = Counter()

    def worker(idx):
        patches = get_random_patches(collection, sampler, dates, radius=1325, bands=ALL_BANDS, crop=crop, debug=args.debug)
        if args.save_path is not None:
            location_path = os.path.join(args.save_path, f'{idx:06d}')
            os.makedirs(location_path, exist_ok=True)
            for patch in patches:
                save_patch(
                    raster=patch['raster'],
                    coords=patch['coords'],
                    metadata=patch['metadata'],
                    path=location_path,
                    preview=args.preview
                )
        count = counter.update(len(patches))
        if count % args.log_freq == 0:
            print(f'Downloaded {count} images in {time.time() - start_time:.3f}s.')
        return

    if args.indices_file is not None:
        indices = map(int, open(args.indices_file).readlines())
    else:
        indices = range(args.num_locations)

    if args.num_workers == 0:
        for i in indices:
            worker(i)
    else:
        with Pool(processes=args.num_workers) as p:
            p.map(worker, indices)
