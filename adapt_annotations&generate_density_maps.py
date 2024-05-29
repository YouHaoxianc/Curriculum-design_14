import cv2
import json
import numpy as np
import os
import scipy

with open('default.json') as f:
    data = json.load(f)
image_info = []
items = data['items']
for item in range(len(items)):
    shape = cv2.imread(f'images/{items[item]['id']}.jpg').shape
    box_list = []
    point_list = []
    annotations = items[item]['annotations']
    for annotation in range(len(annotations)):
        label = annotations[annotation]['type']
        if label == 'bbox':
            bbox = annotations[annotation]['bbox']
            x1 = round(bbox[0])
            y1 = round(bbox[1])
            x2 = round(bbox[0]+bbox[2])
            y2 = round(bbox[1]+bbox[3])
            if x2 == shape[1]:
                x2 -= 1
            if y2 == shape[0]:
                y2 -= 1
            box_list.append([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
        elif label == 'points':
            point_list.append(annotations[annotation]['points'])
    image_info.append(dict(zip(['box_examples_coordinates', 'points'], [box_list, point_list])))
image_list = os.listdir('images')
data = dict(zip(image_list, image_info))
with open('annotations.json', 'w') as f:
    json.dump(data, f, indent=4)
for image in image_list:
    points = np.array(data[image]['points'])
    img = cv2.imread(f'images/{image}')
    shape = (img.shape[0], img.shape[1])
    tree = scipy.spatial.KDTree(points)
    dists, _ = tree.query(points, 2)
    avg = np.average(dists[:, 1])
    sigma = avg / 8
    density_map = np.zeros(shape, 'float32')
    for i in range(points.shape[0]):
        y = int(points[i, 1].round())
        x = int(points[i, 0].round())
        density_map[y, x] = 1
    m, n = [(ss - 1.) / 2. for ss in (avg, avg)]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    filt = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    filt[filt < np.finfo(filt.dtype).eps * filt.max()] = 0
    sum_filt = filt.sum()
    if sum_filt != 0:
        filt /= sum_filt
    density_map = cv2.filter2D(density_map, -1, filt, 0)
    if not os.path.exists('density_maps'):
        os.makedirs('density_maps')
    np.save(f'density_maps/{image.split('.')[0]}.npy', density_map)
