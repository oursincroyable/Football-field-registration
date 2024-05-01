import numpy as np
import skimage.segmentation as seg

#define the groundtruth keypoints on the minimap
field_width = 114.83
field_height = 74.37
dw = np.linspace(0, field_width, 13)
dh = np.linspace(0, field_height, 7)
grid_dw, grid_dh = np.meshgrid(dw, dh, indexing='ij')
grid_field = np.stack((grid_dw, grid_dh), axis=2).reshape(-1,2)

#define a color palette for visualization
palette = []
for _, p in enumerate(grid_field):
    palette.append((np.rint(p[0]*255/field_width),np.rint(p[0]*255/field_width),np.rint(p[1]*255/field_height)))

# projected keypoints in football broadcast images
def project_keypoints(image, hom):
    proj_grid = np.concatenate((grid_field, np.ones((91, 1))), axis=1) @ np.linalg.inv(hom.T)
    proj_grid /= proj_grid[:, 2, np.newaxis]

    for i in range(0, len(proj_grid)):
        proj_grid[i][2] = i + 1

    # proj_img_grid = keypoints in the projected field on the image
    proj_img_grid = []
    for p in proj_grid:
        # image resolution = 1280*720
        if 0 <= p[0] < 1280 and 0 <= p[1] < 720:
            proj_img_grid.append(p)

    # produce the groundtruch label in full resolution

    label = np.zeros((720, 1280), dtype=np.float32)

    for p in proj_img_grid:
        x = np.rint(p[0]).astype(np.int32)
        y = np.rint(p[1]).astype(np.int32)

        if 0 <= x < 1280 and 0 <= y < 720:
            label[y, x] = p[2]

    label = seg.expand_labels(label, distance=20)

    for id, color in enumerate(palette):
        image[label == id+1, :] = color

    return image

#predict keypoints for test_dataset
def predict_keypoints(idx):
    input = test_dataset[idx]['pixel_values'].unsqueeze(dim=0)
    image = Image.open(test_dataset[idx]['image_path'])
    image = np.array(image)

    with torch.no_grad():
        outputs = model(input.to(device))

    predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(720,1280)])[0]
    predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()

    for i, color in enumerate(palette):
        image[predicted_segmentation_map == i+1, :] = color

    return image

#estimate homography matrix from predicted keypoints/masks
def predict_homography(label):
    classes = np.unique(label)
    classes = classes[classes != 0]

    src = []
    tgt = []
    for c in classes:
        src.append(ndimage.center_of_mass(label == c)[::-1])
        tgt.append(grid_field[c - 1])

    return cv.findHomography(np.array(src).reshape(-1, 1, 2),
                                 np.array(tgt).reshape(-1, 1, 2),
                                 method=cv.RANSAC,
                                 ransacReprojThreshold=8)[0]

compute an error of estimated homography matrices 
def error(hom_gt, hom_es):

    proj_grid = np.concatenate((grid_field, np.ones((91, 1))), axis=1) @ np.linalg.inv(hom_gt.T)
    proj_grid /= proj_grid[:, 2, np.newaxis]

    num = 0
    e = 0

    for i in range(91):
        x, y = proj_grid[i, :2]
        if 0 <= x <= 1280 and 0 <= y <= 720:
            num += 1
            X, Y, Z = np.array([x, y, 1]) @ hom_es.T
            X, Y = X/Z, Y/Z
            e += ((X-grid_field[i][0])**2 + (Y-grid_field[i][1])**2)**0.5

    return num, e
