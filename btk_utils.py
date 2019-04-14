"""Contains functions to perform detection, deblending and measurement
    on images with BlendingToolKit(btk).
"""
import sep
import btk
import numpy as np
import scarlet
from scipy import spatial
import descwl
import matplotlib.pyplot as plt
from astropy.table import vstack
import utils, model, config


def get_ax(rows=1, cols=1, size=4):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


class LilacDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def __init__(self, draw_generator, norm_val=None,
                 augmentation=False, *args, **kwargs):
        super(LilacDataset, self).__init__(*args, **kwargs)
        self.draw_generator = draw_generator
        self.augmentation = augmentation
        if norm_val:
            self.mean = norm_val[0]
            self.std = norm_val[1]
        else:
            self.mean = 0
            self.std = 1

    def get_targets(self, isolated_images, pixel_threshold):
        """Returns target for network to predict i.e images of isolated objects
        and their bounding boxes.

        Profile of objects in isolated_images is limited to pixel_threshold in
        the i band. All pixel values in the i band lower than pixel_threshold
        are set to zero. Bounding box corresponding to the object is determined
        from the minimum and maximum x and y coordinates of non zero pixels.
        The object SBP is limited to inside this bounding box for *all* bands.
        The network will aim to produce images of objects within it's bounding
        box.

        Args:
            isolated_images: Images of objects without any contribution from
                             neighboring sources, as if they were isolated.
                             [band, num_objects, x, y]
            pixel_threshold: pixels with value greater than or equal to are set
                             to 1, lesser to 0, in i band images.

        Returns:
            target_images: isolated images within a mask determined by the
                           pixel_threshold.
            bboxes: x, y coordinates of bottom left and top right corners for
                    each object [num_objects, [y1, x1, y2, x2]]

        """
        # apply pixel threshold to i band images
        i_images = isolated_images[3]
        i_images[i_images < pixel_threshold] = 0
        i_images[i_images >= pixel_threshold] = 1
        target_images = np.zeros_like(isolated_images)
        bboxes = np.zeros([isolated_images.shape[1], 4])
        for i in range(isolated_images.shape[0]):
            arr = i_images[i]
            horizontal_indices = np.where(np.any(arr, axis=0))[0]
            vertical_indices = np.where(np.any(arr, axis=1))[0]
            x1, x2 = horizontal_indices[0], horizontal_indices[-1]
            y1, y2 = vertical_indices[0], vertical_indices[-1]
            target_images[i, x1:x2, y1:y1] = isolated_images[i, x1:x2, y1:y1]
            bboxes[i] = [y1, x1, y2+1, x2+1]
        return target_images, bboxes

    def load_data(self, count=None):
        """loads training and test input and output data
        Keyword Arguments:
            filename -- Numpy file where data is saved
        """
        if not count:
            count = 240
        self.load_objects(count)
        print("Loaded {} blends".format(count))

    def load_objects(self, count):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("galaxy", 1, "object")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            self.add_image("galaxy", image_id=i, path=None,
                           object="object")

    def normalize_images(self, images):
        images = (images - self.mean)/self.std
        return images

    def augment_bbox(self, bboxes, end_pixel):
        mult_y = np.array([0, 0, 1, 1])
        mult_x = np.array([0, 1, 0, 1])
        h0 = (bboxes[:, 2] - bboxes[:, 0]) / 2.
        w0 = (bboxes[:, 3] - bboxes[:, 1]) / 2.
        x0 = np.mean(bboxes[:, 1::2], axis=1)
        y0 = np.mean(bboxes[:, ::2], axis=1)
        aug_bbox = np.zeros((4, len(bboxes), 4), dtype=np.int32)
        for i in range(len(mult_x)):
            new_x0 = np.abs(end_pixel*mult_x[i] - x0)
            new_y0 = np.abs(end_pixel*mult_y[i] - y0)
            new_x0[x0 == 0.5] = 0.5
            new_y0[x0 == 0.5] = 0.5
            new_bbox = np.array(
                [new_y0 - h0, new_x0 - w0, new_y0 + h0, new_x0 + w0])
            new_bbox = np.transpose(new_bbox, axes=(1, 0,))
            aug_bbox[i] = new_bbox
        return aug_bbox

    def augment_data(self, images, isolated_images, bboxes, class_ids):
        """Performs data augmentation by performing rotation and reflection"""
        aug_image = np.stack([images[:, :, :],
                              images[:, ::-1, :],
                              images[::-1, :, :],
                              images[::-1, ::-1, :]])
        aug_isolated_image = np.stack([isolated_images[:, :, :],
                                       isolated_images[:, ::-1, :],
                                       isolated_images[::-1, :, :],
                                       isolated_images[::-1, ::-1, :]])
        aug_bbox = self.augment_bbox(bboxes, images.shape[1] - 1)
        aug_class = np.stack([class_ids, class_ids, class_ids, class_ids])
        return aug_image, aug_isolated_image, aug_bbox, aug_class

    def load_input(self):
        """Generates image + bbox for undetected objects if any"""
        output, = next(self.draw_generator)
        self.batch_blend_list = output['blend_list']
        self.obs_cond = output['obs_condition']
        input_images, input_isolated_images = [], []
        input_bboxes, input_class_ids = [], []
        self.det_cent, self.true_cent = [], []
        for i in range(len(output['blend_list'])):
            blend_images = output['blend_images'][i]
            blend_list = output['blend_list'][i]
            isolated_images, bboxes = self.get_targets(
                output['isolated_images'][i],
                self.obs_cond[i][3].mean_sky_level)
            self.true_cent.append(
                np.stack([blend_list['dx'], blend_list['dy']]).T)
            bboxes = np.concatenate((bboxes, [[0, 0, 1, 1]]))
            class_ids = np.concatenate((np.ones(len(bboxes),
                                       dtype=np.int32), [0]))
            if self.augmentation:
                aug_output = self.augment_data(blend_images, isolated_images,
                                               bboxes, class_ids)
                input_images.extend(aug_output[0])
                input_isolated_images.extend(aug_output[1])
                input_bboxes.extend(aug_output[2])
                input_class_ids.extend(aug_output[3])
            else:
                input_images.append(blend_images)
                input_isolated_images.extend(isolated_images)
                input_bboxes.append(bboxes)
                input_class_ids.append(class_ids)
        input_images = self.normalize_images(input_images.astype(np.float32))
        input_isolated_images = self.normalize_images(
            input_isolated_images.astype(np.float32))
        return input_images, input_isolated_images, input_bboxes, input_class_ids

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "resid":
            return info["object"]
        else:
            super(self.__class__).image_reference(self, image_id)


def make_draw_generator(catalog_name, batch_size, max_number,
                        sampling_function, selection_function=None,
                        wld_catalog=None):
        """
        Creates the default btk.meas_generator for input catalog
        Args:
            catalog_name: CatSim like catalog to draw galaxies from.
            max_number: Maximum number of galaxies per blend.
            sampling_function: Function describing how galaxies are drawn from
                               catalog_name.
            wld_catalog: A WLD pre-run astropy table. Used if sampling function
                         requires grouped objects.
        """
        # Load parameters
        param = btk.config.Simulation_params(
            catalog_name, max_number=max_number, stamp_size=25.6,
            batch_size=batch_size, seed=199)
        if wld_catalog:
            param.wld_catalog = wld_catalog
        print("setting seed", param.seed)
        np.random.seed(param.seed)
        # Load input catalog
        catalog = btk.get_input_catalog.load_catalog(
            param, selection_function=selection_function)
        # Generate catalogs of blended objects
        blend_generator = btk.create_blend_generator.generate(
            param, catalog, sampling_function)
        # Generates observing conditions
        observing_generator = btk.create_observing_generator.generate(
            param)
        # Generate images of blends in all the observing bands
        draw_blend_generator = btk.draw_blends.generate(
            param, blend_generator, observing_generator)
        return draw_blend_generator


class lilac_btk_model(btk.compute_metrics.Metrics_params):
    def __init__(self,  model_name, model_path, output_dir,
                 training=False, new_model_name=None, images_per_gpu=1,
                 validation_for_training=False, *args, **kwargs):
        super(lilac_btk_model, self).__init__(*args, **kwargs)
        self.training = training
        self.model_path = model_path
        self.output_dir = output_dir
        self.validation_for_training = validation_for_training
        #file_name = "train" + model_name
        #train = __import__(file_name)

        #class InferenceConfig(train.InputConfig):
        class InferenceConfig(config.Config):
            GPU_COUNT = 1
            IMAGES_PER_GPU = images_per_gpu
            STEPS_PER_EPOCH = 500  # 200
            VALIDATION_STEPS = 20
            IMAGE_MIN_DIM = 128
            IMAGE_MAX_DIM = 128
            RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
            MEAN_PIXEL = np.zeros(6)
            IMAGE_SHAPE = np.array([128, 128, 6])
            #RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2])
            #BBOX_STD_DEV = np.array([0.1, 0.1, 0.2])
            if new_model_name:
                NAME = new_model_name

        self.config = InferenceConfig()
        if self.training:
            self.config.display()

    def make_model(self, catalog_name, count=256,
                         sampling_function=None, max_number=2,
                         augmentation=False, norm_val=None,
                         selection_function=None, wld_catalog=None,
                         meas_params=None):
        """Creates dataset and loads model"""
        # If no user input sampling function then set default function
        self.draw_generator = make_draw_generator(catalog_name,
                                                  self.config.BATCH_SIZE,
                                                  max_number,
                                                  sampling_function,
                                                  selection_function,
                                                  wld_catalog,
                                                  )
        self.dataset = LilacDataset(self.draw_generator, norm_val=norm_val,
                                    augmentation=augmentation)
        self.dataset.load_data(count=count)
        self.dataset.prepare()
        if augmentation:
            self.config.BATCH_SIZE *= 4
            self.config.IMAGES_PER_GPU *= 4
        if self.training:
            self.model = model.MaskRCNN(mode="training",
                                            config=self.config,
                                            model_dir=self.output_dir)
            if self.validation_for_training:
                val_meas_generator = make_draw_generator(catalog_name,
                                                         self.config.BATCH_SIZE,
                                                         max_number,
                                                         sampling_function,
                                                         selection_function,
                                                         wld_catalog,
                                                         )
                self.dataset_val = LilacDataset(val_meas_generator,
                                                norm_val=norm_val)
                self.dataset_val.load_data(count=count)
                self.dataset_val.prepare()
        else:
            print(self.config.DETECTION_MIN_CONFIDENCE)
            self.model = model.MaskRCNN(mode="inference",
                                            config=self.config,
                                            model_dir=self.output_dir)
        if self.model_path:
            print("Loading weights from ", self.model_path)
            self.model.load_weights(self.model_path, by_name=True)

