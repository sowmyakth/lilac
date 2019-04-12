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
from mrcnn import utils
import mrcnn.model_btk_only as model_btk


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


def get_random_shift(Args, number_of_objects, maxshift=None):
    """Returns a random shift from the center in x and y coordinates
    between 0 and max-shift (in arcseconds).
    """
    if not maxshift:
        maxshift = Args.stamp_size / 10.  # in arcseconds
    dx = np.random.uniform(-maxshift, maxshift,
                           size=number_of_objects)
    dy = np.random.uniform(-maxshift, maxshift,
                           size=number_of_objects)
    return dx, dy


def resid_sampling_function(Args, catalog):
    """Randomly picks entries from input catalog that are brighter than 25.3
    mag in the i band. The centers are randomly distributed within 1/5 of the
    stamp size.
    """
    number_of_objects = 1
    a = np.hypot(catalog['a_d'], catalog['a_b'])
    cond = (a <= 1.4) & (a > 0.6)
    q_bright, = np.where(cond & (catalog['i_ab'] <= 25.3))
    if np.random.random() >= 0.9:
        q, = np.where(cond & (catalog['i_ab'] > 25.3) & (catalog['i_ab'] < 28))
    else:
        q, = np.where(cond & (catalog['i_ab'] <= 25.3))
    blend_catalog = vstack([catalog[np.random.choice(q_bright, size=1)],
                            catalog[np.random.choice(q,
                                                     size=number_of_objects)]])
    blend_catalog['ra'], blend_catalog['dec'] = 0., 0.
    # Add small shift so that center does not perfectly align with stamp center
    dx, dy = get_random_shift(Args, 1, maxshift=3*Args.pixel_scale)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    dr = np.random.uniform(3, 10)*Args.pixel_scale
    theta = np.random.uniform(0, 360) * np.pi / 180.
    dx2 = dr * np.cos(theta)
    dy2 = dr * np.sin(theta)
    blend_catalog['ra'][1] += dx2
    blend_catalog['dec'][1] += dy2
    return blend_catalog


def resid_general_sampling_function(Args, catalog):
    """Randomly picks entries from input catalog that are brighter than 25.3
    mag in the i band. The centers are randomly distributed within 1/5 of the
    stamp size.
    At least one bright galaxy (i<=24) is always selected.
    """
    number_of_objects = np.random.randint(0, Args.max_number)
    a = np.hypot(catalog['a_d'], catalog['a_b'])
    cond = (a <= 1.4) & (a > 0.6)
    q_bright, = np.where(cond & (catalog['i_ab'] <= 24))
    if np.random.random() >= 0.9:
        q, = np.where(cond & (catalog['i_ab'] < 28))
    else:
        q, = np.where(cond & (catalog['i_ab'] <= 25.3))
    blend_catalog = vstack([catalog[np.random.choice(q_bright, size=1)],
                            catalog[np.random.choice(q,
                                                     size=number_of_objects)]])
    blend_catalog['ra'], blend_catalog['dec'] = 0., 0.
    # keep number density of objects constant
    maxshift = Args.stamp_size/30.*number_of_objects**0.5
    dx, dy = get_random_shift(Args, number_of_objects + 1,
                              maxshift=maxshift)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    # Shift center of all objects so that the blend isn't exactly in the center
    dx, dy = get_random_shift(Args, 1, maxshift=5*Args.pixel_scale)
    return blend_catalog


def new_sampling_function(Args, catalog):
    """Randomly picks entries from input catalog that are brighter than 25.3
    mag in the i band. The centers are randomly distributed within 1/5 of the
    stamp size.
    """
    number_of_objects = np.random.randint(1, Args.max_number)
    a = np.hypot(catalog['a_d'], catalog['a_b'])
    cond = (a <= 1.4) & (a > 0.6)
    q_bright, = np.where(cond & (catalog['i_ab'] <= 25.3))
    q, = np.where(cond & (catalog['i_ab'] <= 26))
    blend_catalog = vstack([catalog[np.random.choice(q_bright, size=1)],
                            catalog[np.random.choice(q, size=number_of_objects)]])
    blend_catalog['ra'], blend_catalog['dec'] = 0., 0.
    dx, dy = get_random_shift(Args, number_of_objects + 1)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    return blend_catalog


def group_sampling_function(Args, catalog, min_group_size=5):
    """Blends are defined from *groups* of galaxies from the Cat-Sim like
    catalog previously analyzed with WLD. Function selects galaxies
    Note: the pre-run WLD images are not used here. We only use the pre-run
    catalog (in i band) to identify galaxies that belong to a group.

    Randomly picks entries from input catalog that are brighter than 25.3
    mag in the i band. The centers are randomly distributed within 1/5 of the
    stamp size.
    """
    if not hasattr(Args, 'wld_catalog'):
        raise Exception(
            "A pre-run WLD catalog should be input as Args.wld_catalog")
    else:
        wld_catalog = Args.wld_catalog
    group_ids = np.unique(
        wld_catalog['grp_id'][wld_catalog['grp_size'] >= min_group_size])
    group_id = np.random.choice(group_ids)
    ids = wld_catalog['db_id'][wld_catalog['grp_id'] == group_id]
    blend_catalog = vstack([catalog[catalog['galtileid'] == i] for i in ids])
    blend_catalog['ra'] -= np.mean(blend_catalog['ra'])
    blend_catalog['dec'] -= np.mean(blend_catalog['dec'])
    # convert ra dec from degrees to arcsec
    blend_catalog['ra'] *= 3600
    blend_catalog['dec'] *= 3600
    # Add small shift so that center does not perfectly align with stamp center
    dx, dy = get_random_shift(Args, 1, maxshift=3*Args.pixel_scale)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    # make sure galaxy centers don't lie too close to edge
    cond1 = np.abs(blend_catalog['ra']) < Args.stamp_size/2. - 3
    cond2 = np.abs(blend_catalog['dec']) < Args.stamp_size/2. - 3
    no_boundary = blend_catalog[cond1 & cond2]
    if len(no_boundary) == 0:
        return no_boundary
    # make sure number of galaxies in blend is less than Args.max_number
    num = min([len(no_boundary), Args.max_number])
    select = np.random.choice(range(len(no_boundary)), num, replace=False)
    return no_boundary[select]


def basic_selection_function(catalog):
    """Apply selection cuts to the input catalog"""
    a = np.hypot(catalog['a_d'], catalog['a_b'])
    q, = np.where((a <= 2) & (catalog['i_ab'] <= 26))
    return catalog[q]


def resid_obs_conditions(Args, band):
    """Returns the default observing conditions from the WLD package
    for a given survey_name and band
    Args
        Args: Class containing parameters to generate blends
        band: filter name to get observing conditions for.
    Returns
        survey: WLD survey class with observing conditions.
    """
    survey = descwl.survey.Survey.get_defaults(
        survey_name=Args.survey_name,
        filter_band=band)
    survey['zenith_psf_fwhm'] = 0.67
    survey['exposure_time'] = 5520
    survey['mirror_diameter'] = 0
    return survey


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
            batch_size=batch_size, draw_isolated=False, seed=199)
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
            param, resid_obs_conditions)
        # Generate images of blends in all the observing bands
        draw_blend_generator = btk.draw_blends.generate(
            param, blend_generator, observing_generator)
        return draw_blend_generator

