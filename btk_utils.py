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
    def __init__(self, meas_generator, norm_val=None,
                 augmentation=False, *args, **kwargs):
        super(LilacDataset, self).__init__(*args, **kwargs)
        self.meas_generator = meas_generator
        self.augmentation = augmentation
        if norm_val:
            self.mean = norm_val[0]
            self.std = norm_val[1]
        else:
            self.mean = 0
            self.std = 1

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
        x0 = np.mean(bboxes[:, 1::2], axis=1)
        y0 = np.mean(bboxes[:, ::2], axis=1)
        aug_bbox = np.zeros((4, len(bboxes), 4), dtype=np.int32)
        for i in range(len(mult_x)):
            new_x0 = np.abs(end_pixel*mult_x[i] - x0)
            new_y0 = np.abs(end_pixel*mult_y[i] - y0)
            new_x0[x0 == 0.5] = 0.5
            new_y0[x0 == 0.5] = 0.5
            new_bbox = np.array(
                [new_y0 - h0, new_x0 - h0, new_y0 + h0, new_x0 + h0])
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
        output, deb, _ = next(self.meas_generator)
        self.batch_blend_list = output['blend_list']
        self.obs_cond = output['obs_condition']
        input_images, input_isolated_images = [], []
        input_bboxes, input_class_ids = [], []
        self.det_cent, self.true_cent = [], []
        for i in range(len(output['blend_list'])):
            blend_images = output['blend_images'][i]
            blend_list = output['blend_list'][i]
            isolated_images = output['isolated_images'][i]
            self.true_cent.append(
                np.stack([blend_list['dx'], blend_list['dy']]).T)
            x, y, h = get_targets(blend_list, self.obs_cond[i][3])
            bboxes = np.array([y, x, y + h, x + h], dtype=np.int32).T
            bboxes = np.concatenate((bbox, [[0, 0, 1, 1]]))
            class_ids = np.concatenate((np.ones(len(x), dtype=np.int32), [0]))
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


# Check if needed!!!!


def resid_merge_centers(det_cent, bbox,
                        distance_upper_bound=1, center_shift=0):
    """Combines centers from detection algorithm and iteratively
    detected centers. Also corrects for shift of 4 pixels in center
    Args:
        det_cent: centers detected by detection algorithm.
        bbox: Edges of ResidDetectron bounding box (y1, x1, y2, x2).
        distance_upper_bound: If network prediction is within this distance of
                              a det_cent, select the network prediction and
                              remove det_cent from final merged predictions.
        center_shift: Value to offset the bbox centers by. Applicable if
            padding was applied to the residual image causing detected centers
            and bounding box centers to offset.
    """
    # remove duplicates
    if len(bbox) == 0:
        return det_cent
    q, = np.where(
        (bbox[:, 0] > 3+center_shift) & (bbox[:, 1] > 3+center_shift))
    # centers of bbox as mean of edges
    resid_det = np.dstack([np.mean(bbox[q, 1::2], axis=1) - center_shift,
                           np.mean(bbox[q, ::2], axis=1) - center_shift])[0]
    unique_det_cent = np.unique(det_cent, axis=0)
    if len(unique_det_cent) == 0:
        return resid_det
    z_tree = spatial.KDTree(unique_det_cent)
    resid_det = resid_det.reshape(-1, 2)
    match = z_tree.query(resid_det,
                         distance_upper_bound=distance_upper_bound)
    trim = np.setdiff1d(range(len(unique_det_cent)), match[1])
    trim_det_cent = [unique_det_cent[i] for i in trim]
    if len(trim_det_cent) == 0:
        return resid_det
    iter_det = np.vstack([trim_det_cent, resid_det])
    return iter_det


def get_undetected(true_cat, meas_cent, obs_cond, distance_upper_bound=10):
    """Returns bounding boxes for galaxies that were undetected. The bounding
    boxes are square with height set as twice the PSF convolved HLR. Since
    CatSim catalog has separate bulge and disk parameters, the galaxy HLR is
    approximated as the flux weighted average of bulge and disk HLR.

    The function returns the x and y coordinates of the lower left corner of
    box, and the height of each undetected galaxy. A galaxy is marked as
    undetected if no detected center lies within distance_upper_bound of it's
    true center.
    Args:
        true_cat: CatSim-like catalog of true galaxies.
        meas_cent: ndarray of x and y coordinates of detected centers.
        obs_cond: wld.survey class corresponding to observing conditions in the
                  band in which PSF convolved HLR is to be estimated.
        distance_upper_bound: Distance up-to which a detected center can be
                              matched to a true center.
    Returns:
        Bounding box of undetected galaxies

    """
    psf_sigma = obs_cond.zenith_psf_fwhm*obs_cond.airmass**0.6
    pixel_scale = obs_cond.pixel_scale
    peaks = np.stack([true_cat['dx'], true_cat['dy']]).T
    if len(peaks) == 0:
        undetected = range(len(true_cat))
    else:
        z_tree = spatial.KDTree(peaks)
        meas_cent = np.array(meas_cent).reshape(-1, 2)
        match = z_tree.query(meas_cent,
                             distance_upper_bound=distance_upper_bound)
        undetected = np.setdiff1d(range(len(true_cat)), match[1])
    numer = true_cat['a_d']*true_cat['b_d']*true_cat['fluxnorm_disk'] + true_cat['a_b']*true_cat['b_b']*true_cat['fluxnorm_bulge']
    hlr = numer / (true_cat['fluxnorm_disk'] + true_cat['fluxnorm_bulge'])
    h = np.hypot(hlr, 1.18*psf_sigma)*2 / pixel_scale
    h = np.array(h, dtype=np.int32)
    x0 = true_cat['dx'] - h/2
    y0 = true_cat['dy'] - h/2
    return x0[undetected], y0[undetected], h[undetected]


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


def scarlet_initialize(images, peaks,
                       bg_rms, iters, e_rel):
    """ Deblend input images with scarlet
    Args:
        images: Numpy array of multi-band image to run scarlet on
               [Number of bands, height, width].
        peaks: Array of x and y coordinates of centroids of objects in
               the image [number of sources, 2].
        bg_rms: Background RMS value of the images [Number of bands]
        iters: Maximum number of iterations if scarlet doesn't converge
               (Default: 200).
    e_rel: Relative error for convergence (Default: 0.015)
    Returns
        blend: scarlet.Blend object for the initialized sources
        rejected_sources: list of sources (if any) that scarlet was
        unable to initialize the image with.
    """
    sources = []
    for n, peak in enumerate(peaks):
        try:
            result = scarlet.ExtendedSource(
                (peak[1], peak[0]),
                images,
                bg_rms)
            sources.append(result)
        except scarlet.source.SourceInitError:
            print("No flux in peak {0} at {1}".format(n, peak))
    blend = scarlet.Blend(sources).set_data(images, bg_rms=bg_rms)
    return blend


def scarlet_multi_initialize(images, peaks,
                             bg_rms, iters, e_rel):
    """ Initializes scarlet MultiComponentSource at locations input as
    peaks in the (multi-band) input images.
    Args:
        images: Numpy array of multi-band image to run scarlet on
                [Number of bands, height, width].
        peaks: Array of x and y coordinates of centroids of objects in
               the image [number of sources, 2].
        bg_rms: Background RMS value of the images [Number of bands]
    Returns
        blend: scarlet.Blend object for the initialized sources
        rejected_sources: list of sources (if any) that scarlet was
                          unable to initialize the image with.
    """
    sources = []
    for n, peak in enumerate(peaks):
        try:
            result = scarlet.MultiComponentSource(
                (peak[1], peak[0]),
                images,
                bg_rms)
            sources.append(result)
        except scarlet.source.SourceInitError:
            print("No flux in peak {0} at {1}".format(n, peak))
    blend = scarlet.Blend(sources).set_data(images, bg_rms=bg_rms)
    return blend


def scarlet_fit(images, peaks,
                bg_rms, iters, e_rel):
    """Fits a scarlet model for the input image and centers"""
    try:
        blend = scarlet_multi_initialize(images, peaks,
                                         bg_rms, iters, e_rel)
        blend.fit(iters, e_rel=e_rel)
    except (np.linalg.LinAlgError, ValueError):
        blend = scarlet_initialize(images, peaks,
                                   bg_rms, iters, e_rel)
        try:
            blend.fit(iters, e_rel=e_rel)
        except(np.linalg.LinAlgError, ValueError):
            print("scarlet did not fit")
    return blend


class Scarlet_resid_params(btk.measure.Measurement_params):
    def __init__(self, iters=400, e_rel=.015,
                 detect_centers=True, detect_coadd=False,
                 *args, **kwargs):
        super(Scarlet_resid_params, self).__init__(*args, **kwargs)
        self.iters = iters
        self.e_rel = e_rel
        self.detect_centers = detect_centers
        self.detect_coadd = detect_coadd

    def get_centers_coadd(self, image):
        """Runs SEP on coadd of input image and returns detected centroids
        Args:
            image: Input image (multi-band) to perform detection on
                   [bands, x, y].
        Returns:
            x and y coordinates of detected centroids.
        """
        detect = image.mean(axis=0)  # simple average for detection
        bkg = sep.Background(detect)
        catalog = sep.extract(detect, 1.5, err=bkg.globalrms)
        q, = np.where((catalog['x'] > 0) & (catalog['y'] > 0))
        return np.stack((catalog['x'][q], catalog['y'][q]), axis=1)

    def get_centers_i_band(self, image):
        """Runs SEP on i band of input image and returns detected centroids
        Args:
            image: Input image (multi-band) to perform detection on
                   [bands, x, y].
        Returns:
            x and y coordinates of detected centroids.
        """
        detect = image[3]  # simple average for detection
        bkg = sep.Background(detect)
        catalog = sep.extract(detect, 1.5, err=bkg.globalrms)
        q, = np.where((catalog['x'] > 0) & (catalog['y'] > 0))
        return np.stack((catalog['x'][q], catalog['y'][q]), axis=1)

    def get_deblended_images(self, data, index):
        """Returns scarlet modeled blend  and centers for the given blend"""
        images = np.transpose(data['blend_images'][index], axes=(2, 0, 1))
        blend_cat = data['blend_list'][index]
        if self.detect_centers:
            if self.detect_coadd:
                peaks = self.get_centers_i_band(images)
            else:
                peaks = self.get_centers_coadd(images)
        else:
            peaks = np.stack((blend_cat['dx'], blend_cat['dy']), axis=1)
        if len(peaks) == 0:
            return [data['blend_images'][index], peaks]
        bg_rms = [data['obs_condition'][index][i].mean_sky_level**0.5 for i in range(len(images))]
        blend = scarlet_fit(images, peaks, np.array(bg_rms),
                            self.iters, self.e_rel)
        selected_peaks = [[src.center[1], src.center[0]]for src in blend.components]
        try:
            model = np.transpose(blend.get_model(), axes=(1, 2, 0))
        except(ValueError):
            print("Unable to create scarlet model")
            temp_model = np.zeros_like(data['blend_images'][index])
            return [temp_model, []]
        return [model, selected_peaks]


def get_psf_sky(obs_cond, psf_stamp_size):
    """Returns PSF image and mean background sky level for input obs_condition.
    Args:
        obs_cond: wld.survey class corresponding to observing conditions in the
                  band in which PSF convolved HLR is to be estimated.
        psf_stamp_size: Size of image to draw PSF model into (pixels).
    Returns:
        PSF image and mean background sky level
    """
    mean_sky_level = obs_cond.mean_sky_level
    psf = obs_cond.psf_model
    psf_image = psf.drawImage(
       nx=psf_stamp_size,
       ny=psf_stamp_size).array
    return psf_image, mean_sky_level


def get_stack_input(image, obs_cond, psf_stamp_size, detect_coadd):
    """Returns input for running stack detection on either coadd image or
    i band.
    Args:
        image: Input image (multi-band) to perform detection on
               [bands, x, y].
        obs_cond: wld.survey class corresponding to observing conditions in the
                  band in which PSF convolved HLR is to be estimated.
        psf_stamp_size: Size of image to draw PSF model into (pixels).
        detect_coadd: If True then detection (and measurement) is
                      performed on coadd over bands
    Returns
        image, variance image and psf image
    """
    if detect_coadd:
        input_image = np.zeros(image.shape[1:2], dtype=np.float32)
        variance_image = np.zeros(image.shape[1:2], dtype=np.float32)
        for i in range(len(obs_cond)):
            psf_image, mean_sky_level = get_psf_sky(obs_cond[i],
                                                    psf_stamp_size)
            variance_image += image[:, :, i] + mean_sky_level
            input_image += image[:, :, i]
    else:
        i = 3  # detection in i band
        psf_image, mean_sky_level = get_psf_sky(obs_cond[i],
                                                psf_stamp_size)
        variance_image += image[:, :, i] + mean_sky_level
        input_image += image[:, :, i]
    # since PSF is same for all bands, PSF of coadd is the same
    return input_image, variance_image, psf_image


def get_stack_catalog(image, obs_cond, detect_coadd=False,
                      psf_stamp_size=41, min_pix=1,
                      thr_value=5, bkg_bin_size=32):
    """Perform detection, deblending and measurement on the i band image of
    the blend image for input index in the batch.
    Args:
        image: Input image (multi-band) to perform detection on
               [bands, x, y].
        obs_cond: wld.survey class corresponding to observing conditions in the
                  band in which PSF convolved HLR is to be estimated.
        detect_coadd: If True then detection (and measurement) is
                      performed on coadd over bands
    Returns
        Stack detection (+ measurement) output catalog
    """
    image_array, variance_array, psf_image = get_stack_input(
        image, obs_cond, psf_stamp_size, detect_coadd)
    psf_array = psf_image.astype(np.float64)
    cat = btk.utils.run_stack(
        image_array, variance_array, psf_array, min_pix=min_pix,
        bkg_bin_size=bkg_bin_size, thr_value=thr_value)
    cat_chldrn = cat[
        (cat['deblend_nChild'] == 0) & (cat['base_SdssCentroid_flag'] == False)]
    cat_chldrn = cat_chldrn.copy(deep=True)
    return cat_chldrn.asAstropy()


def get_stack_centers(catalog):
    """Returns stack detected centroids from detection catalog.
    Args:
        catalog: Stack detection output catalog
    Returns:
        x and y coordinates of detected centroids.
    """
    xs = catalog['base_SdssCentroid_y']
    ys = catalog['base_SdssCentroid_x']
    q, = np.where((xs > 0) & (ys > 0))
    return np.stack((ys[q], xs[q]), axis=1)


class Stack_iter_params(btk.measure.Measurement_params):
    min_pix = 1
    bkg_bin_size = 32
    thr_value = 5
    psf_stamp_size = 41
    iters = 200
    e_rel = .015

    def __init__(self, detect_coadd=False, *args, **kwargs):
        super(Stack_iter_params, self).__init__(*args, **kwargs)
        self.detect_coadd = detect_coadd
        self.catalog = {}

    def get_deblended_images(self, data, index):
        """Returns scarlet modeled blend  and centers for the given blend"""
        images = np.transpose(data['blend_images'][index], axes=(2, 0, 1))
        catalog = get_stack_catalog(data['blend_images'][index],
                                    data['obs_condition'][index],
                                    detect_coadd=self.detect_coadd,
                                    psf_stamp_size=self.psf_stamp_size,
                                    min_pix=self.min_pix,
                                    bkg_bin_size=self.bkg_bin_size,
                                    thr_value=self.thr_value)

        self.catalog[index] = catalog
        peaks = get_stack_centers(catalog)
        if len(peaks) == 0:
            print("Unable to create scarlet model, no peaks")
            temp_model = np.zeros_like(data['blend_images'][index])
            return [temp_model, peaks]
        bg_rms = [data['obs_condition'][index][i].mean_sky_level**0.5 for i in range(len(images))]
        try:
            blend = scarlet_fit(images, peaks,
                                np.array(bg_rms), self.iters,
                                self.e_rel)
            selected_peaks = [[src.center[1], src.center[0]]for src in blend.components]
            model = np.transpose(blend.get_model(), axes=(1, 2, 0))
        except(ValueError, IndexError) as e:
            print("Unable to create scarlet model")
            temp_model = np.zeros_like(data['blend_images'][index])
            return [temp_model, []]
        return [model, selected_peaks]

    def make_measurement(self, data, index):
        """ Returns catalog from the deblending step which involved performing
        detection, deblending and measurement on the i band image of
        the blend image for input index in the batch using the DM stack.
         """
        return self.catalog[index]


def make_meas_generator(catalog_name, batch_size, max_number,
                        sampling_function, selection_function=None,
                        wld_catalog=None, meas_params=None):
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
        if meas_params is None:
            print("scarlet_resid_params")
            meas_params = Scarlet_resid_params()
        meas_generator = btk.measure.generate(
            meas_params, draw_blend_generator, param)
        return meas_generator


class Resid_btk_model(btk.compute_metrics.Metrics_params):
    def __init__(self,  model_name, model_path, output_dir,
                 training=False, new_model_name=None, images_per_gpu=1,
                 validation_for_training=False, *args, **kwargs):
        super(Resid_btk_model, self).__init__(*args, **kwargs)
        self.training = training
        self.model_path = model_path
        self.output_dir = output_dir
        self.validation_for_training = validation_for_training
        file_name = "train" + model_name
        train = __import__(file_name)

        class InferenceConfig(train.InputConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = images_per_gpu
            STEPS_PER_EPOCH = 500  # 200
            VALIDATION_STEPS = 20
            RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2])
            BBOX_STD_DEV = np.array([0.1, 0.1, 0.2])
            if new_model_name:
                NAME = new_model_name

        self.config = InferenceConfig()
        self.config.display()

    def make_resid_model(self, catalog_name, count=256,
                         sampling_function=None, max_number=2,
                         augmentation=False, norm_val=None,
                         selection_function=None, wld_catalog=None,
                         meas_params=None):
        """Creates dataset and loads model"""
        # If no user input sampling function then set default function
        if not sampling_function:
            sampling_function = resid_general_sampling_function
        self.meas_generator = make_meas_generator(catalog_name,
                                                  self.config.BATCH_SIZE,
                                                  max_number,
                                                  sampling_function,
                                                  selection_function,
                                                  wld_catalog,
                                                  meas_params)
        self.dataset = ResidDataset(self.meas_generator, norm_val=norm_val,
                                    augmentation=augmentation)
        self.dataset.load_data(count=count)
        self.dataset.prepare()
        if augmentation:
            self.config.BATCH_SIZE *= 4
            self.config.IMAGES_PER_GPU *= 4
        if self.training:
            self.model = model_btk.MaskRCNN(mode="training",
                                            config=self.config,
                                            model_dir=self.output_dir)
            if self.validation_for_training:
                val_meas_generator = make_meas_generator(catalog_name,
                                                         self.config.BATCH_SIZE,
                                                         max_number,
                                                         sampling_function,
                                                         selection_function,
                                                         wld_catalog,
                                                         meas_params)
                self.dataset_val = ResidDataset(val_meas_generator,
                                                norm_val=norm_val)
                self.dataset_val.load_data(count=count)
                self.dataset_val.prepare()
        else:
            self.model = model_btk.MaskRCNN(mode="inference",
                                            config=self.config,
                                            model_dir=self.output_dir)
        if self.model_path:
            print("Loading weights from ", self.model_path)
            self.model.load_weights(self.model_path, by_name=True)

    def get_detections(self, index):
        """
        Returns model detected centers and true center for data entry index.
        Args:
            index: Index of dataset to perform detection on.
        Returns:
            x and y coordinates of iteratively detected centers, centers
            detected initially and true centers.
        Useful for evaluating model detection performance."""
        image, gt_bbox, gt_class_id = self.dataset.load_input()
        true_centers = self.dataset.true_cent
        detected_centers = self.dataset.det_cent
        results1 = self.model.detect(image, verbose=0)
        iter_detected_centers = []
        for i, r1 in enumerate(results1):
            iter_detected_centers.append(resid_merge_centers(
                detected_centers[i], r1['rois'], center_shift=0))
        return iter_detected_centers, detected_centers, true_centers


def stack_resid_merge_centers(det_cent, resid_cent,
                              distance_upper_bound=1):
    """Combines centers detected by stack on image and in iterative step.
    """
    # remove duplicates
    if len(resid_cent) == 0:
        return det_cent
    unique_det_cent = np.unique(det_cent, axis=0)
    if len(unique_det_cent) == 0:
        return resid_cent
    z_tree = spatial.KDTree(unique_det_cent)
    resid_cent = resid_cent.reshape(-1, 2)
    match = z_tree.query(resid_cent,
                         distance_upper_bound=distance_upper_bound)
    trim = np.setdiff1d(range(len(unique_det_cent)), match[1])
    trim_det_cent = [unique_det_cent[i] for i in trim]
    if len(trim_det_cent) == 0:
        return resid_cent
    iter_det = np.vstack([trim_det_cent, resid_cent])
    return iter_det


class Stack_iter_btk_param(btk.compute_metrics.Metrics_params):
    def __init__(self, catalog_name, batch_size=1, max_number=2,
                 sampling_function=None, selection_function=None,
                 wld_catalog=None, meas_params=None, detect_coadd=False,
                 *args, **kwargs):
        super(Stack_iter_btk_param, self).__init__(*args, **kwargs)
        if not sampling_function:
            print("resid_sampling")
            sampling_function = resid_general_sampling_function
        self.meas_generator = make_meas_generator(
            catalog_name, batch_size, max_number, sampling_function,
            selection_function, wld_catalog, meas_params)
        self.detect_coadd = detect_coadd

    def get_resid_iter_detections(self, index):
        """
        Returns model detected centers and true center for data entry index.
        Args:
            index: Index of dataset to perform detection on.
        Returns:
            x and y coordinates of iteratively detected centers, centers
            detected initially and true centers.
        Useful for evaluating model detection performance."""
        image, gt_bbox, gt_class_id = self.dataset.load_input()
        true_centers = self.dataset.true_cent
        detected_centers = self.dataset.det_cent
        results1 = self.model.detect(image, verbose=0)
        iter_detected_centers = []
        for i, r1 in enumerate(results1):
            iter_detected_centers.append(resid_merge_centers(
                detected_centers[i], r1['rois'], center_shift=0))
        return iter_detected_centers, detected_centers, true_centers

    def get_iter_centers(self):
        """Performs stack detection on residual image and returns detected
        centroids."""
        output, deb, cat = next(self.meas_generator)
        self.output = output
        self.deblend_output = deb
        self.obs_cond = output['obs_condition']
        resid_centers = []
        self.det_cent, self.true_cent = [], []
        for i in range(len(output['blend_list'])):
            blend_image = output['blend_images'][i]
            blend_list = output['blend_list'][i]
            model_image = deb[i][0]
            detected_centers = deb[i][1]
            self.det_cent.append(detected_centers)
            self.true_cent.append(
                np.stack([blend_list['dx'], blend_list['dy']]).T)
            resid_images = blend_image - model_image
            resid_cat = get_stack_catalog(resid_images[:, :, 3],
                                          output['obs_condition'][i][3],
                                          detect_coadd=self.detect_coadd)
            resid_centers.append(get_stack_centers(resid_cat))
        return resid_centers

    def get_stck_iter_detections(self, index):
        """Returns model detected centers and true center for data entry index.
        Args:
            index: Index of dataset to perform detection on.
        Returns:
            x and y coordinates of iteratively detected centers, centers
            detected initially and true centers.
        Useful for evaluating model detection performance."""
        results1 = self.get_iter_centers()
        iter_detected_centers = []
        for i, r1 in enumerate(results1):
            iter_detected_centers.append(stack_resid_merge_centers(
                self.det_cent[i], r1))
        return iter_detected_centers
