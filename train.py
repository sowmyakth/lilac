# Script to train data
#  #use utils.generate_anchors_no_spill
import os
import btk_utils
import dill


# Root directory of the project
ROOT_DIR = '/home/users/sowmyak/lilac'
# Directory to save logs and trained model
MODEL_DIR = '/scratch/users/sowmyak/lilac/logs'
# path to images
DATA_PATH = '/scratch/users/sowmyak/data'

# from mrcnn.model import log


def main(Args):
    """Train group blends with btk"""
    norm = [1.9844158727667542, 413.83759806375525,
            51.2789974336363, 1038.4760551905683]
    count = 2000  # 40000
    catalog_name = os.path.join(DATA_PATH, 'OneDegSq.fits')
    new_model_name = "lilac_" + Args.model_name + "_btk_2gal"
    # Define parameters for mrcnn model with btk here
    resid_model = btk_utils.lilac_btk_model(
        Args.model_name, Args.model_path, MODEL_DIR, training=True,
        new_model_name=new_model_name, images_per_gpu=4)
    # Load parametrs for dataset and load model
    resid_model.make_model(catalog_name, count=count,
                                 max_number=2, augmentation=True,
                                 norm_val=norm)
    learning_rate = resid_model.config.LEARNING_RATE/10.
    history1 = resid_model.model.train(resid_model.dataset,
                                       resid_model.dataset_val,
                                       learning_rate=learning_rate,
                                       epochs=40,
                                       layers='all')
    name = new_model_name + '_run1_loss'
    with open(name + ".dill", 'wb') as handle:
        dill.dump(history1.history, handle)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='basic',
                        help="Name of model to evaluate")
    parser.add_argument('--model_path', type=str, default=None,
                        help="Saved weights of model")
    args = parser.parse_args()
    main(args)
