
import argparse


def get_local_expl_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Explain a prediction')
    parser.add_argument('--trained_model',
                        type=str,
                        default='./run_prototypes/cub200/resnet50-nat/identity/second-run/checkpoints/best_test_model',
                        # default='./run_prototypes/chordal-9-epoch-0.1-nooptim/checkpoints/best_test_model/',
                        help='Directory to trained trained_model')
    parser.add_argument('--dataset',
                        type=str,
                        # default='ETH-80',
                        default='CUB-200-2011',
                        help='Data set on which the trained_model was trained')
    parser.add_argument('--sample_dir',
                        type=str,
                        default='./samples',#Laysan_Albatross_0001_545.jpg',
                        # default='./samples/Black_Footed_Albatross_0003_796136.jpg',
                        # default='./samples/Sooty_Albatross_0003_1078.jpg',
                        # default='./samples/Red_Faced_Cormorant_0010_23421.jpg',
                        help='Directory to image to be explained, or to a folder containing multiple test images')
    parser.add_argument('--results_dir',
                        type=str,
                        default='./explanations',
                        help='Directory where explanations will be saved')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Resize images to this size')
    # parser.add_argument('--dir_for_saving_images',
    #                     type=str,
    #                     default='upsampling_results',
    #                     help='Directoy for saving the prototypes, patches and heatmaps')
    parser.add_argument('--upsample_threshold',
                        type=float,
                        default=0.98,
                        help='Threshold (between 0 and 1) for visualizing the nearest patch of an image after upsampling. The higher this threshold, the larger the patches.')
    args = parser.parse_args()
    return args
