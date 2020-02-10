import argparse
import configparser
import sys

from src.utils.utils import *


def process_params():
    # Search for config path
    args = load_args()
    if not args.config_path:
        # -----Get arguments from command line-----
        params = process_command_args(args)
    else:
        # -----Get arguments from config file-----
        params = process_config_file(args.config_path)

    return params


def prepare_environment(params):
    # Create results folder
    params["experiment_folder"] = create_result_directories(params["results_path"])
    # Create temporal folders if required
    if params["track_path"] is not None:
        params["track_folder"] = create_tracking_directories(params["track_path"])
    # Set seed
    set_seed(params["seed"])


def set_seed(seed=None):
    if not seed:
        seed = random.randint(1, 10000)

    random.seed(seed)
    torch.manual_seed(seed)


def load_data(data_path, img_size, img_channels, batch_size=60, norm=0.5, shuffle=False, center_crop=False,
              dl_workers=1):
    # Set transform
    transform = set_transform(img_size=img_size, img_channels=img_channels, center_crop=center_crop, norm=norm)
    # Load dataset
    dataset = dataset_utils.ImageFolder(root=data_path, transform=transform)
    # Create data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=dl_workers)

    return data_loader


def set_transform(img_size, img_channels, center_crop=False, norm=0.5):
    if center_crop:
        transformations = [transforms.Resize(img_size),
                           transforms.CenterCrop(img_size),
                           transforms.ToTensor()]
    else:
        transformations = [transforms.Resize(img_size),
                           transforms.ToTensor()]

    if img_channels == 3:
        transformations.append(transforms.Normalize((norm, norm, norm), (norm, norm, norm)))
    elif img_channels == 2:
        transformations.append(transforms.Normalize((norm, norm), (norm, norm)))
    elif img_channels == 1:
        transformations.append(transforms.Normalize((norm,), (norm,)))
    else:
        raise ValueError("Image depth not supported")

    transform = transforms.Compose(transformations)

    return transform


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', help="Path to configuration file '.ini.")
    parser.add_argument('-dsp', '--data_set_path', help="Path to the dataset.",
                        required='--config_path' not in sys.argv)
    parser.add_argument('-lm', '--load_models',
                        help="Path to the stored models /path/to/models (/generator and /discriminator).")
    parser.add_argument('-gt', '--gan_type', help="Gan type ('DCGAN', RasGAN').", default="DCGAN")
    parser.add_argument('-gen_n', '--gen_model_name', help="Generator model name, ex DCGAN_generator",
                        default="DCGAN_generator")
    parser.add_argument('-disc_n', '--disc_model_name', help="Generator model name, ex DCGAN_generator",
                        default="DCGAN_generator")
    parser.add_argument('-opt', '--optimizer', help="Sets the optimizer (ADAM)", default="ADAM")
    parser.add_argument('-lr', '--learning_rate', help="Learning rate", default=0.0002)
    parser.add_argument('-bs', '--batch_size', help="Data batch size.", default=60)
    parser.add_argument('-beta', help="Beta1 for the optimizer (0.5 default).", default=0.5)
    parser.add_argument('-nep', '--num_epochs', help="Number of training epochs.", default=2)
    parser.add_argument('-vs', '--vector_size', help="Length of the vector size.", default=100)
    parser.add_argument('-img_s', '--image_size', help="Output data image size",
                        required='--config_path' not in sys.argv)
    parser.add_argument('-img_c', '--image_channels', help="Output data image channels",
                        required='--config_path' not in sys.argv)
    parser.add_argument('-n_gpu', '--number_gpu', help="The number of GPU utilized, 0 for not using CUDA", default=0)
    parser.add_argument('-dl_w', '--data_loader_workers', help="Number of workers for the data loader", default=1)
    parser.add_argument('-sh_d', '--shuffle_data', help="Shuffle data set", default=1)
    parser.add_argument('-ccr', '--center_crop', action='store_true', help="Center crop data set images", default=True)
    parser.add_argument('-norm', '--image_normalization', help="Image_normalization", default=0.5)
    parser.add_argument('-s', '--seed', help="Seed used during initialization", default=1)
    parser.add_argument('-rp', '--results_path', help="Path for storing the results, it will generate the folders"
                                                      "/generated_images /models /report /plot",
                        required='--config_path' not in sys.argv)
    parser.add_argument('-tr_p', '--track_path',
                        help="Path for storing the temporary images and the models")
    parser.add_argument('-tr_fr', '--track_freq', help="Stores the temporary models and images every X "
                                                       "iterations", default=20)
    parser.add_argument('-tr_ni', '--track_num_images',
                        help="Number of images to generate at each tracking operation.", default=1)
    parser.add_argument('-fid', '--fid_score', action='store_true', help="Enable FID score calculation.")
    parser.add_argument('-fid_fr', '--fid_score_fr', help="FID score calculation frequency (Epochs).", default=5)
    parser.add_argument('-fid_bs', '--fid_score_bs', help="FID score batch size.", default=2)
    parser.add_argument('-fid_opt', '--fid_optimization', help="FID score batch size.", default=0)
    return parser.parse_args()


def process_command_args(args):
    params = {}
    # Environment params
    params["n_gpu"] = args.number_gpu
    params["seed"] = args.seed
    params["load_model"] = args.load_models
    # Data params
    params["dl_workers"] = args.data_loader_workers
    params["data_path"] = args.data_set_path
    params["shuffle_data"] = bool(args.shuffle_data)
    params["norm"] = float(args.image_normalization)
    params["center_crop"] = bool(args.center_crop)
    params["image_size"] = int(args.image_size)
    params["image_channels"] = int(args.image_channels)
    params["results_path"] = args.results_path
    # Training params
    params["optimizer"] = args.optimizer
    params["learning_rate"] = float(args.learning_rate)
    params["batch_size"] = int(args.batch_size)
    params["num_epochs"] = int(args.num_epochs)
    params["vector_size"] = int(args.vector_size)
    params["beta"] = float(args.beta)
    params["gan_type"] = args.gan_type
    params["gen_model"] = args.gen_model_name
    params["disc_model"] = args.disc_model_name
    # Track params
    params["track_path"] = args.track_path
    params["track_fr"] = int(args.track_freq)
    params["track_num_images"] = int(args.track_num_images)
    # FID Score
    params["fid"] = bool(args.fid_score)
    params['fid_opt'] = bool(args.fid_optimization)
    params["fid_fr"] = int(args.fid_score_fr)
    params["fid_bs"] = int(args.fid_score_bs)

    return params


def process_config_file(file_path):
    conf = load_config_file(file_path)
    params = {}
    # Environment params
    params["n_gpu"] = int(conf['environment']['n_gpu'])
    params["seed"] = int(conf['environment']['seed'])
    # Data params
    params["shuffle_data"] = conf['data']['shuffle_data'] = '1'
    params["dl_workers"] = int(conf['data']['dl_workers'])
    params["norm"] = float(conf['data']['norm'])
    params["center_crop"] = conf['data']['center_crop'] == '1'
    params["data_path"] = conf['data']['data_path']
    params["image_size"] = int(conf['data']['img_size'])
    params["image_channels"] = int(conf['data']['img_channels'])
    params["results_path"] = conf['data']['results_path']
    # Training params
    params["load_model"] = conf['training']['load_model']
    params["optimizer"] = conf['training']['optimizer']
    params["learning_rate"] = float(conf['training']['learning_rate'])
    params["batch_size"] = int(conf['training']['batch_size'])
    params["num_epochs"] = int(conf['training']['num_epochs'])
    params["vector_size"] = int(conf['training']['vector_size'])
    params["beta"] = float(conf['training']['beta'])
    params["gan_type"] = conf['training']['gan_type']
    params["gen_model"] = conf['training']['gen_model']
    params["disc_model"] = conf['training']['disc_model']
    # Track params
    params["track_path"] = conf['tracking']['track_path']
    params["track_fr"] = int(conf['tracking']['track_fr'])
    params["track_num_images"] = int(conf['tracking']['track_num_images'])
    # FID Score
    params["fid"] = conf['evaluation']['fid_score'] == '1'
    params['fid_opt'] = conf['evaluation']['fid_opt'] == '1'
    params["fid_fr"] = int(conf['evaluation']['fid_freq'])
    params["fid_bs"] = int(conf['evaluation']['fid_batch_size'])

    return params


def load_config_file(file_path):
    try:
        conf = configparser.ConfigParser()
        conf.read(file_path)
        return conf
    except configparser.Error as e:
        raise Exception(f"Error while parsing configuration file {file_path}, error: {e}")
