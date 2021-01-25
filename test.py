import argparse
import os
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from PIL import Image


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        num_workers=2,
        process_dataset=False,
        crop_size=256
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    try:
        os.mkdir('Results')
    except:
        pass
    PATH = f'Results/{datetime.now()}'
    os.mkdir(PATH)
    img_counter = 0
    with torch.no_grad():
        for i, (clean, noisy) in enumerate(tqdm(data_loader)):
            Batch_Size = clean.shape[0]
            clean, noisy = clean.to(device), noisy.to(device)
            latent_space = torch.normal(0.0, 0.0, clean.shape).to(device)
            output = model(latent_space, noisy, reverse=True)[0]
            out = output.cpu().permute([0, 2, 3, 1]).numpy()

            clean_out = clean.cpu().permute([0, 2, 3, 1]).numpy()
            for j in range(Batch_Size):
                out_img = out[j]
                clean_img = clean_out[j]
                save_img = Image.fromarray(concat_images(out_img, clean_img))
                save_img.save(f'{PATH}/{img_counter}.png')
                img_counter += 1

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, clean)
            batch_size = clean.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, clean) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
