import sys
import os
import torch
import torch.nn as nn
import argparse
import cv2
import numpy as np
from model_prog import FDGRU, MLP, run_eval_decoder, progToData, Sampler, run_train_decoder
#import ni_batch_voxel_model_prog as batch_model_prog
from torch.utils.data import DataLoader
from ShapeAssembly import ShapeAssembly, hier_execute, make_hier_prog
import utils
import pdb
import random
import glob
from skimage.transform import resize
import skimage.io
from skimage import io
from tqdm import tqdm
from losses import FScore, ProgLoss
import metrics
import trimesh
import ast
import matplotlib.image as mpimg
#import resnet
#import vgg

CATEGORY = 'chair'
VIEWS = 'four_view_resized'

PROGRAMS_DIR = f'data/{CATEGORY}'
VAL_INDICES_PATH = f'data_splits/{CATEGORY}/val.txt'
TRAIN_INDICES_PATH = f'data_splits/{CATEGORY}/train.txt'
OUTPUT_PROGRAMS_DIR = f'mask_and_depth_encoder_output/{CATEGORY}/programs'
OUTPUT_MODELS_DIR = f'mask_and_depth_encoder_output/{CATEGORY}/models'
OUTPUT_OBJS_DIR = f'mask_and_depth_encoder_output/{CATEGORY}/objs'
IMAGES_DIR = f'depth_and_mask_data/{VIEWS}'

device = torch.device('cuda')

BATCH_SIZE = 1
NUM_EPOCHS = 10000
NUM_EVAL = 200
LEARNING_RATE = .0001
LATENT_SPACE_DIMENSION = 256

NUM_FSCORE_SAMPLES = 10000

IMAGE_SHAPE = (150, 200)

shape_assembly = ShapeAssembly()

def _col(samples):
    return samples

def _encodings_collate(samples):
    images = [s[0] for s in samples]
    encodings = [s[1] for s in samples]
    index = [s[2] for s in samples]
    progs = [s[3] for s in samples]
    return (images, encodings, index, progs)

def get_indices(index_file):
    indices = set()
    with open(index_file) as f:
        for line in f:
            indices.add(line.strip())
    return indices

def get_data(images_dir, indices, args):
    samples = []
    print('Getting data...')
    #indices = [35532, 39813]

    for index in tqdm(indices, position=0, leave=True):
        gt_program_path = glob.glob(f'{PROGRAMS_DIR}/{index}.txt')[0]
        for i in range(args.images_per_program):
            mask_path = f'{images_dir}/{index}_maskview{i}.png'
            depth_path = f'{images_dir}/{index}_depthview{i}.png'
            nocs_path = f'{images_dir}/{index}_nocsview{i}.png'
            gt_prog_tensorized = progToData(utils.loadHPFromFile(gt_program_path))
            mask_image = skimage.io.imread(mask_path, as_gray=True)
            depth_image = skimage.io.imread(depth_path, as_gray=True)
            if args.use_nocs:
                nocs_image = skimage.io.imread(nocs_path)
            if (mask_image is not None) and (depth_image is not None):
                mask_image = torch.tensor(mask_image).float() / 255.0  # need to divide by 255 because skimage.io.imread outputs integer array by default
                depth_image = torch.tensor(depth_image).float() / 255.0
                if args.use_nocs:
                    nocs_image = torch.tensor(nocs_image).float() / 255.0
                    image = torch.cat((mask_image.unsqueeze(2), depth_image.unsqueeze(2), nocs_image), dim=2).permute(2, 0, 1)
                else:
                    image = torch.stack([mask_image, depth_image], dim=2).permute(2, 0, 1)
                encoding = torch.squeeze(torch.load(f'{args.encodings_dir}/{index}.enc')).float()
                samples.append((image, encoding, index, gt_prog_tensorized))
                if i >= args.images_per_program - 1:
                    break

        #if len(samples) >= 15:
            #break

    return samples

def train_depth_and_mask_encoder(train_data_list, val_data_list, encoder, decoder, optimizer, args):
    for i in range(NUM_EPOCHS):
        encoder.train()
        decoder.train()
        total_error = 0
        train_data = DataLoader(train_data_list, batch_size=BATCH_SIZE, shuffle=True, collate_fn=_encodings_collate)
        val_data = DataLoader(val_data_list, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_encodings_collate)
        print(f'*********** EPOCH {i} ***********')
        for (image_batch, target_encoding, indices, gt_prog_tensorized) in tqdm(train_data, position=0, leave=True):
            predicted_encoding = encoder.forward(torch.stack(image_batch).to(device))
            target_encoding = torch.stack(target_encoding).to(device)
            loss = torch.nn.functional.mse_loss(input=predicted_encoding, target=target_encoding, reduction='none')
            back_loss = loss.sum(dim=1).mean()
            optimizer.zero_grad()
            back_loss.backward()
            optimizer.step()
            total_error += float(loss.sum())

        val_loss = 0
        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            for (image_batch, target_encoding, indices, gt_prog_tensorized) in tqdm(val_data, position=0, leave=True):
                predicted_encoding = encoder.forward(torch.stack(image_batch).to(device))
                target_encoding = torch.stack(target_encoding).to(device)
                loss = torch.nn.functional.mse_loss(input=predicted_encoding, target=target_encoding, reduction='none')
                val_loss += float(loss.sum())

        print(f'**************EPOCH {i}******************')
        print(f'TRAINING LOSS: {total_error / len(train_data.dataset)}')
        print(f'VALIDATION LOSS: {val_loss / len(val_data.dataset)}')

        with open(f"logs/{args.exp_name}_training_graph.txt", "a") as file:
            file.write(f'{total_error / len(train_data.dataset)}, {val_loss / len(val_data.dataset)}\n')

        # every 100 epochs, save a model
        if i % 50 == 0 and i != 0:
            pass
            #torch.save(encoder, f'{OUTPUT_MODELS_DIR}/model_{i}.pt')
        if i % 10 == 0:
            train_data = DataLoader(train_data_list[:NUM_EVAL], batch_size=1, shuffle=False, collate_fn=_col)
            val_data = DataLoader(val_data_list[:NUM_EVAL], batch_size=1, shuffle=False, collate_fn=_col)
            eval_model(train_data, encoder, decoder, 'training_data', args)
            eval_model(val_data, encoder, decoder, 'validation_data', args)

def makeOBJ(index, model_name, encoder, decoder):
    with torch.no_grad():
        mask_image_path = f'{IMAGES_DIR}/{index}_maskview0.png' # fix these for different view configurations
        depth_image_path = f'{IMAGES_DIR}/{index}_depthview0.png'
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
        if (mask_image is not None) and (depth_image is not None):
            mask_image = torch.tensor(resize(mask_image, IMAGE_SHAPE)).float().to(device)
            depth_image = torch.tensor(resize(depth_image, IMAGE_SHAPE)).float().to(device)
            image = torch.stack([mask_image, depth_image], dim=2).permute(2, 0, 1)
            predicted_encoding = encoder.forward(torch.unsqueeze(image, 0)).unsqueeze(0)
            prog, _ = run_eval_decoder(predicted_encoding, decoder, False)
            verts, faces = hier_execute(prog)
            utils.writeObj(verts, faces, f"{OUTPUT_OBJS_DIR}/{index}_{model_name}.obj")
            utils.writeHierProg(prog, f"{OUTPUT_PROGRAMS_DIR}/{index}_{model_name}.txt")
            print('made obj')

def train_end_to_end(encoder, decoder, train_data, val_data, train_data_list, val_data_list, args):
    dec_opt = torch.optim.Adam(
        decoder.parameters(),
        lr=.0001,
        eps=1e-6
    )

    enc_opt = torch.optim.Adam(
        encoder.parameters(),
        lr=.0001,
        eps=1e-6
    )

    enc_sch = torch.optim.lr_scheduler.StepLR(
        enc_opt,
        step_size=500,
        gamma=.9
    )

    dec_sch = torch.optim.lr_scheduler.StepLR(
        dec_opt,
        step_size=500,
        gamma=.9
    )

    for i in range(NUM_EPOCHS):
        training_loss = run_end_to_end(encoder, decoder, train_data, enc_opt, dec_opt, True, args)
        #dec_sch.step()
        #enc_sch.step()
        val_loss = run_end_to_end(encoder, decoder, val_data, None, None, False, args)
        print(f'**************EPOCH {i}******************')
        print(f'TRAINING LOSS: {training_loss / len(train_data.dataset)}')
        print(f'VALIDATION LOSS: {val_loss / len(val_data.dataset)}')
        with open(f"logs/{args.exp_name}_training_graph.txt", "a") as file:
            file.write(f'{training_loss / len(train_data.dataset)}, {val_loss / len(val_data.dataset)}\n')
        if i % 10 == 0:
            train_eval_data = DataLoader(train_data_list[:NUM_EVAL], batch_size=1, shuffle=False, collate_fn=_col)
            val_eval_data = DataLoader(val_data_list[:NUM_EVAL], batch_size=1, shuffle=False, collate_fn=_col)
            eval_model(train_eval_data, encoder, decoder, 'training_data', args)
            eval_model(val_eval_data, encoder, decoder, 'validation_data', args) # **************************************
        if i % 50 == 0 and i > 0:
            torch.save(encoder, f'{OUTPUT_MODELS_DIR}/encoder_{i}.pt')
            torch.save(decoder, f'{OUTPUT_MODELS_DIR}/decoder_{i}.pt')


def run_end_to_end(encoder, decoder, data, enc_opt, dec_opt, is_training, args):
    epoch_loss = 0
    if is_training:
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()
    for batch in tqdm(data, position=0, leave=True):
        epoch_loss += run_one_batch(encoder, decoder, batch, enc_opt, dec_opt, args)
    
    return epoch_loss

def run_one_batch(encoder, decoder, batch, enc_opt, dec_opt, args):
    loss_config = getLossConfig()
    batch_result = {}
    for (image_batch, target_encoding, indices, gt_prog_tensorized) in tuple(batch):
        encoding = encoder.forward(image_batch.unsqueeze(0).to(device)).unsqueeze(0)
        shape_result = run_train_decoder(
            gt_prog_tensorized, encoding, decoder
        )

        for res in shape_result:
            if res not in batch_result:
                batch_result[res] = shape_result[res]
            else:
                batch_result[res] += shape_result[res]

    loss = 0.
    if len(batch_result) > 0:
        for key in loss_config:
            batch_result[key] *= loss_config[key]
            if torch.is_tensor(batch_result[key]):
                loss += batch_result[key]

    float_loss = float(loss)
    if torch.is_tensor(loss) and enc_opt is not None and dec_opt is not None:
        dec_opt.zero_grad()
        enc_opt.zero_grad()
        loss.backward()
        if args.train_decoder:
            dec_opt.step()
        enc_opt.step()

    return float_loss

def getLossConfig():
    loss_config = {
        'cmd': 1.0,
        'cub_prm': 50.0,
        #'cub_prm': 1.0,

        'xyz_prm': 50.0,
        'uv_prm': 50.0,
        'sym_prm': 50.0,
        #'xyz_prm': 1.0,
        #'uv_prm': 1.0,
        #'sym_prm': 1.0,

        'cub': 1.0,
        'sym_cub': 1.0,
        'sq_cub': 1.0,

        'leaf': 1.0,
        'bb': 50.0,
        #'bb': 1.0,

        'axis': 1.0,
        'face': 1.0,
        'align': 1.0
    }

    return loss_config


def eval_model(data, encoder, decoder, dataset_name, args):
    decoder.eval()
    encoder.eval()

    eval_results = []

    named_results = {
        'count': 0.,
        'miss_hier_prog': 0.
    }
    recon_sets = []
    count = 0
    with torch.no_grad():
        for batch in tqdm(data, position=0, leave=True):
            for (image_batch, target_encoding, indices, gt_prog_tensorized) in batch:

                named_results[f'count'] += 1.
                predicted_encoding = encoder.forward(image_batch.unsqueeze(0).to(device)).unsqueeze(0)
                prog, shape_result = run_eval_decoder(
                    predicted_encoding, decoder, False, gt_prog_tensorized
                )

                for key in shape_result:
                    nkey = f'{key}'
                    if nkey not in named_results:
                        named_results[nkey] = shape_result[key]
                    else:
                        named_results[nkey] += shape_result[key]

                if prog is None:
                    named_results[f'miss_hier_prog'] += 1.
                    continue

                recon_sets.append((prog, gt_prog_tensorized, indices))

                if args.make_objs_in_eval:
                    if (len(prog['prog']) > 0):
                        try:
                            print(f'creating an obj for {indices}')
                            verts, faces = hier_execute(prog)
                            utils.writeObj(verts, faces, f"{OUTPUT_OBJS_DIR}/{int(indices)}{dataset_name}.obj")
                            utils.writeHierProg(prog, f"{OUTPUT_PROGRAMS_DIR}/{int(indices)}{dataset_name}.txt")
                            verts, faces = hier_execute(gt_prog_tensorized)
                            utils.writeObj(verts, faces, f"{OUTPUT_OBJS_DIR}/{int(indices)}{dataset_name}_gt.obj")
                            utils.writeHierProg(prog, f"{OUTPUT_PROGRAMS_DIR}/{int(indices)}{dataset_name}_gt.txt")
                        except Exception as e:
                            print(f'failed to make obj because of: {e}')

            count += 1

    # For reconstruction, get metric performance
    recon_results, recon_misses = metrics.recon_metrics(
        recon_sets, 'eval_depth_mask_encoder', f'depth_mask_encoder_{CATEGORY}_exp', dataset_name, 0, True
    )

    for key in recon_results:
        named_results[key] = recon_results[key]

    named_results[f'miss_hier_prog'] += recon_misses

    named_results[f'prog_creation_perc'] = (named_results[f'count'] - named_results[f'miss_hier_prog']) / named_results[f'count']

    eval_results.append((dataset_name, named_results))

    print_eval_results(eval_results, args)


def print_eval_results(eval_results, args):
    for name, named_results in eval_results:
        if named_results['nc'] > 0:
            named_results['cub_prm'] /= named_results['nc']

        if named_results['na'] > 0:
            named_results['xyz_prm'] /= named_results['na'] 
            named_results['cubc'] /= named_results['na']

        if named_results['count'] > 0:
            named_results['bb'] /= named_results['count']

        if named_results['nl'] > 0:
            named_results['cmdc'] /= named_results['nl']

        if named_results['ns'] > 0:
            named_results['sym_cubc'] /= named_results['ns']
            named_results['axisc'] /= named_results['ns']
            named_results['sym_prm'] /= named_results['ns'] # symmetry parameter loss needs to be divided by the number of symmetry lines in the programs

        if named_results['np'] > 0:
            named_results['corr_line_num'] /= named_results['np']
            named_results['bad_leaf'] /= named_results['np']

        if named_results['nsq'] > 0:
            named_results['uv_prm'] /= named_results['nsq']
            named_results['sq_cubc'] /= named_results['nsq']
            named_results['facec'] /= named_results['nsq']

        if named_results['nap'] > 0:
            named_results['palignc'] /= named_results['nap']

        if named_results['nan'] > 0:
            named_results['nalignc'] /= named_results['nan']

        named_results.pop('nc')
        named_results.pop('nan')
        named_results.pop('nap')
        named_results.pop('na')
        named_results.pop('ns')
        named_results.pop('nsq')
        named_results.pop('nl')
        named_results.pop('count')
        named_results.pop('np')
        named_results.pop('cub')
        named_results.pop('sym_cub')
        named_results.pop('axis')
        named_results.pop('cmd')
        named_results.pop('miss_hier_prog')


        # cub_prm (Cube Parameter loss): L1 loss on length, width, and height of cuboid statement
        # xyz_prm (XYZ Parameter loss): L1 loss on x,y,z coordinates of attach statement
        # uv_prm (UV Parameter loss): L1 loss on u and v coordinates of squeeze statements
        # sym_prm (Symmetry parameter loss): L1 loss on continuous symmetry parameters
        # bbox loss (Bounding box loss): L1 loss on bounding box dimensions
        # cmdc (Command percent correct): percent of correct predictions of the line type
        # cubc (Cube indices correct in attach statements)
        # sq_cubc (Cube indices correct in squeeze statements)
        # facec (Cube faces correct in squeeze statements)
        # palignc: percent of correct predictions of a face being aligned
        # nalignc: percent of correct predictions of a face being not aligned (this could appear as 0 even if everything is correct)
        # sym_cubc: (Cube indices correct in symmetry statements)
        # axisc: (Axis correct in symmetry statement)
        # corr_line_num: percent of time program had correct number of lines
        # bad_leaf: percent of time a leaf was predicted incorrectly
        with open(f"logs/{args.exp_name}_{name}_eval_results.txt", "a") as file:
            file.write(f'{named_results["fscores"]},{named_results["iou_shape"]},{named_results["cmdc"]},{named_results["cubc"]}\n')

        print(f''' Evaluation on {name} set:
              Eval {name} F-score = {named_results['fscores']}
              Eval {name} IoU = {named_results['iou_shape']}
              Eval {name} PD = {named_results['param_dist_parts']}
              Eval {name} Prog Creation Perc: {named_results['prog_creation_perc']}
              Eval {name} Cub Prm Loss = {named_results['cub_prm']}  
              Eval {name} XYZ Prm Loss = {named_results['xyz_prm']}
              Eval {name} UV Prm Loss = {named_results['uv_prm']}
              Eval {name} Sym Prm Loss = {named_results['sym_prm']}
              Eval {name} BBox Loss = {named_results['bb']}
              Eval {name} Cmd Corr % {named_results['cmdc']}
              Eval {name} Cub Corr % {named_results['cubc']}
              Eval {name} Squeeze Cub Corr % {named_results['sq_cubc']}
              Eval {name} Face Corr % {named_results['facec']}
              Eval {name} Pos Align Corr % {named_results['palignc']}
              Eval {name} Neg Align Corr % {named_results['nalignc']}
              Eval {name} Sym Cub Corr % {named_results['sym_cubc']}
              Eval {name} Sym Axis Corr % {named_results['axisc']}
              Eval {name} Corr Line # % {named_results['corr_line_num']}
              Eval {name} Bad Leaf % {named_results['bad_leaf']}''')

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CNN1(nn.Module):

    def __init__(self, num_classes=1000):
        super(CNN1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 2000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2000, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CNN2(nn.Module):

    def __init__(self, num_classes=1000):
        super(CNN2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.maxpool = nn.AdaptiveMaxPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CNN3(nn.Module):

    def __init__(self, num_classes=1000):
        super(CNN3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(27648, 2000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2000, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CNN4(nn.Module):

    def __init__(self, num_classes=1000):
        super(CNN4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    model = AlexNet(**kwargs)
    return model

def cnn1(pretrained=False, progress=True, **kwargs):
    model = CNN1(**kwargs)
    return model

def cnn2(pretrained=False, progress=True, **kwargs):
    model = CNN2(**kwargs)
    return model

def cnn3(pretrained=False, progress=True, **kwargs):
    model = CNN3(**kwargs)
    return model

def cnn4(pretrained=False, progress=True, **kwargs):
    model = CNN4(**kwargs)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run depth and mask encoder")
    parser.add_argument('action', choices=['train', 'train_end_to_end', 'make_obj', 'eval'], type=str)
    parser.add_argument('--encoder_path', type=str)
    parser.add_argument('--decoder_path', type=str)
    parser.add_argument('--index', type=str)
    parser.add_argument('--images_per_program', default=1, type=int)
    parser.add_argument('--train_decoder', type=str, default="True")
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--make_objs_in_eval', type=str, default="False")
    parser.add_argument('--use_nocs', type=str, default="False")
    parser.add_argument('--encodings_dir', type=str)
    args = parser.parse_args()

    args.use_nocs = ast.literal_eval(args.use_nocs)
    args.train_decoder = ast.literal_eval(args.train_decoder)
    args.make_objs_in_eval = ast.literal_eval(args.make_objs_in_eval)

    if args.action == 'train':
        if args.encoder_path:
            encoder = torch.load(f'{args.encoder_path}').to(device)
            print('loaded pretrained encoder')
        else: 
            encoder = alexnet(num_classes=LATENT_SPACE_DIMENSION).to(device)
            print('created new encoder')
        if args.decoder_path:
            decoder = torch.load(f'{args.decoder_path}').to(device)
            print('using pretrained decoder')
        train_indices = get_indices(TRAIN_INDICES_PATH)
        val_indices = get_indices(VAL_INDICES_PATH)
        train_data_list = get_data(IMAGES_DIR, train_indices, args)
        val_data_list = get_data(IMAGES_DIR, val_indices, args)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
        train_depth_and_mask_encoder(train_data_list, val_data_list, encoder, decoder, optimizer, args)
    elif args.action == 'train_end_to_end':
        if args.encoder_path:
            encoder = torch.load(f'{args.encoder_path}').to(device)
            print('loaded pretrained encoder')
        else:
            encoder = alexnet(num_classes=LATENT_SPACE_DIMENSION).to(device)
            print('created new encoder')
        if args.decoder_path:
            decoder = torch.load(f'{args.decoder_path}').to(device)
            print('loaded pretrained decoder')
        else:
            decoder = FDGRU(LATENT_SPACE_DIMENSION).to(device)
            print('created new decoder')

        train_indices = get_indices(TRAIN_INDICES_PATH)
        val_indices = get_indices(VAL_INDICES_PATH)
        train_data_list = get_data(IMAGES_DIR, train_indices, args)
        val_data_list = get_data(IMAGES_DIR, val_indices, args)
        train_data = DataLoader(train_data_list, batch_size=1, shuffle=True, collate_fn = _col)
        val_data = DataLoader(val_data_list, batch_size=1, shuffle=False, collate_fn = _col)
        train_end_to_end(encoder, decoder, train_data, val_data, train_data_list, val_data_list, args)
    elif args.action == 'make_obj':
        if (args.encoder is not None) and (args.index is not None):
            encoder = torch.load(f'{OUTPUT_MODELS_DIR}/{args.encoder}')
            decoder = torch.load(f'{OUTPUT_MODELS_DIR}/{args.decoder}').to(device)
            makeOBJ(args.index, args.encoder, encoder, decoder)
    elif args.action == 'eval':
        if (args.encoder is not None):
            encoder = torch.load(f'{OUTPUT_MODELS_DIR}/{args.encoder}')
            print('encoder loaded')
            if (args.decoder is not None):
                decoder = torch.load(f'{OUTPUT_MODELS_DIR}/{args.decoder}')
                print('decoder loaded')
            else:
                decoder = torch.load(DECODER_PATH).to(device)
            train_indices = get_indices(TRAIN_INDICES_PATH)
            val_indices = get_indices(VAL_INDICES_PATH)
            train_data = get_data(IMAGES_DIR, train_indices, args)
            val_data = get_data(IMAGES_DIR, val_indices, args)
            train_data = DataLoader(train_data, batch_size=1, shuffle=False, collate_fn = _col)
            val_data = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn = _col)
            with torch.no_grad():
                eval_model(train_data, encoder, decoder, 'training_data', args)
                eval_model(val_data, encoder, decoder, 'validation_data', args)

