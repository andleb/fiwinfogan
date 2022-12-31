import argparse
import os
import re

import itertools as it

import numpy as np
import torch
import torch.optim as optim
from scipy.io.wavfile import read
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fiwinfogan import WaveGANGenerator, WaveGANQNetwork
from utils import get_continuation_fname


class AudioDataSet:
    def __init__(self, datadir, slice_len):
        print("Loading data")
        dir = os.listdir(datadir)
        x = np.zeros((len(dir), 1, slice_len))
        i = 0
        for file in tqdm(dir):
            audio = read(os.path.join(datadir, file))[1]
            if audio.shape[0] < slice_len:
                audio = np.pad(audio, (0, slice_len - audio.shape[0]))
            audio = audio[:slice_len]

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32767
            elif audio.dtype == np.float32:
                pass
            else:
                raise NotImplementedError('Scipy cannot process atypical WAV files.')
            audio /= np.max(np.abs(audio))
            x[i, 0, :] = audio
            i += 1

        self.len = len(x)
        self.audio = torch.from_numpy(np.array(x, dtype=np.float32))

    def __getitem__(self, index):
        return self.audio[index]

    def __len__(self):
        return self.len


def gradient_penalty(D, real, fake, epsilon):
    x_hat = epsilon * real + (1 - epsilon) * fake
    scores, _ = D(x_hat)

    grad = torch.autograd.grad(
        outputs=scores,
        inputs=x_hat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True
    )[0]

    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1)  # norm along each batch
    penalty = ((grad_norm - 1) ** 2).unsqueeze(1)

    return penalty


if __name__ == "__main__":
    # Training Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datadir',
        type=str,
        required=True,
        help='Training Directory'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        required=True,
        help='Log/Results Directory'
    )
    parser.add_argument(
        '--num_categ',
        type=int,
        default=0,
        help='Q-net categories'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=5000,
        help='Epochs'
    )
    parser.add_argument(
        '--slice_len',
        type=int,
        default=16384,
        help='Length of training data'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--cont',
        type=str,
        default="",
        help='''continue: default from the last saved iteration. '''
             '''Provide the epoch number if you wish to resume from a specific point'''
             '''Or set "last" to continue from last available'''
    )

    parser.add_argument(
        '--save_int',
        type=int,
        default=50,
        help='Save interval in epochs'
    )

    args = parser.parse_args()

    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datadir = args.datadir
    logdir = args.logdir
    SLICE_LEN = args.slice_len
    NUM_CATEG = args.num_categ
    NUM_EPOCHS = args.num_epochs
    WAVEGAN_DISC_NUPDATES = 5
    BATCH_SIZE = args.batch_size

    # TODO:
    # * add other weights, losses if doing pure InfoWaveGAN
    # Q penalty portion (the rest are 1)
    LAMBDA_GRAD = 10
    LAMBDA_INFO = 0.5

    LEARNING_RATE = 1e-4
    BETA1 = 0.5
    BETA2 = 0.9

    CONT = args.cont
    SAVE_INT = args.save_int

    # Load data
    dataset = AudioDataSet(datadir, SLICE_LEN)
    dataloader = DataLoader(
        dataset,
        BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    def make_new():
        G = WaveGANGenerator(slice_len=SLICE_LEN, ).to(device).train()
        D = WaveGANQNetwork(slice_len=SLICE_LEN, num_categ=NUM_CATEG).to(device).train()

        # Optimizers
        optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

        optimizer_Q = optim.RMSprop(it.chain(G.parameters(), D.parameters()),
                                    lr=LEARNING_RATE)
        criterion_Q = torch.nn.BCEWithLogitsLoss()

        return G, D, optimizer_G, optimizer_D, optimizer_Q, criterion_Q

    # DEBUG:
    torch.autograd.set_detect_anomaly(True)

    # Load models
    G, D, optimizer_G, optimizer_D, optimizer_Q, criterion_Q = make_new()
    start_epoch = 0
    start_step = 0

    if CONT.lower() != "":
        try:
            print("Loading model from existing checkpoints...")
            fname, start_epoch = get_continuation_fname(CONT, logdir)

            G.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_G.pt")))
            D.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_D.pt")))

            optimizer_G.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Gopt.pt")))
            optimizer_D.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Dopt.pt")))
            optimizer_Q.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Qopt.pt")))

            start_step = int(re.search(r'_step(\d+).*', fname).group(1))
            print(f"Successfully loaded model. Continuing training from epoch {start_epoch},"
                  f" step {start_step}")

        # Don't care why it failed
        except Exception as e:
            print("Could not load from existing checkpoint, initializing new model...")
            print(e)
    else:
        print("Starting a new training")

    # Set Up Writer
    writer = SummaryWriter(logdir)
    step = start_step

    for epoch in range(start_epoch + 1, NUM_EPOCHS):

        print("Epoch {} of {}".format(epoch, NUM_EPOCHS))
        print("-----------------------------------------")
        pbar = tqdm(dataloader)
        real = dataset[:BATCH_SIZE].to(device)

        for i, real in enumerate(pbar):

            if i % WAVEGAN_DISC_NUPDATES == 0:
                # INFORMATIONAL LOSS
                optimizer_Q.zero_grad()

            # DISCRIMINATOR UPDATE
            optimizer_D.zero_grad()

            real = real.to(device)

            _z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_CATEG).uniform_(-1, 1).to(device)
            # TODO: Infogan has uniform on -1, 1 instead of setting it to a bit value
            # while infoGAN has a different loss
            c = torch.FloatTensor(BATCH_SIZE, NUM_CATEG).bernoulli_().to(device)
            z = torch.cat((c, _z), dim=1)

            fake = G(z)

            #NOTE:  create & retain graph
            epsilon = torch.rand(BATCH_SIZE, 1, 1).repeat(1, 1, SLICE_LEN).to(device)
            gradPenalty = gradient_penalty(D,
                                           real,
                                           fake,
                                           epsilon)
            # Save latent code estimates - cHat - for later
            D_fake, _ = D(fake)
            D_real, _ = D(real)

            # TODO: replace this with MSE for infoGAN?
            D_loss = torch.mean(D_fake - D_real + LAMBDA_GRAD * gradPenalty)
            writer.add_scalar('Loss/Discriminator', D_loss.detach().item(), step)

            D_loss.backward(retain_graph=True)

            if i % WAVEGAN_DISC_NUPDATES == 0:

                # GENERATOR UPDATE
                optimizer_G.zero_grad()

                # G Loss
                fake = G(z)
                D_fake, _ = D(fake)
                G_loss = torch.mean(-D_fake)
                writer.add_scalar('Loss/Generator', G_loss.detach().item(), step)

                G_loss.backward(retain_graph=True)

                # TODO: try to keep these
                _, cHat = D(G(z))

                Q_loss = LAMBDA_INFO * criterion_Q(cHat, c)
                writer.add_scalar('Loss/Q_Network', Q_loss.detach().item(), step)

                Q_loss.backward()

                # Step
                optimizer_Q.step()
                optimizer_G.step()

            optimizer_D.step()
            step += 1

        if not epoch % SAVE_INT:
            torch.save(G.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_G.pt'))
            torch.save(D.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_D.pt'))

            torch.save(optimizer_G.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Gopt.pt'))
            torch.save(optimizer_D.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Dopt.pt'))
            torch.save(optimizer_Q.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Qopt.pt'))
