import torch
from dataset import *
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model_1d import Discriminator
from generator_model_1d import Generator
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, use_img):
    Scrappie_reals = 0
    Scrappie_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (real, scrappie) in enumerate(loop):  # zebra is real, horse is scrappie equivalent
        real = real.to(config.DEVICE).float()
        scrappie = scrappie.to(config.DEVICE).float()
        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_scrappie = gen_H(real)
            D_Scrappie_real = disc_H(scrappie)
            D_Scrappie_fake = disc_H(fake_scrappie.detach())
            Scrappie_reals += D_Scrappie_real.mean().item()
            Scrappie_fakes += D_Scrappie_fake.mean().item()
            D_Scrappie_real_loss = mse(
                D_Scrappie_real, torch.ones_like(D_Scrappie_real))
            D_Scrappie_fake_loss = mse(
                D_Scrappie_fake, torch.zeros_like(D_Scrappie_fake))
            D_Scrappie_loss = D_Scrappie_real_loss + D_Scrappie_fake_loss

            fake_real = gen_Z(scrappie)
            D_real_signal_real = disc_Z(real)
            D_real_signal_fake = disc_Z(fake_real.detach())
            D_real_signal_real_loss = mse(
                D_real_signal_real, torch.ones_like(D_real_signal_real))
            D_real_signal_fake_loss = mse(
                D_real_signal_fake, torch.zeros_like(D_real_signal_fake))
            D_real_signal_loss = D_real_signal_real_loss + D_real_signal_fake_loss

            # put it togethor
            D_loss = (D_Scrappie_loss + D_real_signal_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_Scrappie_fake = disc_H(fake_scrappie)
            D_real_signal_fake = disc_Z(fake_real)
            loss_G_H = mse(D_Scrappie_fake, torch.ones_like(D_Scrappie_fake))
            loss_G_Z = mse(D_real_signal_fake,
                           torch.ones_like(D_real_signal_fake))

            # cycle loss
            cycle_zebra = gen_Z(fake_scrappie)
            cycle_horse = gen_H(fake_real)
            cycle_zebra_loss = l1(real, cycle_zebra)
            cycle_horse_loss = l1(scrappie, cycle_horse)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = gen_Z(real)
            identity_horse = gen_H(scrappie)
            identity_zebra_loss = l1(real, identity_zebra)
            identity_horse_loss = l1(scrappie, identity_horse)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + cycle_horse_loss * config.LAMBDA_CYCLE
                + identity_horse_loss * config.LAMBDA_IDENTITY
                + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            if use_img:
                save_image(fake_scrappie*0.5+0.5,
                           f"saved_images/horse_{idx}.png")
                save_image(fake_real*0.5+0.5, f"saved_images/zebra_{idx}.png")
            else:
                if not os.path.exists('./signals_result'):
                    os.makedirs('signals_result')
                torch.save([scrappie, real, fake_scrappie, fake_real],
                           'signals_result/signals_GAN.pt')
            #     plt.plot(fake_horse, label='fake_horse')
            #     plt.plot(fake_zebra, label='fake_zebra')
            #     plt.legend()
            #     plt.savefig(f'pic{idx}')

        loop.set_postfix(H_real=Scrappie_reals/(idx+1),
                         H_fake=Scrappie_fakes/(idx+1))


def test_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, use_img):
    Scrappie_reals = 0
    Scrappie_fakes = 0
    loop = tqdm(loader, leave=True)

    with torch.no_grad():
        # zebra is real, horse is scrappies equivalent
        for idx, (real, scrappie) in enumerate(loop):
            real = real.to(config.DEVICE).float()
            scrappie = scrappie.to(config.DEVICE).float()
            # Train Discriminators H and Z
            with torch.cuda.amp.autocast():
                fake_scrappie = gen_H(real)
                D_Scrappie_real = disc_H(scrappie)
                D_Scrappie_fake = disc_H(fake_scrappie.detach())
                Scrappie_reals += D_Scrappie_real.mean().item()
                Scrappie_fakes += D_Scrappie_fake.mean().item()
                fake_real_signal = gen_Z(scrappie)

                if idx == 0:
                    scrappie_list = scrappie
                    real_list = real
                    f_s_list = fake_scrappie
                    f_r_list = fake_real_signal
                else:
                    scrappie_list = torch.cat((scrappie_list, scrappie))
                    real_list = torch.cat((real, real_list))
                    f_s_list = torch.cat((fake_scrappie, f_s_list))
                    f_r_list = torch.cat((fake_real_signal, f_r_list))

        if use_img:
            save_image(fake_scrappie*0.5+0.5,
                       f"saved_images/horse_{idx}.png")
            save_image(fake_real_signal*0.5+0.5,
                       f"saved_images/zebra_{idx}.png")
        else:
            if not os.path.exists('./signals_result'):
                os.makedirs('signals_result')
            torch.save([scrappie, real, fake_scrappie, fake_real_signal],
                       'signals_result/signals_GAN_test.pt')
            #     plt.plot(fake_horse, label='fake_horse')
            #     plt.plot(fake_zebra, label='fake_zebra')
            #     plt.legend()
            #     plt.savefig(f'pic{idx}')

            loop.set_postfix(H_real=Scrappie_reals/(idx+1),
                             H_fake=Scrappie_fakes/(idx+1))


def main():
    training = int(input('0-test, 1 -train: '))
    use_img = False
    if use_img:
        in_channels = 3
        img_channels = 3
    else:
        in_channels = 1
        img_channels = 1

    disc_H = Discriminator(in_channels=in_channels).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=in_channels).to(config.DEVICE)
    gen_Z = Generator(img_channels=img_channels,
                      num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=img_channels,
                      num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        try:
            load_checkpoint(
                config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
            )
            load_checkpoint(
                config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
            )
            load_checkpoint(
                config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
            )
            load_checkpoint(
                config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
            )
        except:
            print('could not load - model not existing')

    if use_img:
        dataset = HorseZebraDataset(
            root_horse=config.TRAIN_DIR+"/horses", root_zebra=config.TRAIN_DIR+"/zebras", transform=config.transforms
        )
        val_dataset = HorseZebraDataset(
            root_horse="cyclegan_test/horse1", root_zebra="cyclegan_test/zebra1", transform=config.transforms
        )
    else:
        X_data = pd.read_csv('./real_signals_tr_d1s1.csv').to_numpy()
        y_data = pd.read_csv('./real_labels_tr_d1s1.csv').to_numpy()
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_data, y_data, test_size=0.3)

        X_data_s = pd.read_csv('./scrappie_signals.csv').to_numpy()
        y_data_s = pd.read_csv('./scrappie_labels.csv').to_numpy()
        X_s_train, X_s_validation, y_s_train, y_s_validation = train_test_split(
            X_data_s, y_data_s, test_size=0.3)

        dataset = SignalDataset(X_train, y_train, X_s_train, y_s_train)
        val_dataset = SignalDataset(
            X_validation, y_validation, X_s_validation, y_s_validation)

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    if training == 1:
        print('TRAINING BEGIN')
        for epoch in range(config.NUM_EPOCHS):
            print(epoch)
            train_fn(disc_H, disc_Z, gen_Z, gen_H, loader,
                     opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, use_img)

            if config.SAVE_MODEL:
                save_checkpoint(
                    gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
                save_checkpoint(
                    gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
                save_checkpoint(disc_H, opt_disc,
                                filename=config.CHECKPOINT_CRITIC_H)
                save_checkpoint(disc_Z, opt_disc,
                                filename=config.CHECKPOINT_CRITIC_Z)
    else:
        print('TESTING BEGIN')
        test_fn(disc_H, disc_Z, gen_Z, gen_H, val_loader,
                opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, use_img)


if __name__ == "__main__":
    main()
