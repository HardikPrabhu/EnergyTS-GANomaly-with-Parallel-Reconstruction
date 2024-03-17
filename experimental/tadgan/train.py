import logging
import json
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import model

logging.basicConfig(filename='train.log', level=logging.DEBUG)


def critic_x_iteration(sample):
    optim_cx.zero_grad()

    x = sample.view(1, batch_size, signal_shape)
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    critic_score_valid_x = torch.mean(torch.ones(valid_x.shape) * valid_x)  # Wasserstein Loss

    # The sampled z are the anomalous points - points deviating from actual distribution of z (obtained through encoding x)
    z = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    x_ = decoder(z)
    fake_x = critic_x(x_)
    fake_x = torch.squeeze(fake_x)
    critic_score_fake_x = torch.mean(torch.ones(fake_x.shape) * fake_x)  # Wasserstein Loss

    alpha = torch.rand(x.shape)
    ix = Variable(alpha * x + (1 - alpha) * x_)  # Random Weighted Average
    ix.requires_grad_(True)
    v_ix = critic_x(ix)
    v_ix.mean().backward()
    gradients = ix.grad
    # Gradient Penalty Loss
    gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

    # Critic has to maximize Cx(Valid X) - Cx(Fake X).
    # Maximizing the above is same as minimizing the negative.
    wl = critic_score_fake_x - critic_score_valid_x
    loss = wl + gp_loss
    loss.backward()
    optim_cx.step()

    return loss


def critic_z_iteration(sample):
    optim_cz.zero_grad()

    x = sample.view(1, batch_size, signal_shape)
    z = encoder(x)
    print("all good", x.shape, z.shape)
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)
    critic_score_valid_z = torch.mean(torch.ones(valid_z.shape) * valid_z)

    z_ = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    fake_z = critic_z(z_)
    fake_z = torch.squeeze(fake_z)
    critic_score_fake_z = torch.mean(torch.ones(fake_z.shape) * fake_z)  # Wasserstein Loss

    wl = critic_score_fake_z - critic_score_valid_z

    alpha = torch.rand(z.shape)
    iz = Variable(alpha * z + (1 - alpha) * z_)  # Random Weighted Average
    iz.requires_grad_(True)
    v_iz = critic_z(iz)
    v_iz.mean().backward()
    gradients = iz.grad
    gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))

    loss = wl + gp_loss
    loss.backward()
    optim_cz.step()

    return loss


def encoder_iteration(sample):
    optim_enc.zero_grad()

    x = sample.view(1, batch_size, signal_shape)
    valid_x = critic_x(x)
    valid_x = torch.squeeze(valid_x)
    critic_score_valid_x = torch.mean(torch.ones(valid_x.shape) * valid_x)  # Wasserstein Loss

    z = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    x_ = decoder(z)
    fake_x = critic_x(x_)
    fake_x = torch.squeeze(fake_x)
    critic_score_fake_x = torch.mean(torch.ones(fake_x.shape) * fake_x)

    enc_z = encoder(x)
    gen_x = decoder(enc_z)

    mse = mse_loss(x.float(), gen_x.float())
    loss_enc = mse + critic_score_valid_x - critic_score_fake_x
    loss_enc.backward(retain_graph=True)
    optim_enc.step()

    return loss_enc


def decoder_iteration(sample):
    optim_dec.zero_grad()

    x = sample.view(1, batch_size, signal_shape)
    z = encoder(x)
    valid_z = critic_z(z)
    valid_z = torch.squeeze(valid_z)
    critic_score_valid_z = torch.mean(torch.ones(valid_z.shape) * valid_z)

    z_ = torch.empty(1, batch_size, latent_space_dim).uniform_(0, 1)
    fake_z = critic_z(z_)
    fake_z = torch.squeeze(fake_z)
    critic_score_fake_z = torch.mean(torch.ones(fake_z.shape) * fake_z)

    enc_z = encoder(x)
    gen_x = decoder(enc_z)

    mse = mse_loss(x.float(), gen_x.float())
    loss_dec = mse + critic_score_valid_z - critic_score_fake_z
    loss_dec.backward(retain_graph=True)
    optim_dec.step()

    return loss_dec


def train(n_epochs=2000):
    logging.debug('Starting training')
    cx_epoch_loss = list()
    cz_epoch_loss = list()
    encoder_epoch_loss = list()
    decoder_epoch_loss = list()

    for epoch in range(n_epochs):
        logging.debug('Epoch {}'.format(epoch))
        n_critics = 5

        cx_nc_loss = list()
        cz_nc_loss = list()

        for i in range(n_critics):
            cx_loss = list()
            cz_loss = list()

            for batch, sample in enumerate(train_loader):
                loss = critic_x_iteration(sample)
                cx_loss.append(loss)

                loss = critic_z_iteration(sample)
                cz_loss.append(loss)

            cx_nc_loss.append(torch.mean(torch.tensor(cx_loss)))
            cz_nc_loss.append(torch.mean(torch.tensor(cz_loss)))

        logging.debug('Critic training done in epoch {}'.format(epoch))
        encoder_loss = list()
        decoder_loss = list()

        for batch, sample in enumerate(train_loader):
            enc_loss = encoder_iteration(sample)
            dec_loss = decoder_iteration(sample)
            encoder_loss.append(enc_loss)
            decoder_loss.append(dec_loss)

        cx_epoch_loss.append(torch.mean(torch.tensor(cx_nc_loss)))
        cz_epoch_loss.append(torch.mean(torch.tensor(cz_nc_loss)))
        encoder_epoch_loss.append(torch.mean(torch.tensor(encoder_loss)))
        decoder_epoch_loss.append(torch.mean(torch.tensor(decoder_loss)))
        logging.debug('Encoder decoder training done in epoch {}'.format(epoch))
        logging.debug('critic x loss {:.3f} critic z loss {:.3f} \nencoder loss {:.3f} decoder loss {:.3f}\n'.format(
            cx_epoch_loss[-1], cz_epoch_loss[-1], encoder_epoch_loss[-1], decoder_epoch_loss[-1]))

        if epoch % 10 == 0:
            torch.save(encoder.state_dict(), encoder.encoder_path)
            torch.save(decoder.state_dict(), decoder.decoder_path)
            torch.save(critic_x.state_dict(), critic_x.critic_x_path)
            torch.save(critic_z.state_dict(), critic_z.critic_z_path)


if __name__ == "__main__":
    # Load config
    with open('../../config.json', 'r') as file:
        config = json.load(file)

    prefix = "../../"

    # Training configs
    nz = config['training']['latent_dim']
    window_size = config['preprocessing']['window_size']
    batch_size = config['training']['batch_size']
    signal_shape = window_size  # window-size
    lr = 0.0002
    latent_space_dim = config['training']['latent_dim']
    n_epochs = config['training']['num_epochs']

    b_id = "all"
    if config['data']["only_building"] is not None:
        b_id = config['data']["only_building"]  # 1 building at a time (recommended)

    encoder_path = f'encoder_{b_id}.pth'
    decoder_path = f'decoder_{b_id}.pth'
    critic_x_path = f'critic_x_{b_id}.pth'
    critic_z_path = f'critic_z_{b_id}.pth'

    # Add dataset code here...
    X_train = torch.load(f"X_train_{b_id}.pt")
    train_loader = DataLoader(X_train, batch_size=batch_size, drop_last=True)

    logging.info('Number of train datapoints is {}'.format(len(X_train)))

    encoder = model.Encoder(encoder_path, signal_shape)
    decoder = model.Decoder(decoder_path, signal_shape)
    critic_x = model.CriticX(critic_x_path, signal_shape)
    critic_z = model.CriticZ(critic_z_path)
    mse_loss = torch.nn.MSELoss()
    optim_enc = optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_dec = optim.Adam(decoder.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_cx = optim.Adam(critic_x.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_cz = optim.Adam(critic_z.parameters(), lr=lr, betas=(0.5, 0.999))

    train(n_epochs=n_epochs)

    # save all the models
    torch.save(encoder, encoder_path)
    torch.save(decoder,decoder_path)
    torch.save(critic_x,critic_x_path)
    torch.save(critic_z,critic_z_path)
