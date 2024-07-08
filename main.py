import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
PAINT_POINTS = np.vstack([np.linspace(-1, 1, 16) for _ in range(32)])


def get_real_data(data_dim, batch_size):
    for i in range(300):                #每个循环产生batch_size个长度为data_dim的二次函数曲线
        a = np.random.uniform(1, 2, size=batch_size)[:, np.newaxis]
        base = np.linspace(-1, 1, data_dim)[np.newaxis, :].repeat(batch_size, axis=0)
        yield a * np.power(base, 2) + (a-1)      #生成器，可以循环调用取值，可以返回值

# test

# tmp = next(get_real_data(16, 32))
# plt.plot(tmp[0, :])
# plt.show()
# # print(next(get_real_data(16, 32)))
# end


class G(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(G, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.generator = nn.Sequential(
            nn.Linear(self.latent_dim, 32),
            nn.ReLU()
        )
        self.out = nn.Linear(32, self.data_dim)

    def forward(self, x):
        x = self.generator(x)
        output = self.out(x)
        return output


class D(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(D, self).__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.discriminator = nn.Sequential(
            nn.Linear(self.data_dim, 32),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.discriminator(x)
        output = self.out(x)
        return output


# 超参
LATENT_DIM = 10
DATA_DIM = 16
BATCH_SIZE = 32
EPOCH = 200
N_IDEAS = 10
LR_G, LR_D = 0.0001, 0.0001
g = G(latent_dim=LATENT_DIM, data_dim=DATA_DIM).cuda()
d = D(latent_dim=LATENT_DIM, data_dim=DATA_DIM).cuda()

opt_G = torch.optim.Adam(g.parameters(), lr=LR_G)
opt_D = torch.optim.Adam(d.parameters(), lr=LR_D)

for epoch in range(EPOCH):
    for t, data in enumerate(get_real_data(DATA_DIM, BATCH_SIZE)):
        G_ideas = torch.randn(BATCH_SIZE, N_IDEAS).cuda()
        G_paintings = g(G_ideas)
        prob_artist1 = d(G_paintings)
        G_loss = torch.mean(torch.log(1. - prob_artist1))
        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        prob_artist0 = d(torch.from_numpy(data).float().cuda())
        prob_artist1 = d(G_paintings.detach())
        D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        if t % 50 == 0:  # plotting
            print('Epoch: ', epoch, 'G loss', G_loss.data.cpu().numpy(), 'D loss', D_loss.data.cpu().numpy())
            plt.cla()
            plt.plot(PAINT_POINTS[0], G_paintings.data.cpu().numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
            plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
            plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
            plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.cpu().numpy().mean(),
                     fontdict={'size': 13})
            plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.cpu().numpy(), fontdict={'size': 13})
            plt.ylim((0, 3));
            plt.legend(loc='upper right', fontsize=10);
            plt.draw();
            plt.pause(0.01)







