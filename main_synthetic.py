import sys

from tqdm import trange

import torchsl
from data_visualizer import DataVisualizer
from torchsl.utils.data import MultiviewTensorDataset
from torchsl.utils.data.synthetics import *


def main():
    dv = DataVisualizer()
    toy_datasets = [
        make_multiview_blobs_old(n_classes=3, n_views=3, n_features=2, seed=111),
        make_multiview_blobs_old(n_classes=5, n_views=3, n_features=3, seed=5),
    ]
    common_space_dim = 1
    pcmvda_epoches = 100

    for dataset_id, (Xs, y) in enumerate(toy_datasets):
        print(f'Toy dataset {dataset_id + 1}:', Xs[0].shape, y.shape)

        dv.multiview_scatter(Xs, y, kde=False, class_legend=True, view_legend=True, title='Original')

        # mvda
        projector = torchsl.nn.Linears([X.size(1) for X in Xs], common_space_dim, requires_grad=False)
        optim = torchsl.optim.GEPSolver(projector.parameters(), solver='svd', eps=0)
        Sw, Sb = torchsl.mvda(Xs, y)
        optim.step(Sw, Sb)
        Ys = projector(Xs)
        dv.multiview_scatter(Ys, y, kde=True, class_legend=False, view_legend=False, title='MvDA')

        # pcmvda (initialized from mvda)
        projector.requires_grad_(True)
        optim = torch.optim.AdamW(projector.parameters(), lr=0.01)
        dataset = MultiviewTensorDataset(Xs, y, batch_size=len(y), stratified=True, shuffle=False)
        pbar = trange(pcmvda_epoches, file=sys.stdout)
        for epoch_id, epoch in enumerate(pbar):
            for batch_id, (Xs_batch, y_batch) in enumerate(dataset):
                optim.zero_grad()
                loss = torchsl.pcmvda_loss(projector(Xs_batch), y_batch)
                loss.backward(retain_graph=True)
                optim.step()
                pbar.set_description(f"[{epoch_id}] pcmvda_loss: {loss.item():.4f}")

        projector.eval()
        with torch.no_grad():
            Ys = projector(Xs)
        dv.multiview_scatter(Ys, y, kde=True, class_legend=False, view_legend=False, title='pc-MvDA')
        dv.show(title=f'Toy dataset {dataset_id + 1}', adjust=(0.06, 0.06, 0.96, 0.85))
        print()


if __name__ == '__main__':
    main()
