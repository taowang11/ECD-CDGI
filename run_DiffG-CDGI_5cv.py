import argparse
import numpy as np
import torch
from sklearn import metrics
from model import ECD_CDGINet
from data.data_loader import load_net_specific_data
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--is_5_CV_test', type=bool, default=True, help='Run 5-CV test.')
parser.add_argument('--dataset_file', type=str, default='./data/PathNet/dataset_PathNet_ten_5CV.pkl',
                    help='The path of the input pkl file.')  # When setting is_5_CV_test=True, make sure the pkl file include masks of different 5CV splits.
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--w_decay', type=float, default=0.00001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--in_channels', type=int, default=58, help='Dimension of node features.')
parser.add_argument('--hidden_channels', type=int, default=100, help='Dimension of hidden Linear layers.')
parser.add_argument('--device', type=int, default=0, help='The id of GPU.')
args = parser.parse_args()
device = torch.device('cuda:%d' % args.device if torch.cuda.is_available() else 'cpu')
data = load_net_specific_data(args)
data = data.to(device)


@torch.no_grad()
def test(data, mask):
    model.eval()
    x = model(data)
    pred = torch.sigmoid(x[mask])
    precision, recall, _thresholds = metrics.precision_recall_curve(data.y[mask].cpu().numpy(),
                                                                    pred.cpu().detach().numpy())
    area = metrics.auc(recall, precision)
    return metrics.roc_auc_score(data.y[mask].cpu().numpy(), pred.cpu().detach().numpy()), area, data.y[
        mask].cpu().numpy(), pred.cpu().detach().numpy()


# Ten times of 5_CV
AUC = np.zeros(shape=(10, 5))
AUPR = np.zeros(shape=(10, 5))

for i in range(10):
    for cv_run in range(5):
        tr_mask, te_mask = data.mask[i][cv_run]
        model = ECD_CDGINet(args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            # Training model
            model.train()
            optimizer.zero_grad()
            pred = model(data)
            loss = F.binary_cross_entropy_with_logits(pred[tr_mask], data.y[tr_mask].view(-1, 1))
            loss.backward()
            optimizer.step()
        AUC[i][cv_run], AUPR[i][cv_run], a, b = test(data, te_mask)
        a, b, all_y_true1, all_y_scores1 = test(data, te_mask)
        print(f'Training epoch: {epoch:03d}')
        # print('Round--%d CV--%d  AUC: %.5f, AUPR: %.5f' % (i, cv_run + 1, AUC[i][cv_run], AUPR[i][cv_run]))
        print('Round--%d CV--%d  AUC: %.5f, AUPR: %.5f' % (i, cv_run + 1, AUC[i][cv_run], AUPR[i][cv_run]))
    print('Round--%d Mean AUC: %.5f, Mean AUPR: %.5f' % (i, np.mean(AUC[i, :]), np.mean(AUPR[i, :])))

print('10 rounds for 5CV-- Mean AUC: %.4f, Mean AUPR: %.4f' % (AUC.mean(), AUPR.mean()))
