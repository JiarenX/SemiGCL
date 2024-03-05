from parse_args import *
from utils import *
from mvgrl import *
import torch.nn.functional as F
import loss_func


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    adj_s, adj_val_s, diff_idx_s, diff_val_s, feature_s, label_s, idx_train_s, _, _, idx_tot_s = load_data(dataset=args.source_dataset+'.mat', device=device,
                                                                                                           seed=args.seed, is_blog=args.is_blog,
                                                                                                           alpha_ppr=args.alpha_ppr, diff_k=args.diff_k)
    adj_t, adj_val_t, diff_idx_t, diff_val_t, feature_t, label_t, idx_train_t, idx_val_t, idx_test_t, idx_tot_t = load_data(dataset=args.target_dataset+'.mat', device=device,
                                                                                                                            shot=args.target_shot, seed=args.seed, is_blog=args.is_blog,
                                                                                                                            alpha_ppr=args.alpha_ppr, diff_k=args.diff_k)
    n_samples = args.n_samples.split(',')
    output_dims = args.output_dims.split(',')
    emb_model = GraphSAGE(**{
        "aggregator_class": aggregator_lookup[args.aggregator_class],
        "input_dim": feature_s.shape[1],
        "layer_specs": [
            {
                "n_sample": int(n_samples[0]),
                "output_dim": int(output_dims[0]),
                "activation": F.relu,
            },
            {
                "n_sample": int(n_samples[-1]),
                "output_dim": int(output_dims[-1]),
                "activation": F.relu,
            }
        ],
        "device": device
    }).to(device)
    cly_model = Predictor(num_class=label_s.shape[1], inc=2*int(output_dims[-1]), temp=args.T).to(device)
    total_params = list(emb_model.parameters()) + list(cly_model.parameters())
    if args.cal_ssl:
        ssl_model = MVGRL(int(output_dims[-1])).to(device)
        total_params += list(ssl_model.parameters())
    cly_optim = torch.optim.Adam(total_params, lr=args.lr_cly, weight_decay=args.weight_decay)
    lr_lambda = lambda epoch: (1 + 10*float(epoch) / (args.epochs+0))**(-0.75)
    scheduler = torch.optim.lr_scheduler.LambdaLR(cly_optim, lr_lambda=lr_lambda)
    best_micro_f1, best_macro_f1 = 0, 0
    num_batch = int(max(feature_s.shape[0]/(args.batch_size/2), idx_test_t.shape[0]/(args.batch_size/2)))
    for epoch in range(args.epochs):
        s_batches = batch_generator(idx_tot_s, int(args.batch_size/2))
        t_batches = batch_generator(idx_test_t, int(args.batch_size/2))
        emb_model.train()
        cly_model.train()
        p = float(epoch) / args.epochs
        grl_lambda = min(2. / (1. + np.exp(-10. * p)) - 1, 0.1)/args.mme_param
        for iter in range(num_batch):
            b_nodes_s = next(s_batches)
            b_nodes_t = next(t_batches)
            source_features, cly_loss_s = do_iter(emb_model, cly_model, adj_s, adj_val_s, feature_s, label_s, diff_idx_s,
                                                  diff_val_s, idx=b_nodes_s)
            target_features, _ = do_iter(emb_model, cly_model, adj_t, adj_val_t, feature_t, label_t, diff_idx_t,
                                                 diff_val_t, idx=b_nodes_t)
            if idx_train_t.shape[0] == 0:
                total_cly_loss = cly_loss_s
            else:
                feats_train_t, cly_loss_t = do_iter(emb_model, cly_model, adj_t, adj_val_t, feature_t, label_t, diff_idx_t,
                                                    diff_val_t, idx=idx_train_t)
                total_cly_loss = cly_loss_s + cly_loss_t
            ssl_loss = torch.zeros(1).to(device)
            if args.cal_ssl:
                ssl_model.train()
                shuf_idx_s = np.arange(label_s.shape[0])
                np.random.shuffle(shuf_idx_s)
                shuf_feat_s = feature_s[shuf_idx_s, :]
                shuf_idx_t = np.arange(label_t.shape[0])
                np.random.shuffle(shuf_idx_t)
                shuf_feat_t = feature_t[shuf_idx_t, :]
                h_s_1 = emb_model(b_nodes_s, adj_s, adj_val_s, feature_s)
                h_s_2 = emb_model(b_nodes_s, diff_idx_s, diff_val_s, feature_s)
                h_s_3 = emb_model(b_nodes_s, adj_s, adj_val_s, shuf_feat_s)
                h_s_4 = emb_model(b_nodes_s, diff_idx_s, diff_val_s, shuf_feat_s)
                logits_s = ssl_model(h_s_1, h_s_2, h_s_3, h_s_4)
                labels_ssl_s = torch.cat([torch.ones(h_s_1.shape[0] * 2), torch.zeros(h_s_1.shape[0] * 2)]).unsqueeze(0).to(device)
                ssl_loss_s = F.binary_cross_entropy_with_logits(logits_s, labels_ssl_s)
                b_nodes_t_plus = torch.cat((b_nodes_t, idx_train_t), dim=0)
                h_t_1 = emb_model(b_nodes_t_plus, adj_t, adj_val_t, feature_t)
                h_t_2 = emb_model(b_nodes_t_plus, diff_idx_t, diff_val_t, feature_t)
                h_t_3 = emb_model(b_nodes_t_plus, adj_t, adj_val_t, shuf_feat_t)
                h_t_4 = emb_model(b_nodes_t_plus, diff_idx_t, diff_val_t, shuf_feat_t)
                logits_t = ssl_model(h_t_1, h_t_2, h_t_3, h_t_4)
                labels_ssl_t = torch.cat([torch.ones(h_t_1.shape[0] * 2), torch.zeros(h_t_1.shape[0] * 2)]).unsqueeze(0).to(device)
                ssl_loss_t = F.binary_cross_entropy_with_logits(logits_t, labels_ssl_t)
                ssl_loss = args.ssl_param * (ssl_loss_s + ssl_loss_t)
            domain_loss = args.mme_param * loss_func.adentropy(cly_model, target_features, grl_lambda)
            loss = total_cly_loss + ssl_loss + domain_loss
            cly_optim.zero_grad()
            loss.backward()
            cly_optim.step()

        emb_model.eval()
        cly_model.eval()
        cly_loss_bat_s, micro_f1_s, macro_f1_s, embs_whole_s, targets_whole_s = evaluate(emb_model, cly_model, adj_s, adj_val_s, feature_s,
                                                                                         label_s, diff_idx_s, diff_val_s, idx_tot_s, args.batch_size, mode='test')
        print("epoch {:03d} | source loss {:.4f} | source micro-F1 {:.4f} | source macro-F1 {:.4f}".
              format(epoch, cly_loss_bat_s, micro_f1_s, macro_f1_s))
        cly_loss_bat_t, micro_f1_t, macro_f1_t, embs_whole_t, targets_whole_t = evaluate(emb_model, cly_model, adj_t, adj_val_t, feature_t,
                                                                                         label_t, diff_idx_t, diff_val_t, idx_test_t, args.batch_size, mode='test')
        print("target loss {:.4f} | target micro-F1 {:.4f} | target macro-F1 {:.4f}".format(cly_loss_bat_t, micro_f1_t, macro_f1_t))
        if micro_f1_t > best_micro_f1:
            best_micro_f1 = micro_f1_t
            best_macro_f1 = macro_f1_t
        scheduler.step()

    print("test metrics on target graph:")
    print('---------- random seed: {:03d} ----------'.format(args.seed))
    print("micro-F1 {:.4f} | macro-F1 {:.4f}".format(best_micro_f1, best_macro_f1))


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    main(args)
