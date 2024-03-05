import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='SemiGCL')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--source_dataset', type=str, default='citationv1')
    parser.add_argument('--target_dataset', type=str, default='acmv9')
    parser.add_argument('--target_shot', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr_cly', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--alpha_ppr', type=float, default=0.2)
    parser.add_argument('--diff_k', type=int, default=10)
    parser.add_argument('--aggregator_class', type=str, default='diffusion')
    parser.add_argument('--n_samples', type=str, default='10,10')
    parser.add_argument('--output_dims', type=str, default='256,64')
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--is_blog', action='store_true')
    parser.add_argument('--cal_ssl', action='store_true')
    parser.add_argument('--ssl_param', type=float, default=1.0)
    parser.add_argument('--mme_param', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    return args
