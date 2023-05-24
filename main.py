import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd


from data_handler import DataHandler
from ptbrnn import PTBRNN

epsilon = 1e-10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

def detach_h_and_c(h_and_c, use_gru, num_of_layers=2):
    for i in range(num_of_layers):
        if use_gru:
            h_and_c[i].detach_()
        else:
            for j in range(2):
                h_and_c[i][j].detach_()

    return h_and_c

def init_h_and_c(num_seq_in_batch, emb_dim, use_gru):
    h_and_c = []
    if use_gru:
        h_and_c.append(torch.zeros([1, num_seq_in_batch, emb_dim], device=device))  # layer 0
        h_and_c.append(torch.zeros([1, num_seq_in_batch, emb_dim], device=device))  # layer 1
    else:
        h_and_c.append((torch.zeros([1, num_seq_in_batch, emb_dim], device=device),
                        torch.zeros([1, num_seq_in_batch, emb_dim], device=device)))  # layer 0
        h_and_c.append((torch.zeros([1, num_seq_in_batch, emb_dim], device=device),
                        torch.zeros([1, num_seq_in_batch, emb_dim], device=device)))  # layer 1

    return h_and_c

def plot_grad_norms(grad_norms,):
    plt.plot(range(1, grad_norms.shape[1] + 1), torch.norm(torch.Tensor(grad_norms), dim=0))
    plt.xlabel('iter')
    plt.ylabel('grad norm')
    plt.title(f'grad norm per iter')
    plt.legend()
    plt.show()

def train(model, train_data, args, criterion, optimizer, device):
    model.train()
    seq_len = args.seq_len
    num_seq_in_batch = train_data.shape[0]
    num_of_batches = int(np.floor(train_data.shape[1]/seq_len))
    h_and_c = init_h_and_c(num_seq_in_batch, model.emb_dim, args.use_gru)
    grad_norm = np.zeros([11,num_of_batches])
    avg_ce = 0
    for i in range(0,num_of_batches*seq_len,seq_len):
        cur_iter = int(i/seq_len)
        h_and_c = detach_h_and_c(h_and_c, args.use_gru)
        input = train_data[:,i:i+seq_len]
        wanted_output = train_data[:,i+1:i+seq_len+1]
        input, wanted_output = input.to(device), wanted_output.to(device)
        optimizer.zero_grad()
        output, h_and_c = model(input, h_and_c)
        loss = criterion(output.reshape(-1,model.vocab_size), wanted_output.reshape(-1))
        loss.backward()
        optimizer.step()

        cur_perplexity = torch.exp(loss.detach())
        avg_ce += loss.detach()

        # calcs grad norm per iter
        for g in optimizer.param_groups:
            for ipg, param_group in enumerate(g['params']):
                grad_norm[ipg,cur_iter] = float(torch.norm(param_group.grad.view(-1)))

        if cur_iter % 100 == 0:
            print('iter = {}, perp = {:.3f}, grad_norm = {:.3f}'.format(cur_iter, cur_perplexity, torch.norm(torch.Tensor([grad_norm[:,cur_iter]]))))

    avg_perplexity = torch.exp(avg_ce/num_of_batches)
    return float(avg_perplexity)


def evaluate(model, test_data, args, criterion, device):
    model.eval()
    seq_len = args.seq_len
    num_seq_in_batch = test_data.shape[0]
    num_of_batches = int(np.floor(test_data.shape[1] / seq_len))
    h_and_c = init_h_and_c(num_seq_in_batch, model.emb_dim, args.use_gru)
    avg_ce=0
    with torch.no_grad():
        for i in range(0, num_of_batches * seq_len, seq_len):
            cur_iter = int(i / seq_len)
            input = test_data[:, i:i + seq_len]
            wanted_output = test_data[:, i + 1:i + seq_len + 1]
            input, wanted_output = input.to(device), wanted_output.to(device)
            output, h_and_c = model(input, h_and_c)
            loss = criterion(output.reshape(-1, model.vocab_size), wanted_output.reshape(-1))
            avg_ce += loss.detach()

        avg_perplexity = torch.exp(avg_ce/num_of_batches)
        print('avg perplexity = {}'.format(avg_perplexity))

    return float(avg_perplexity)


def parse_args():
    parser = argparse.ArgumentParser(
        description='''Next word prediction model based on multilayer LSTM\GRU.\nUsage examples:
    train - 'python3 main.py --phase train --epochs 13 --dropout_p 0 --show --lr 3.5 --weight_init 0.1 --gdp 4 --seq_len 20'
    test the model - 'python3 main.py --phase test --model-path model.pt' ''',
        formatter_class=argparse.RawTextHelpFormatter)
    # parser.add_argument('-i', '--input', type=str, default='./data', help='Directory for data dir')
    parser.add_argument('-o', '--output', type=str, default='./dump', help='Directory for output dir')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train/test/plot')
    parser.add_argument('--dropout_p', type=float, default=0, help='Add dropout, enter probability')
    parser.add_argument('--lr', type=float, default=1, help='learning rate')
    parser.add_argument('--epochs', type=int, default=13, help='Training number of epochs')
    parser.add_argument('--model-path', type=str, default=None, help='Path to saved model, test only')
    parser.add_argument('--show', action='store_true', help='Show plots')
    parser.add_argument('--seq_len', type=int, default=20, help='sequence length')
    parser.add_argument('--use_gru', action='store_true', help='use gru instead of lstm')
    parser.add_argument('--rand_seed', type=float, default=1, help='random seed')
    parser.add_argument('--weight_init', type=float, default=0.1, help='weight_init')
    parser.add_argument('--gdp', type=int, default=4, help='grad decay param - from that iter, start decaying the grad by a factor of 0.5')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.rand_seed)
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    data_handler = DataHandler()
    train_data = data_handler.get_data('train')
    val_data = data_handler.get_data('validation')
    test_data = data_handler.get_data('test')

    model = PTBRNN(vocab_size=data_handler.vocab_size, dropout_p=args.dropout_p, use_gru=args.use_gru).to(device)
    model.init_weights(args.weight_init)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    num_epochs = args.epochs

    if args.phase == "train":
        train_perp_list = []
        val_perp_list = []
        results = []
        try:
            for epoch in range(num_epochs):
                print('\nstarting epoch {}'.format(epoch))
                if epoch >= args.gdp:
                    toggle_print = 1
                    for g in optimizer.param_groups:
                        g['lr'] *= 0.5
                        if toggle_print == 1:
                            print('reduced lr by factor of 2. current lr is {:.3f}'.format(g['lr']))
                            toggle_print = 0

                avg_train_perplexity = train(model, train_data, args, criterion, optimizer, device)

                # Evaluate the model on the validation set
                avg_val_perplexity = evaluate(model, val_data, args, criterion, device)

                print('Epoch {}: avg training perp = {:.3f}, avg validation perp = {:.3f}'.format(
                    epoch, avg_train_perplexity, avg_val_perplexity))

                train_perp_list.append(avg_train_perplexity)
                val_perp_list.append(avg_val_perplexity)

        except KeyboardInterrupt:
            print("Training stopped by keyboard interrupt")

        if args.use_gru:
            model_name = f'gru_dropout_{args.dropout_p}'
        else:
            model_name = f'lstm_dropout_{args.dropout_p}'
        torch.save(model.state_dict(), f"{args.output}/{model_name}.pt")

        results.append({'scenario': model_name, 'train_perp': f'{train_perp_list[-1]:.3f}',
                        'test_perp': f'{val_perp_list[-1]:.3f}'})
        df = pd.DataFrame(results)
        df.to_csv(f'{args.output}/results.csv', mode='a', header=not os.path.isfile(f'{args.output}/results.csv'),
                  index='False')

        plt.plot(range(1, num_epochs + 1), train_perp_list, label='Train')
        plt.plot(range(1, num_epochs + 1), val_perp_list, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('average perplexity')
        plt.title(f'average perplexity vs. Epoch for {model_name}')
        plt.legend()
        plt.savefig(f"{args.output}/{model_name}.png")
        if args.show:
            plt.show()

    elif args.phase == "test":
        if args.model_path == None:
            print("Specify trained model path!")
        else:
            if 'lstm' in args.model_path:
                args.use_gru = False
            else:
                args.use_gru = True
            model = PTBRNN(vocab_size=data_handler.vocab_size, dropout_p=args.dropout_p, use_gru=args.use_gru).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            avg_test_perplexity = evaluate(model, test_data, args, criterion, device)
            print(f"Test average perplexity: {avg_test_perplexity} for {args.model_path}")

    elif args.phase == "plot":
        df = pd.read_csv(f'{args.output}/results.csv')
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.axis('tight')
        df.drop(columns=df.columns[0], axis=1, inplace=True)
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', colWidths=[.7, .2, .2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        plt.title("Results for different parameters")
        if args.show:
            plt.show()
        else:
            plt.savefig(f'{args.output}/results.png')



