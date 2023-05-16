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
torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: {}'.format(device))

def detach_h_and_c(h_and_c, num_of_layers=2):
    for i in range(num_of_layers):
        for j in range(2):
            h_and_c[i][j].detach_()

    return h_and_c

def plot_grad_norms(grad_norms):
    legend_names = ['layer_0_h', 'layer_0_x', 'layer_0_h_bias', 'layer_0_x_bias', 'layer_1_h', 'layer_1_x', 'layer_1_h_bias', 'layer_1_x_bias', 'token2emb_mat', 'emb2token_mat', 'emb2token_bias']
    for i in range(grad_norms.shape[0]):
        plt.plot(range(1, grad_norms.shape[1] + 1), 10*np.log10(grad_norms[i,:]), label=legend_names[i])

    plt.xlabel('iter')
    plt.ylabel('grad norms [dB]')
    plt.title(f'grad norms per iter')
    plt.legend()
    plt.show()

def train(model, train_data, args, criterion, optimizer, device):
    model.train()
    seq_len = args.seq_len
    num_seq_in_batch = train_data.shape[0]
    num_of_batches = int(np.floor(train_data.shape[1]/seq_len))
    # num_of_batches = 5

    h_and_c = []
    h_and_c.append((torch.zeros([1, num_seq_in_batch, model.emb_dim], device=device),
                    torch.zeros([1, num_seq_in_batch, model.emb_dim], device=device)))  # layer 0
    h_and_c.append((torch.zeros([1, num_seq_in_batch, model.emb_dim], device=device),
                    torch.zeros([1, num_seq_in_batch, model.emb_dim], device=device)))  # layer 1

    avg_perplexity=0
    grad_norm = np.zeros([11,num_of_batches])
    for i in range(0,num_of_batches*seq_len,seq_len):
        cur_iter = int(i/seq_len)
        h_and_c = detach_h_and_c(h_and_c)
        input = train_data[:,i:i+seq_len]
        wanted_output = train_data[:,i+1:i+seq_len+1]
        input, wanted_output = input.to(device), wanted_output.to(device)
        optimizer.zero_grad()
        output, h_and_c = model(input, h_and_c)
        loss = criterion(output.reshape(-1,model.vocab_size), wanted_output.reshape(-1))
        loss.backward()
        optimizer.step()

        cur_perplexity = torch.exp(loss.detach())
        avg_perplexity += cur_perplexity
        if cur_iter % 100 == 0:
            print('iter = {}, perp = {}'.format(cur_iter, cur_perplexity))

        for g in optimizer.param_groups:
            for ipg, param_group in enumerate(g['params']):
                grad_norm[ipg,cur_iter] = float(torch.norm(param_group.view(-1)))/len(param_group.view(-1))
        # print('grad norms = ', ["{:.2f}".format(i) for i in grad_norm[:,cur_iter]])

    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()
    # accuracy = 100 * correct / total

    plot_grad_norms(grad_norm)

    avg_perplexity = avg_perplexity/num_of_batches
    return float(avg_perplexity)


def evaluate(model, test_data, args, criterion, device):
    model.eval()
    seq_len = args.seq_len
    num_seq_in_batch = test_data.shape[0]
    num_of_batches = int(np.floor(test_data.shape[1] / seq_len))

    h_and_c = []
    h_and_c.append((torch.zeros([1, num_seq_in_batch, model.emb_dim], device=device),
                    torch.zeros([1, num_seq_in_batch, model.emb_dim], device=device)))  # layer 0
    h_and_c.append((torch.zeros([1, num_seq_in_batch, model.emb_dim], device=device),
                    torch.zeros([1, num_seq_in_batch, model.emb_dim], device=device)))  # layer 1

    avg_perplexity=0
    with torch.no_grad():
        for i in range(0, num_of_batches * seq_len, seq_len):
            cur_iter = int(i / seq_len)
            input = test_data[:, i:i + seq_len]
            wanted_output = test_data[:, i + 1:i + seq_len + 1]
            input, wanted_output = input.to(device), wanted_output.to(device)
            output, h_and_c = model(input, h_and_c)
            loss = criterion(output.reshape(-1, model.vocab_size), wanted_output.reshape(-1))

            cur_perplexity = torch.exp(loss)
            avg_perplexity += cur_perplexity

            if i==seq_len: #FIXME DEBUG - output the first prediction
                wanted_output = wanted_output.cpu()
                decoded_wanted = data_handler.decode_seq(wanted_output)
                output = output.cpu()
                decoded_predicted = data_handler.decode_seq(np.argmax(output,2))
                for iseq in range(num_seq_in_batch):
                    print('seq {}: {} \ {}'.format(iseq, ' '.join(decoded_wanted[iseq,:]).replace('\n','<eos>'), decoded_predicted[iseq,-1].replace('\n','<eos>')))

        avg_perplexity = avg_perplexity/num_of_batches
        print('avg perplexity = {}'.format(avg_perplexity))

    return float(avg_perplexity)


def parse_args():
    parser = argparse.ArgumentParser(
        description='''Classify FashionMNIST using LeNet5.\nUsage examples:
    train - 'python3 Lenet5.py --phase train --epochs 50'
    test the model - 'python3 LeNet5.py --phase test --model-path model.pt' ''',
        formatter_class=argparse.RawTextHelpFormatter)
    # parser.add_argument('-i', '--input', type=str, default='./data', help='Directory for data dir')
    parser.add_argument('-o', '--output', type=str, default='./dump', help='Directory for output dir')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train/test/plot')
    # parser.add_argument('--batchnorm', action='store_true', default=False, help='Add batchnorm')
    parser.add_argument('--dropout_p', type=float, default=0, help='Add dropout, enter probability')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    # parser.add_argument('--weight_decay', type=float, default=0, help='Add weight decay')
    parser.add_argument('--epochs', type=int, default=13, help='Training number of epochs')
    # parser.add_argument('--model-path', type=str, default=None, help='Path to saved model, test only')
    parser.add_argument('--show', action='store_true', help='Show plots')
    # parser.add_argument('--seq_len', type=float, default=35, help='sequence length')
    parser.add_argument('--seq_len', type=float, default=20, help='sequence length')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # batch_size = 64

    data_handler = DataHandler()
    train_data = data_handler.get_data('train')
    val_data = data_handler.get_data('validation')
    test_data = data_handler.get_data('test')


    model = PTBRNN(vocab_size=data_handler.vocab_size, dropout_p=args.dropout_p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    num_epochs = args.epochs

    if args.phase == "train":
        train_perp_list = []
        val_perp_list = []
        results = []
        try:
            for epoch in range(num_epochs):
                print('\nstarting epoch {}'.format(epoch))
                if (args.dropout_p > epsilon and epoch > 7) or (args.dropout_p <= epsilon and epoch > 3):
                    for g in optimizer.param_groups:
                        g['lr'] *= 0.5
                    print('reduced lr by factor of 2')

                avg_train_perplexity = train(model, train_data, args, criterion, optimizer, device)

                # Evaluate the model on the validation set
                avg_val_perplexity = evaluate(model, val_data, args, criterion, device)

                print('Epoch {}: avg training perp = {:.3f}, avg validation perp = {:.3f}'.format(
                    epoch, avg_train_perplexity, avg_val_perplexity))

                train_perp_list.append(avg_train_perplexity)
                val_perp_list.append(avg_val_perplexity)

        except KeyboardInterrupt:
            print("Training stopped by keyboard interrupt")

        model_name = f'dropout_{args.dropout_p}'
        torch.save(model.state_dict(), f"{args.output}/{model_name}.pt")

        results.append({'scenario': model_name, 'train_acc': f'{train_perp_list[-1]:.3f}',
                        'test_acc': f'{val_perp_list[-1]:.3f}'})
        df = pd.DataFrame(results)
        df.to_csv(f'{args.output}/results.csv', mode='a', header=not os.path.isfile(f'{args.output}/results.csv'),
                  index='False')

        plt.plot(range(1, num_epochs + 1), train_perp_list, label='Train')
        plt.plot(range(1, num_epochs + 1), val_perp_list, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('average perplexity')
        plt.title(f'average perplexity vs. Epoch for {model_name}')
        plt.legend()
        if args.show:
            plt.show()
        else:
            plt.savefig(f"{args.output}/{model_name}.png")

    elif args.phase == "test":
        if args.model_path == None:
            print("Specify trained model path!")
        else:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            test_accuracy = evaluate(model, test_data, criterion, device)
            print(f"Test accuracy: {test_accuracy} for {args.model_path}")

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



