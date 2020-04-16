import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from model import RNNLM

device = 'gpu'



def normalize_sizes(y_pred, y_true):
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true


# 定义计算的准确率
def compute_accuracy(y_pred, y_true, mask_index=0):
    y_pred, y_true = normalize_sizes(y_pred, y_true)

    _, y_pred_indices = y_pred.max(dim=1)

    correct_indices = torch.eq(y_pred_indices, y_true).float()
    valid_indices = torch.ne(y_true, mask_index).float()

    n_correct = (correct_indices * valid_indices).sum().item()
    n_valid = valid_indices.sum().item()

    return n_correct / n_valid * 100


# 定义序列的损失函数
def sequence_loss(y_pred, y_true, mask_index=0):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)


def train():
    lang_dataset, dataloader, input_sents, output_sents = \
        get_dataset_dataloader(path, config.batch_size)
    vocab_size = len(lang_dataset.vocab)

    model = RNNLM(
        vocab_size,
        5,
        256
    )

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_loss = []
    train_acc = []
    # initialize the loss
    best_loss = 9999999.0
    for epoch in range(config.num_epochs):
        # 初始化 hidden state
        states = (Variable(torch.zeros(config.num_layers, config.batch_size, config.hidden_size)).to(device),
                  Variable(torch.zeros(config.num_layers, config.batch_size, config.hidden_size)).to(device))

        running_loss = 0.0
        running_acc = 0.0
        model.train()
        batch_index = 0
        for data_dict in tqdm(dataloader):
            batch_index += 1
            optimizer.zero_grad()
            x = data_dict['x'].to(device)
            y = data_dict['y'].to(device)
            y_pred, states = model(x, states)
            loss = sequence_loss(y_pred, y)
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += (loss.item() - running_loss) / batch_index
            acc_t = compute_accuracy(y_pred, y)
            running_acc += (acc_t - running_acc) / (batch_index + 1)
        print('Epoch = %d, Train loss = %f, Train accuracy = %f, Train perplexity = %f' % (
            epoch, running_loss, running_acc, math.exp(running_loss)))
        train_loss.append(running_loss)
        train_acc.append(running_acc)
        if running_loss < best_loss:
            # 模型保存
            torch.save(model, './model_save/best_model_epoch%d_loss_%f.pth' % (epoch, loss))
            best_loss = running_loss
        print(' '.join(generate(model, lang_dataset, 'the')))

    return train_loss, train_acc