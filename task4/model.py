import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm
from data_loader import Dictionary, Corpus
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
embed_size = 128
hidden_size = 256
num_layers = 1
num_epochs = 50
num_samples = 500     # number of words to be sampled
batch_size = 20
seq_length = 30
learning_rate = 0.002

# Load "Penn Treebank" dataset
corpus = Corpus()
ids = corpus.get_data('../../LDC/train/train.shuf.en', batch_size)
# ids = corpus.get_data('silver', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // batch_size



# RNN based language model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        


        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)

        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))

        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)


def build_model():
    return RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)

def detach(states):
    return [state.detach() for state in states] 

def train(model, name, log_path='log', training=True):
    global_step = 0
    model_cnt = 0
    model_max = 5

    if not os.path.isdir(name + '_result'):
        os.mkdir(name + '_result')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                torch.zeros(num_layers, batch_size, hidden_size).to(device))

        for i in range(0, ids.size(1) - seq_length, seq_length):
            inputs = ids[:, i:i+seq_length].to(device)
            targets = ids[:, (i+1):(i+1)+seq_length].to(device)

            states = detach(states)
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape(-1))

            if training:
                model.zero_grad()
                loss.backward()
                clip_grad_norm(model.parameters(), 0.5)
                optimizer.step()

            global_step += 1 
            if global_step % 10 == 0:
                print ('Epoch: {}  Loss: {:.4f}   Perplexity: {:5.2f}'
                    .format(epoch+1, loss.item(), np.exp(loss.item())))
            
            if training:
                if global_step % 5 == 0:
                    with open(name+'_result/'+log_path, 'a') as f:
                        f.write('Epoch: {}, step: {}, Loss: {:.4f}, Perplexity: {:5.2f}\n'
                        .format(epoch+1, global_step, loss.item(), np.exp(loss.item())))
                



                if global_step % 100 == 0:
                    torch.save(model.state_dict(), name+'_result/model-%d.ckpt'%(model_cnt % model_max + 1))
                    model_cnt += 1


def test(model):
    with torch.no_grad():
        with open('sample.txt', 'w') as f:
            # Set intial hidden ane cell states
            state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                    torch.zeros(num_layers, 1, hidden_size).to(device))

            # Select one word id randomly
            prob = torch.ones(vocab_size)
            input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)
            print('input', input)
            # w_id = corpus.dictionary.word2idx['sherlock']
            # input = torch.LongTensor([[w_id]]).to(device)
            print('input', input)

            # word = corpus.dictionary.idx2word[w_id]
            # word = '\n' if word == '<eos>' else word + ' '
            # f.write(word)

            for i in range(num_samples):
                output, state = model(input, state)

                prob = output.exp()
                word_id = torch.multinomial(prob, num_samples=1).item()

                input.fill_(word_id)
                # print(input)
                word = corpus.dictionary.idx2word[word_id]
                word = '\n' if word == '<eos>' else word + ' '
                f.write(word)

                if (i+1) % 100 == 0:
                    print('Sampled [{}/{}] words and save to {}'.format(i+1, num_samples, 'sample.txt'))

# Save the model checkpoints

def predict(model, words, max_samples=200):
    with torch.no_grad():
        
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                torch.zeros(num_layers, 1, hidden_size).to(device))

        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

       
        output_sen = []
        words = words.split(' ')
        for word in words:
            if word in corpus.dictionary.word2idx:
                w_id = corpus.dictionary.word2idx[word]
                input = torch.LongTensor([[w_id]]).to(device)
                output, state = model(input, state)
            output_sen.append(word)
        
        for i in range(max_samples):
            output, state = model(input, state)

            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            input.fill_(word_id)
            # print(input)
            word = corpus.dictionary.idx2word[word_id]
            if word == '<eos>':
                break
            else:
                output_sen.append(word)

        print(' '.join(output_sen))


def test_perp(model, test_path):
    
    with open(test_path, 'r') as f:
        test_data = f.readlines()
    criterion = nn.CrossEntropyLoss()

    list_ids = []
    tot_perp = 0
    tot_loss = 0
    tot_tokens = 0
    for line in test_data:
        words = line.split() + ['<eos>']
        for word in words:
            if word.lower() in corpus.dictionary.word2idx:
                
                list_ids.append(corpus.dictionary.word2idx[word.lower()])

    test_ids = torch.LongTensor(list_ids).to(device)
    test_num_batches = test_ids.size(0) // batch_size
    test_ids = test_ids[:test_num_batches*batch_size]
    test_ids = test_ids.view(batch_size, -1)



    with torch.no_grad():
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                torch.zeros(num_layers, batch_size, hidden_size).to(device))
        for i in range(0, test_ids.size(1) - seq_length, seq_length):
            inputs = test_ids[:, i:i+seq_length].to(device)
            targets = test_ids[:, (i+1):(i+1)+seq_length].to(device)
        
            output, states = model(inputs, states)

            loss = criterion(output, targets.reshape(-1))
            print('loss = ', loss.item(), 'perp = ', np.exp(loss.item()))
            tot_loss += loss.item() * inputs.size(1) * inputs.size(0)
            tot_perp += np.exp(loss.item()) * inputs.size(1) * inputs.size(0)
            tot_tokens += inputs.size(1) * inputs.size(0)
        
    tot_loss = tot_loss / tot_tokens
    tot_perp = tot_perp / tot_tokens
    
    with open('LDC_result/test_log', 'a') as f: 
        f.write('Name: {}, Loss: {:.4f}, Perplexity: {:5.2f}, AvgPerplexity: {:5.2f}\n'
                        .format(test_path.split('/')[-1], tot_loss, np.exp(tot_loss), tot_perp))
