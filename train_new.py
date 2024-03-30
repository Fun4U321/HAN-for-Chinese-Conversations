"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import shutil
from src.HAN_new import HAN
from src.dataset_new import MyDataset
from src.utils import get_max_lengths, get_evaluation, my_collate_fn


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--word_hidden_size", type=int, default=50)
    parser.add_argument("--sent_hidden_size", type=int, default=50)
    parser.add_argument("--word_embed_dim", type=int, default=100)
    parser.add_argument("--char_embed_dim", type=int, default=100)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="data/train.csv")
    parser.add_argument("--test_set", type=str, default="data/test.csv")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default="data/glove.6B.100d.txt")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--pretrained_word2vec_path", type=str, default=None)  # 如果有的话，也添加这行
    args = parser.parse_args()
    return args

def train(opt, test_generator=None):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        print("ok")
    else:
        torch.manual_seed(123)
    output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}

    max_word_length, max_char_length = get_max_lengths(opt.train_set)
    print(max_word_length)
    print(max_char_length)
    training_set = MyDataset(opt.train_set, stop_words_path='data/stop_words.txt', max_length_word=max_word_length,
                             max_length_character=max_char_length)
    training_generator = DataLoader(training_set, **training_params, collate_fn=my_collate_fn)
    test_set = MyDataset(opt.test_set, opt.word2vec_path, max_length_word=max_word_length, max_length_character=max_char_length)  # 修改为适当的参数值
    test_generator = DataLoader(test_set, **test_params)

    model = HAN(opt.word_hidden_size, opt.sent_hidden_size, training_set.vocab_size, opt.word_embed_dim,
                       opt.char_embed_dim, max_word_length, training_set.num_classes, opt.pretrained_word2vec_path)


    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    # writer.add_graph(model, torch.zeros(opt.batch_size, max_sent_length, max_word_length))

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum)
    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epoches):
        for iter, (word_inputs, char_inputs, labels) in enumerate(training_generator):
            if torch.cuda.is_available():
                word_inputs = word_inputs.cuda()
                char_inputs = char_inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            predictions = model(word_inputs, char_inputs)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            training_metrics = get_evaluation(labels.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            label_ls = []
            pred_ls = []
            for word_inputs, char_inputs, labels in test_generator:
                if torch.cuda.is_available():
                    word_inputs = word_inputs.cuda()
                    char_inputs = char_inputs.cuda()
                    labels = labels.cuda()
                with torch.no_grad():
                    predictions = model(word_inputs, char_inputs)
                loss = criterion(predictions, labels)
                loss_ls.append(loss * len(labels))
                label_ls.extend(labels.clone().cpu())
                pred_ls.append(predictions.clone().cpu())
            test_loss = sum(loss_ls) / len(test_set)
            predictions = torch.cat(pred_ls, 0)
            labels = np.array(label_ls)
            test_metrics = get_evaluation(labels, predictions.numpy(), list_metrics=["accuracy", "confusion_matrix"])
            output_file.write(
                "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                    epoch + 1, opt.num_epoches,
                    test_loss,
                    test_metrics["accuracy"],
                    test_metrics["confusion_matrix"]))
            print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                test_loss, test_metrics["accuracy"]))
            writer.add_scalar('Test/Loss', test_loss, epoch)
            writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
            model.train()
            if test_loss + opt.es_min_delta < best_loss:
                best_loss = test_loss
                best_epoch = epoch
                torch.save(model, opt.saved_path + os.sep + "whole_model_han")

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, test_loss))
                break

if __name__ == "__main__":
    opt = get_args()
    train(opt)
