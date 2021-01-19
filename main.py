import sys

import numpy
from sentence_transformers import SentenceTransformer
import torch
from matplotlib import pyplot

def analyze():

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    model = SentenceTransformer("LaBSE")
    file1_sentences = open(file1, "r").readlines()
    file2_sentences = open(file2, "r").readlines()
    file1_embeddings = model.encode(file1_sentences, show_progress_bar=True)
    file2_embeddings = model.encode(file2_sentences, show_progress_bar=True)
    file1_torch_embeddings = torch.from_numpy(file1_embeddings)
    file2_torch_embeddings = torch.from_numpy(file2_embeddings)

    cos = torch.nn.CosineSimilarity()
    output = cos(file1_torch_embeddings, file2_torch_embeddings)
    output = output.cpu().detach().numpy()
    pyplot.hist(output, bins=40, color="y")
    mean = numpy.mean(output)
    std = numpy.std(output)
    left_std = mean - std
    right_std = mean + std
    pyplot.axvline(mean, linestyle='dashed', linewidth=1)
    pyplot.axvline(left_std, linestyle='dashed', linewidth=1, color="r")
    pyplot.axvline(right_std, linestyle='dashed', linewidth=1, color="r")
    min_ylim, max_ylim = pyplot.ylim()
    pyplot.text(mean, max_ylim, mean)
    pyplot.text(left_std, max_ylim, left_std)
    pyplot.text(right_std, max_ylim, right_std)
    pyplot.show()

if __name__ == '__main__':
    analyze()