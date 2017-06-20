import argparse
import os
import sys

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
import numpy as np
import torch
from torch.autograd import Variable

sys.path.append('gen-py')
sys.path.append('src/sm-cnn/sm_cnn')
from qa import QuestionAnswering
from sm_cnn.model import QAModel
from sm_cnn.train import Trainer
import sm_cnn.utils as utils


class QuestionAnsweringHandler:

    def __init__(self, model_path):
        # cache word embeddings
        cache_file = os.path.splitext(args.word_vectors_file)[0] + '.cache'
        utils.cache_word_embeddings(args.word_vectors_file, cache_file)

        self.vocab_size, self.vec_dim = utils.load_embedding_dimensions(cache_file)

        self.model = QAModel.load(model_path)
        # self.model.no_ext_feats = True

        # Load embeddings
        with open(cache_file + '.vocab') as f:
            w2v_vocab_list = map(str.strip, f.readlines())

        self.w2v_dict = {}
        w2v_memmap = np.memmap(cache_file, dtype=np.double, shape=(self.vocab_size, self.vec_dim))
        vocab_to_idx = {w: i for i, w in enumerate(w2v_vocab_list)}
        for word in vocab_to_idx:
            self.w2v_dict[word] = w2v_memmap[vocab_to_idx[word]]

        self.unk_term = torch.from_numpy(np.random.uniform(-0.25, 0.25, self.vec_dim))
        self.default_ext_feats = Variable(torch.FloatTensor(1, 4))
        self.default_ext_feats[0, :] = torch.FloatTensor([0, 0, 0, 0])

    def __make_input_matrix(self, sentence):
        terms = list(map(lambda t: t.lower(), sentence.strip().split()[:60]))
        input_tensor = torch.zeros(1, self.vec_dim, len(terms)).type(torch.FloatTensor)
        for i, word in enumerate(terms):
            if word in self.w2v_dict:
                input_tensor[0, :, i] = torch.from_numpy(self.w2v_dict[word])
            else:
                input_tensor[0, :, i] = self.unk_term

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        return Variable(input_tensor)

    def getScore(self, question, answer):
        xq = self.__make_input_matrix(question)
        xa = self.__make_input_matrix(answer)

        pred = self.model(xq, xa, self.default_ext_feats)
        pred = torch.exp(pred)
        return pred.data[0, 1]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Anserini Thrift Server')
    ap.add_argument('model_path', help='Path to saved model')
    ap.add_argument('--word_vectors_file',
        help='NOTE: a cache will be created for faster loading for word vectors',
        default="../../../../data/word2vec/aquaint+wiki.txt.gz.ndim=50.bin")
    args = ap.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    handler = QuestionAnsweringHandler(args.model_path)
    processor = QuestionAnswering.Processor(handler)
    transport = TSocket.TServerSocket(port=9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    print('Server is ready!')
    server.serve()
