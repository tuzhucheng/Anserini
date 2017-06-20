import argparse
import itertools
from multiprocessing.dummy import Pool as ThreadPool
import sys
import time
sys.path.append('gen-py')

from qa import QuestionAnswering

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


def query(client, question, answer):
    return client.getScore(question, answer)


def main(questions, answers, repeat, threads):

    # Make socket
    transport = TSocket.TSocket('localhost', 9090)

    # Buffering is critical. Raw sockets are very slow
    transport = TTransport.TBufferedTransport(transport)

    # Wrap in a protocol
    protocol = TBinaryProtocol.TBinaryProtocol(transport)

    # Create a client to use the protocol encoder
    client = QuestionAnswering.Client(protocol)

    # Connect!
    transport.open()

    pool = ThreadPool(threads)

    start_time = time.time()

    for _ in range(repeat):
        pool.starmap(query, zip(itertools.repeat(client), questions, answers))

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('Throughput (QPS): ', repeat * len(questions) / elapsed_time)

    transport.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Anserini Thrift Client for QA Performance Testing')
    ap.add_argument('dataset', help='Dataset directory')
    ap.add_argument('repeat', type=int, help='Number of times to repeat querying the score for the dataset')
    ap.add_argument('threads', type=int, help='Number of threads to use to send queries')
    args = ap.parse_args()

    questions, answers = [], []

    with open(args.dataset + '/a.toks', 'r') as f:
        for l in f:
            questions.append(l.rstrip())

    with open(args.dataset + '/b.toks', 'r') as f:
        for l in f:
            answers.append(l.rstrip())

    main(questions, answers, args.repeat, args.threads)
