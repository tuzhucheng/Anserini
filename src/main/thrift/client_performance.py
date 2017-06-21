import argparse
import itertools
from multiprocessing.dummy import Pool as ThreadPool
import sys
import time
sys.path.append('gen-py')

import numpy as np
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

from qa import QuestionAnswering


def query(client, question, answer):
    query_start_time = time.time()
    client.getScore(question, answer)
    return time.time() - query_start_time


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

    latencies = []
    for _ in range(repeat):
        latencies.extend(pool.starmap(query, zip(itertools.repeat(client), questions, answers)))

    elapsed_time = time.time() - start_time

    latencies = np.array(latencies)
    print('Throughput (QPS): ', repeat * len(questions) / elapsed_time)
    print('Average Latency (ms): ', np.mean(latencies) * 1000)
    print('Latency p50 (ms): ', np.percentile(latencies, 50) * 1000)
    print('Latency p99 (ms): ', np.percentile(latencies, 99) * 1000)

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
