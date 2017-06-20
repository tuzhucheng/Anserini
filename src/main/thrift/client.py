import argparse
import sys
sys.path.append('gen-py')

from qa import QuestionAnswering

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


def main(question, answer):

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

    score = client.getScore(question, answer)
    print('Question: ', question)
    print('Answer: ', answer)
    print('Similarity Score: ', score)

    transport.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Anserini Thrift Client for QA')
    ap.add_argument('question', help='Question')
    ap.add_argument('answer', help='Answer')
    args = ap.parse_args()

    main(args.question, args.answer)
