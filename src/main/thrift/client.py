import sys
sys.path.append('gen-py')

from qa import QuestionAnswering

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


def main():

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

    score = client.getScore('My Question', 'My Answer')
    print('Score: ', score)

    transport.close()


if __name__ == '__main__':
    main()
