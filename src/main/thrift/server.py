import sys
sys.path.append('gen-py')

from qa import QuestionAnswering

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer


class QuestionAnsweringHandler:

    def __init__(self):
        pass

    def getScore(self, question, answer):
        return 3.14


if __name__ == '__main__':
    handler = QuestionAnsweringHandler()
    processor = QuestionAnswering.Processor(handler)
    transport = TSocket.TServerSocket(port=9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    server.serve()
