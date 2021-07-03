import argparse

from paddlenlp.transformers import BertTokenizer, BertForTokenClassification

from config import parse_argument


parser = argparse.ArgumentParser()
config = parse_argument(parser)


if __name__ == '__main__':
    print(config)
