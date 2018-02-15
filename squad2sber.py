from util import squad_answer2sber

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sberfile', type=str, help='path to sber json file')
    parser.add_argument('--answer', type=str, help='path to model answer file')
    args = parser.parse_args()

    squad_answer2sber(args.sberfile, args.answer)
