from util import sber2squad

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sberfile', type=str, help='path to sber json file')
    args = parser.parse_args()

    sber2squad(args.sberfile)
