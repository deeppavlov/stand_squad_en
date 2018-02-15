import numpy as np
import os
import unicodedata
from tqdm import tqdm
import argparse

def make_char_emb(infile):
    vectors = {}
    with open(infile, 'r') as fin:
        for line in tqdm(fin):
            line_split = line.strip().split(" ")
            vec = np.array(line_split[1:], dtype=float)
            word = line_split[0]
            word = ''.join(c for c in word if not unicodedata.combining(c))
            for char in word:
                if char in vectors:
                    vectors[char] = (vectors[char][0] + vec,
                                     vectors[char][1] + 1)
                else:
                    vectors[char] = (vec, 1)

    outfile = os.path.splitext(infile)[0] + '-char.vec'
    with open(outfile, 'w') as fout:
        for word in vectors:
            avg_vector = np.round(
                (vectors[word][0] / vectors[word][1]), 6).tolist()
            fout.write(word + " " + " ".join(str(x) for x in avg_vector) + "\n")

    print('Char embds saved to: {}'.format(outfile))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, help='path to word embeddings')
    args = parser.parse_args()
    make_char_emb(args.infile)