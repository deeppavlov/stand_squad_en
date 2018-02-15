import os
import tensorflow as tf

from prepro import prepro
from main import train, test, predict, test_sber
from shutil import copyfile

flags = tf.flags

home = os.path.expanduser("~")
train_file = os.path.join(home, "data", "datasets", "sbersquad", "train-v1.1.json")
dev_file = os.path.join(home, "data", "datasets", "sbersquad", "dev-v1.1.json")
test_file = os.path.join(home, "data", "datasets", "sbersquad", "dev-v1.1.json")
predict_file = os.path.join(home, "data", "datasets", "sbersquad", "task_private.json_squad")
glove_word_file = os.path.join(home, "data", "glove", "glove.840B.300d.txt")

suff = "test_pos"
dir = 'log_{}/'.format(suff)
target_dir = "{}data".format(dir)
log_dir = "{}event".format(dir)
save_dir = "{}model".format(dir)
answer_dir = "{}answer".format(dir)

train_record_file = os.path.join(target_dir, "train.tfrecords")
dev_record_file = os.path.join(target_dir, "dev.tfrecords")
test_record_file = os.path.join(target_dir, "test.tfrecords")
predict_record_file = os.path.join(target_dir, "predict.tfrecords")
word_emb_file = os.path.join(target_dir, "word_emb.json")
char_emb_file = os.path.join(target_dir, "char_emb.json")
bpe_emb_file = os.path.join(target_dir, "bpe_emb.json")
pos_emb_file = os.path.join(target_dir, "pos_emb.json")
word2idx_dict_file = os.path.join(target_dir, "word2idx_dict.pckl")
char2idx_dict_file = os.path.join(target_dir, "char2idx_dict.pckl")
bpe_codes_file = os.path.join(target_dir, "bpe_codes.txt")
bpe2idx_dict_file = os.path.join(target_dir, "bpe2idx_dict.pckl")
pos2idx_dict_file = os.path.join(target_dir, "pos2idx_dict.pckl")
train_eval = os.path.join(target_dir, "train_eval.json")
dev_eval = os.path.join(target_dir, "dev_eval.json")
test_eval = os.path.join(target_dir, "test_eval.json")
predict_eval = os.path.join(target_dir, "predict_eval.json")
dev_meta = os.path.join(target_dir, "dev_meta.json")
test_meta = os.path.join(target_dir, "test_meta.json")
predict_meta = os.path.join(target_dir, "predict_meta.json")
answer_file = os.path.join(answer_dir, "answer.json")
predict_answer_file = os.path.join(answer_dir, os.path.basename(predict_file) + "_ans")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(answer_dir):
    os.makedirs(answer_dir)

flags.DEFINE_string("mode", "train", "Running mode train/debug/test/predict/test_sber")

flags.DEFINE_string("target_dir", target_dir, "Target directory for out data")
flags.DEFINE_string("log_dir", log_dir, "Directory for tf event")
flags.DEFINE_string("save_dir", save_dir, "Directory for saving model")
flags.DEFINE_string("answer_dir", answer_dir, "Directory for saving model")
flags.DEFINE_string("train_file", train_file, "Train source file")
flags.DEFINE_string("dev_file", dev_file, "Dev source file")
flags.DEFINE_string("test_file", test_file, "Test source file")
flags.DEFINE_string("predict_file", predict_file, "Predict source file")
flags.DEFINE_string("glove_word_file", glove_word_file, "Glove word embedding source file")

flags.DEFINE_string("train_record_file", train_record_file, "Out file for train data")
flags.DEFINE_string("dev_record_file", dev_record_file, "Out file for dev data")
flags.DEFINE_string("test_record_file", test_record_file, "Out file for test data")
flags.DEFINE_string("predict_record_file", predict_record_file, "Out file for predict data")
flags.DEFINE_string("word_emb_file", word_emb_file, "Out file for word embedding")
flags.DEFINE_string("word2idx_dict_file", word2idx_dict_file, "Out file for word vocab")
flags.DEFINE_string("char_emb_file", char_emb_file, "Out file for char embedding")
flags.DEFINE_string("char2idx_dict_file", char2idx_dict_file, "Out file for char vocab")
flags.DEFINE_string("bpe_codes_file", bpe_codes_file, "Out file for BPE codes")
flags.DEFINE_string("bpe2idx_dict_file", bpe2idx_dict_file, "Out file for bpe vocab")
flags.DEFINE_string("bpe_emb_file", bpe_emb_file, "Out file for bpe embedding")
flags.DEFINE_string("pos2idx_dict_file", pos2idx_dict_file, "Out file for POS vocab")
flags.DEFINE_string("pos_emb_file", pos_emb_file, "Out file for POS embedding")
flags.DEFINE_string("train_eval_file", train_eval, "Out file for train eval")
flags.DEFINE_string("dev_eval_file", dev_eval, "Out file for dev eval")
flags.DEFINE_string("test_eval_file", test_eval, "Out file for test eval")
flags.DEFINE_string("predict_eval_file", predict_eval, "Out file for predict eval")
flags.DEFINE_string("dev_meta", dev_meta, "Out file for dev meta")
flags.DEFINE_string("test_meta", test_meta, "Out file for test meta")
flags.DEFINE_string("predict_meta", predict_meta, "Out file for predict meta")
flags.DEFINE_string("answer_file", answer_file, "Out file for answer") # answer on test file
flags.DEFINE_string("predict_answer_file", predict_answer_file, "Out file for predict answer") # answer on predict file


flags.DEFINE_integer("glove_char_size", 1500, "Char corpus size for Glove")
flags.DEFINE_integer("glove_word_size", int(2.2e6), "Word corpus size for Glove")
flags.DEFINE_integer("glove_bpe_size", 15000, "BPE corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("bpe_glove_dim", 300, "BPE Embedding dimension for Glove")
flags.DEFINE_integer("char_dim", 8, "Embedding dimension for char")
flags.DEFINE_integer("bpe_dim", 50, "Embedding dimension for bpe")
flags.DEFINE_integer("bpe_merges_count", 5000, "BPE train merges count")
flags.DEFINE_integer("pos_dim", 8, "Embedding dimension for POS tags")

flags.DEFINE_integer("para_limit", 400, "Limit length for paragraph")
flags.DEFINE_integer("ques_limit", 50, "Limit length for question")
flags.DEFINE_integer("test_para_limit", 1000, "Limit length for paragraph in test file")
flags.DEFINE_integer("test_ques_limit", 100, "Limit length for question in test file")
flags.DEFINE_integer("char_limit", 16, "Limit length for character sequence")
flags.DEFINE_integer("bpe_limit", 8, "Limit length for bpe sequence")
flags.DEFINE_integer("word_count_limit", -1, "Min count for word")
flags.DEFINE_integer("char_count_limit", -1, "Min count for char")

flags.DEFINE_integer("capacity", 15000, "Batch size of dataset shuffle")
flags.DEFINE_integer("num_threads", 4, "Number of threads in input pipeline")
flags.DEFINE_boolean("use_cudnn", True, "Whether to use cudnn rnn (should be False for CPU)")
flags.DEFINE_boolean("is_bucket", False, "build bucket batch iterator or not")
flags.DEFINE_integer("bucket_range", [40, 401, 40], "the range of bucket")

flags.DEFINE_integer("batch_size", 40, "Batch size")
flags.DEFINE_integer("num_steps", 120000, "Number of steps")
flags.DEFINE_integer("checkpoint", 1000, "checkpoint to save and evaluate the model") # default 1000
flags.DEFINE_integer("period", 100, "period to save batch loss")
flags.DEFINE_integer("val_num_batches", 150, "Number of batches to evaluate the model")
flags.DEFINE_float("init_lr", 0.5, "Initial learning rate for Adadelta")
flags.DEFINE_float("min_lr", 1e-05, "min value for learning rate")
flags.DEFINE_float("keep_prob", 0.6, "Dropout keep prob in rnn") # default 0.7
flags.DEFINE_float("ptr_keep_prob", 0.6, "Dropout keep prob for pointer network") # default 0.7
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_integer("hidden", 75, "Hidden size")
flags.DEFINE_integer("char_hidden", 100, "GRU dimention for char") # default 100
flags.DEFINE_integer("bpe_hidden", 100, "GRU dimention for bpe")
flags.DEFINE_integer("patience", 3, "Patience for learning rate decay")

# Extensions (Uncomment corresponding code in download.sh to download the required data)
glove_char_file = os.path.join(home, "data", "datasets", "fasttext", "ft_native_300_ru_wiki_lenta_nltk_word_tokenize-char.vec")
flags.DEFINE_string("glove_char_file", glove_char_file, "Glove character embedding source file")
flags.DEFINE_boolean("pretrained_char", True, "Whether to use pretrained character embedding")
# set train_pretrained_char True if you don't use pretrained embds!
flags.DEFINE_boolean("train_pretrained_char", True, "Train pretrained character embedding or use unchanged")


fasttext_file = os.path.join(home, "data", "datasets", 'fasttext', "ft_native_300_ru_wiki_lenta_nltk_word_tokenize.vec")
flags.DEFINE_string("fasttext_file", fasttext_file, "Fasttext word embedding source file")
flags.DEFINE_boolean("fasttext", True, "Whether to use fasttext")

flags.DEFINE_boolean("remove_unicode", True, "Remove unicode symbols while preprocessing or not")
flags.DEFINE_boolean("use_char", True, "Use character embeddings")

flags.DEFINE_boolean("use_bpe", True, "Use BPE embeddings")
flags.DEFINE_boolean("use_bpe_pretrained_codes", True, "Use pretrained BPE codes")
flags.DEFINE_string("bpe_pretrained_codes_file",
                    os.path.join(home, 'data', 'datasets', 'wiki_news', 'bpe_codes_wiki_5000.txt'),
                    "Use pretrained BPE codes")
flags.DEFINE_string("glove_bpe_file",
                    os.path.join(home, 'data', 'datasets', 'fasttext', 'ft_native_300_ru_wiki_lenta_nltk_word_tokenize-bpe_5000.vec'),
                    "Glove bpe embedding source file")
flags.DEFINE_boolean("pretrained_bpe_emb", True, "Whether to use pretrained BPE embedding")
flags.DEFINE_boolean("train_pretrained_bpe_emb", True, "Train pretrained BPE embedding or use unchanged")

flags.DEFINE_integer("freeze_steps", 10000, "Number of training steps when pretrained weights are freezed")
flags.DEFINE_boolean("use_pos", True, "Use POS tags")

flags.DEFINE_string("sber_public_file",
                    os.path.join(home, 'data', 'datasets', 'sbersquad', 'task_public.json'),
                    "Sber public test path")

flags.DEFINE_string("sber_private_file",
                    os.path.join(home, 'data', 'datasets', 'sbersquad', 'task_private.json'),
                    "Sber private test path")

flags.DEFINE_string("model_name", "latest", "Model name to load or latest")


def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        train(config)
    elif config.mode == "prepro":
        copyfile(_[0], dir + 'config.py')
        prepro(config)
    elif config.mode == "debug":
        config.num_steps = 2
        config.val_num_batches = 1
        config.checkpoint = 1
        config.period = 1
        train(config)
    elif config.mode == "test":
        if config.use_cudnn:
            print("Warning: Due to a known bug in Tensorlfow, the parameters of CudnnGRU may not be properly restored.")
        test(config)
    elif config.mode == "predict":
        if config.use_cudnn:
            print("Warning: Due to a known bug in Tensorlfow, the parameters of CudnnGRU may not be properly restored.")
        predict(config)
    elif config.mode == "test_sber":
        if config.use_cudnn:
            print("Warning: Due to a known bug in Tensorlfow, the parameters of CudnnGRU may not be properly restored.")
        test_sber(config)
    else:
        print("Unknown mode")
        exit(0)


if __name__ == "__main__":
    tf.app.run()
