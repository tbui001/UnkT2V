import numpy as np
import os
import argparse
from time import time
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from nltk.tokenize import word_tokenize
import nltk

# class callback(CallbackAny2Vec):
#     '''Callback to print loss after each epoch.'''
#     def __init__(self):
#         self.epoch = 0
#         self.loss_to_be_subed = 0
#         self.time = time()
        

#     def on_epoch_end(self, model):
#         loss = model.get_latest_training_loss()
#         loss_now = loss - self.loss_to_be_subed
#         self.loss_to_be_subed = loss
#         log_file = "training_"+str(model.vector_size)+".log" 

#         if ((self.epoch + 1) % 10 == 0) and (self.epoch != 0):
#             t_time = time() - self.time
#             with open(log_file, "a") as text_file:
#                 text_file.write("Loss after epoch {}: {:15.2f}, training time: {:10.4f} s \n".format(self.epoch+1, loss_now, t_time))
#         self.epoch += 1

def preprocessing (file_content):
    corpus = []
    with open(file_content, 'r') as inputfile:
        for line in inputfile:
            tokens = word_tokenize(line)
            corpus.append(tokens)
    
    return corpus

def main():
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", help="Path to the input",required=True)
    ap.add_argument("-p", "--param", help="Input number of vector size",type=int,required=True)
    ap.add_argument("-o", "--output", help="Input number of vector size",required=True)
    
    args = vars(ap.parse_args())
    
    file_content = args["input"] if args["input"] else print("Please add the file location")
    vector_size = args["param"] if args["param"] else print("Please define the parameters")
    outfile = args["output"] if args["output"] else print("Please define the outfile location")
    
    corpus = preprocessing (file_content)
    model = FastText(corpus, min_count = 5, vector_size = vector_size, min_n = 2, max_n = 2, window = 2, sg = 1, workers=12, epochs=500)

    os.makedirs(outfile, exist_ok=True)
    outfile = outfile + "Attr2Vec"+str(vector_size)+".model"
    model.save(outfile)


if __name__ == "__main__":
    main()


