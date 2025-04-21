import numpy as np
import pandas as pd
import argparse
import os
from gensim.models import FastText
import xgboost as xgb
from joblib import dump

from sklearn.model_selection import GridSearchCV


def main():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-tr", "--train", help="training data",required=True)
    ap.add_argument("-te", "--test", help="testing data",required=True)
    ap.add_argument("-e", "--embed", help="model embedding",required=True)
    ap.add_argument("-o", "--output", help="output location",required=True)
    ap.add_argument("-t", "--tag", help="file tag",required=True)
    
    args = vars(ap.parse_args())
    
    trainDataFileName = args["train"] if args["train"] else print("Please input your train data")
    testDataFileName = args["test"] if args["test"] else print("Please input your test data")
    FT_pretrain = args["embed"] if args["test"] else print("Please input your embedding model")
    
    file_tag = args["tag"] if args["tag"] else print("Please input your file tag")
    out_loc = args["output"] if args["output"] else print("Please input your output location")
    
    train_data =pd.read_csv(trainDataFileName)
    test_data =pd.read_csv(testDataFileName)
    FT_model = FastText.load(FT_pretrain)

    # x_train = train_data.iloc[:, 1].apply(lambda text: [FT_model.wv[word] for word in text.split() if word in FT_model.wv])
    # x_test = test_data.iloc[:, 1].apply(lambda text: [FT_model.wv[word] for word in text.split() if word in FT_model.wv])
    x_train = train_data.iloc[:, 1].apply(lambda text: [FT_model.wv.get_vector(word) for word in text.split()])
    x_test = test_data.iloc[:, 1].apply(lambda text: [FT_model.wv.get_vector(word) for word in text.split()])
    
    x_train = np.array(x_train.values.tolist())
    x_test = np.array(x_test.values.tolist())

    y_train = np.array(train_data.iloc[:, 0].tolist())
    y_test = np.array(test_data.iloc[:, 0].tolist())
    
    x_train = x_train.reshape(*x_train.shape[:-2], -1)
    x_test = x_test.reshape(*x_test.shape[:-2], -1)
    
    parameters = {"subsample":[0.75, 1],
             "colsample_bytree":[0.75, 1],
             "max_depth":[2, 6],"min_child_weight":[1, 5],
             "learning_rate":[0.05, 0.3]}
    
    model = xgb.XGBClassifier()

    grid = GridSearchCV(model, parameters, n_jobs=-1, 
                    scoring="roc_auc",
                    verbose=3, refit=True)

    grid.fit(x_train, y_train)
    
    estimator = grid.best_estimator_
    pred_Y = estimator.predict(x_test)
    
    outputFileName = 'XGB_FT' + file_tag
    DIR_ASSETS = out_loc + '/FastText_model/'
    
    RESULT_FILE = DIR_ASSETS + 'result'+file_tag+'.csv'
    LOG_FILE = DIR_ASSETS + 'result'+file_tag+'.txt'
    PATH_MODEL = DIR_ASSETS + outputFileName + '.joblib'
    
    if not os.path.isdir(DIR_ASSETS):
        os.mkdir(DIR_ASSETS)
    dump(estimator, PATH_MODEL)
    
    result_file = open(LOG_FILE, 'w+')
    result_file.write('train data file name:%s\n' % (trainDataFileName))
    result_file.write('test data file name:%s\n' % (testDataFileName))
    result_file.write('Best parameters:\n' + str(grid.best_params_))
    
    df = pd.DataFrame({'test_label':y_test.tolist(), 'predicted_label':pred_Y.tolist()})
    df.to_csv(RESULT_FILE, index=False)
    
    
    
if __name__ == "__main__":
    main()