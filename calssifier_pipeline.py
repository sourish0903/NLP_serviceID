##Packages

from __future__ import division
import pandas as pd
import numpy as np
import math
import subprocess
import os
import io
from gensim.models import FastText,word2vec
# from pyspark import SparkContext
# from pyspark.sql import HiveContext
import gcsfs
# sc = SparkContext()
# sc.setLogLevel("ERROR")
# spark = HiveContext(sc)
import time
import gc
from gensim.test.utils import get_tmpfile
print('Library Import')
import gcsfs
gcsfs.GCSFileSystem(project='my_project')
import pyspark
import pyspark.sql.functions as F
import sqlalchemy
import getpass
import pyspark.sql.window as Window
from pyspark.sql.functions import udf
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from functools import reduce
from pyspark.sql import DataFrame
import os
from hyperopt import STATUS_OK
from timeit import default_timer as timer
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
import numpy as np 
from hyperopt import tpe,Trials,fmin
import pandas as pd
import pyarrow.parquet as pq
from fastparquet import ParquetFile
import pyspark.ml as ml
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.window import Window
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel, GBTClassifier, GBTClassificationModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd
from sklearn.metrics import roc_auc_score
from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
import sklearn.neighbors
import pandas as pd
from scipy.stats import chi2_contingency
from pyspark.sql.types import IntegerType,StringType,FloatType
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    ParameterGrid,
    ParameterSampler,
    RandomizedSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split)
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    make_scorer,
    roc_auc_score,
)
import xgboost as xgb
from ctypes import *
from sklearn import datasets, metrics, model_selection
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV 
from xgboost.sklearn import XGBClassifier,XGBRegressor
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import sklearn.preprocessing as pr
from sklearn.preprocessing import LabelEncoder
import pickle
mode = 'overwrite'
from tensorflow.python.keras import backend as K
import subprocess
from gensim.models import FastText,word2vec
from gensim.models.keyedvectors import FastTextKeyedVectors, KeyedVectors
import gcsfs
import nltk
# nltk.download('punkt')
import seaborn as sns
import re
import string
from nltk.corpus import stopwords
import gensim
from gensim.models import FastText,word2vec
import seaborn as sns
from sklearn.decomposition import PCA
# fast_text_mod = FastText.load('/tmp/fasttext.model/')
# fast_text_mod = KeyedVectors.load('gs://meijer-analystplatform-analyst/raziah/subs/productionModel/science/subs_model_fasttext_2018_sp.model')

selected_fields = ['id', 'title', 'description','user_description','serviceId','createdAt','target_date','tradeClassificationType','sbf_form_text']
nlp_field = ['title', 'description']
target_col_orig = 'serviceId'
data_path = 'nlp_dat_classif.csv'
model_path = mod_path
language = 'german'
nlp_recipe = gensim.models.Word2Vec
classifier = xgb.XGBClassifier


#Functions
class nlp_classif:
    def __init__(self, cv=3,data_split = 0.15,target_col_orig = target_col_orig,data_path = data_path,language=language,\
                 nlp_recipe = nlp_recipe,seed=100,classifier = xgb.XGBClassifier):
        self.cv = cv
        self.target_col_orig = target_col_orig
        self.data_path = data_path
        self.language = language
        self.nlp_recipe = nlp_recipe
        self.data_split = data_split
        self.seed = seed
        self.classifier=classifier
    def data_preprocess(self):
        data = pd.read_csv(data_path,encoding='ISO-8859-1')
        data_useful = data[selected_fields]
        data_useful = data_useful.replace(np.nan, '', regex=True)
        data_useful.createdAt = pd.to_datetime(data_useful.createdAt)
        l = []
        for i in range(len(data_useful)):
            if data_useful.target_date[i].split(':')[0] == 'Wunschtermin':
                d = (pd.to_datetime(data_useful.target_date[i].split(':')[1].strip())-pd.to_datetime(data_useful.createdAt[i])).days 
                s = str(np.where(d<30,'Innerhalb der n?§chsten 30 Tage', np.where(30<d<=90,'In den n?§chsten 3 Monaten','In 3 bis 6 Monaten')))
                l.append(s)
            else:
                l.append(data_useful.target_date[i])
        data_useful['target_date_modified'] = pd.Series(l)
        target = np.sort(data_useful['serviceId'].unique())
        dic = {}
        for i,j in enumerate(list(target)):
            dic[j] = i
        data_useful['serviceId_coded'] = [dic[i] for i in data_useful[self.target_col_orig].values] 
        data_useful['serviceId_coded'] = data_useful['serviceId_coded'].astype('category')
        return data_useful 
    def nlp_processor(self,data_useful):
        data_useful['composite_text'] = data['title'] + data['description'].map(str)
        data_useful['composite_text'] = [re.sub("@[\w]*",'',str(data_useful.composite_text.values[i])).strip() for i in range(len(data_useful))] 
        data_useful['composite_text'] = [re.sub("\d",'',str(data_useful.composite_text.values[i])).strip() for i in range(len(data_useful))] 
        data_useful['composite_text'] = [str(data_useful.composite_text.values[i]).lower().strip() for i in range(len(data_useful))] 
        table = str.maketrans('', '', string.punctuation)
        data_useful['clean_txt'] = data_useful['composite_text'].apply(lambda x: [w.translate(table) for w in x.split(' ') if w not in('.','')])
        # stop_words = stopwords.words(self.language)
        data_useful['clean_txt'] = data_useful['clean_txt'].apply(lambda x: [w for w in x if w not in(stop_words)])
        mod = self.nlp_recipe(data_useful['clean_txt'],size = 100,min_count=1)
        w2_v_feat = pd.DataFrame(columns = range(100))
        for i in range(len(data_useful)):
            w2_v_feat = w2_v_feat.append(pd.Series(np.mean(np.array([mod.wv.get_vector(w) for w in data_useful['clean_txt'][i]]),axis=0)),ignore_index=True)
        w2_v_feat.columns = ['emb_%s' %(c) for c in range(1,101)]
        emb_normalized = sklearn.preprocessing.normalize(w2_v_feat, norm='l2')
        emb_normalized = pd.DataFrame(emb_normalized)
        emb_normalized.columns = ['emb_%s' %(c) for c in range(1,101)]
        return data_useful,emb_normalized
    def final_mod_data_prep(self,data_useful,emb_normalized):
        data_useful['tradeClassificationType'] = data_useful['tradeClassificationType'].astype('category')
        dummy = pd.get_dummies(data_useful[['target_date_modified','tradeClassificationType']])
        mod_data = pd.concat([dummy,emb_normalized,data_useful['serviceId_coded']], axis=1)
        train_X,validation_X,train_Y,validation_Y = train_test_split(mod_data.drop('serviceId_coded',axis=1),mod_data['serviceId_coded'],test_size = self.data_split, random_state = self.seed)
        print(train_X.shape,validation_X.shape,train_Y.shape,validation_Y.shape)
        return train_X,validation_X,train_Y,validation_Y
     def cv_fine_tuned_mod(self,train_X,validation_X,train_Y,validation_Y):
        ind_params = {'n_estimators': 300, 'seed':100,
             'objective': 'multi:softprob'}
        ind_params['nthread'] = 4
        ind_params['eval_metric'] = 'merror'
        ind_params['num_class'] = data_useful['serviceId_coded'].nunique()
        xgb_model = self.classifier(ind_params,seed=100)
        clf = GridSearchCV(xgb_model,
                   {'max_depth': [6,7,8],'subsample': [0.7,0.8], 'colsample_bytree': [0.7,0.8,0.9],'min_child_weight': [1,5],'learning_rate': [0.04],
                    }, verbose=2,cv=self.cv,scoring='accuracy')
        clf.fit(train_X, train_Y)
        print(clf.best_params_)
        print(clf.best_score_)
        bst = clf.best_estimator_
        eval_set  = [(train_X,train_Y), (validation_X,validation_Y)]
        mod_bst = bst.fit(train_X, train_Y,eval_set=eval_set,verbose = 10,early_stopping_rounds=80,
        eval_metric="merror")
        pred_y = mod_bst.predict(validation_X)
        train_id,validation_id = train_test_split(data[['id']],test_size = 0.15, random_state = 20)
        validation_dat = pd.DataFrame({'id':validation_id.values,'actual':validation_Y,'prediction':pred_y})
        inv_dic = {v:k  for k,v in dic.items()}  
        validation_dat['actual'] = [inv_dic[i] for i in validation_dat['actual'].values] 
        validation_dat['prediction'] = [inv_dic[i] for i in validation_dat['prediction'].values] 
        validation_dat.to_csv('valid_actual_pred.csv',index=False)
        print(metrics.confusion_matrix(validation_Y,pred_y))
        print(metrics.classification_report(validation_Y, pred_y, digits=3))
        return mod_bst
		
#Main executable
if __name__ == '__main__':
    data = pd.read_csv(data_path,encoding='ISO-8859-1')
    nc = nlp_classif()
    data_useful = nc.data_preprocess()
    data_useful,emb_feat = nc.nlp_processor(data_useful)
    train_X,validation_X,train_Y,validation_Y = nc.final_mod_data_prep(data_useful,emb_feat)
    mod = nc.cv_fine_tuned_mod(train_X,validation_X,train_Y,validation_Y)
    with open(os.path.join(mod_path,'serviceid_classif.pickle'), 'wb') as f:
        pickle.dump(clf, f)



#Appendix- Stopwords
stop_words = ['aber',
 'alle',
 'allem',
 'allen',
 'aller',
 'alles',
 'als',
 'also',
 'am',
 'an',
 'ander',
 'andere',
 'anderem',
 'anderen',
 'anderer',
 'anderes',
 'anderm',
 'andern',
 'anderr',
 'anders',
 'auch',
 'auf',
 'aus',
 'bei',
 'bin',
 'bis',
 'bist',
 'da',
 'damit',
 'dann',
 'der',
 'den',
 'des',
 'dem',
 'die',
 'das',
 'dass',
 'daß',
 'derselbe',
 'derselben',
 'denselben',
 'desselben',
 'demselben',
 'dieselbe',
 'dieselben',
 'dasselbe',
 'dazu',
 'dein',
 'deine',
 'deinem',
 'deinen',
 'deiner',
 'deines',
 'denn',
 'derer',
 'dessen',
 'dich',
 'dir',
 'du',
 'dies',
 'diese',
 'diesem',
 'diesen',
 'dieser',
 'dieses',
 'doch',
 'dort',
 'durch',
 'ein',
 'eine',
 'einem',
 'einen',
 'einer',
 'eines',
 'einig',
 'einige',
 'einigem',
 'einigen',
 'einiger',
 'einiges',
 'einmal',
 'er',
 'ihn',
 'ihm',
 'es',
 'etwas',
 'euer',
 'eure',
 'eurem',
 'euren',
 'eurer',
 'eures',
 'für',
 'gegen',
 'gewesen',
 'hab',
 'habe',
 'haben',
 'hat',
 'hatte',
 'hatten',
 'hier',
 'hin',
 'hinter',
 'ich',
 'mich',
 'mir',
 'ihr',
 'ihre',
 'ihrem',
 'ihren',
 'ihrer',
 'ihres',
 'euch',
 'im',
 'in',
 'indem',
 'ins',
 'ist',
 'jede',
 'jedem',
 'jeden',
 'jeder',
 'jedes',
 'jene',
 'jenem',
 'jenen',
 'jener',
 'jenes',
 'jetzt',
 'kann',
 'kein',
 'keine',
 'keinem',
 'keinen',
 'keiner',
 'keines',
 'können',
 'könnte',
 'machen',
 'man',
 'manche',
 'manchem',
 'manchen',
 'mancher',
 'manches',
 'mein',
 'meine',
 'meinem',
 'meinen',
 'meiner',
 'meines',
 'mit',
 'muss',
 'musste',
 'nach',
 'nicht',
 'nichts',
 'noch',
 'nun',
 'nur',
 'ob',
 'oder',
 'ohne',
 'sehr',
 'sein',
 'seine',
 'seinem',
 'seinen',
 'seiner',
 'seines',
 'selbst',
 'sich',
 'sie',
 'ihnen',
 'sind',
 'so',
 'solche',
 'solchem',
 'solchen',
 'solcher',
 'solches',
 'soll',
 'sollte',
 'sondern',
 'sonst',
 'über',
 'um',
 'und',
 'uns',
 'unsere',
 'unserem',
 'unseren',
 'unser',
 'unseres',
 'unter',
 'viel',
 'vom',
 'von',
 'vor',
 'während',
 'war',
 'waren',
 'warst',
 'was',
 'weg',
 'weil',
 'weiter',
 'welche',
 'welchem',
 'welchen',
 'welcher',
 'welches',
 'wenn',
 'werde',
 'werden',
 'wie',
 'wieder',
 'will',
 'wir',
 'wird',
 'wirst',
 'wo',
 'wollen',
 'wollte',
 'würde',
 'würden',
 'zu',
 'zum',
 'zur',
 'zwar',
 'zwischen']