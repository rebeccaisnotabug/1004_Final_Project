#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py <student_netID>
'''
#Use getpass to obtain user netID
import getpass
import pandas as pd
import numpy as np


# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row




def main(spark, netID):
    train_path = 'hdfs:/user/bm106/pub/MSD/cf_train.parquet'
    validation_path = 'hdfs:/user/bm106/pub/MSD/cf_validation.parquet'
    
    train = spark.read.parquet(train_path)
    train.createOrReplaceTempView('train')
    validation = spark.read.parquet(validation_path)
    validation.createOrReplaceTempView('validation')
    df = train.sample(False, 0.01, 42)
    df.createOrReplaceTempView('df')
    
    query1 = spark.sql('SELECT count(DISTINCT user_ID) FROM df')
    query1.show()
    
    query2 = spark.sql('SELECT count(DISTINCT user_ID) FROM validation')
    query2.show()
    
    query3 = spark.sql('SELECT count(DISTINCT df.user_ID) FROM df join validation on df.user_id = validation.user_id')
    query3.show()
    
    print('More than 10% of training set users appear in the validation set: ' + str(query3/query2 >= 0.1))
    
    #dfs_list = []
    #dfs_list.append(train)
    #dfs_list.append(validation)
    #df = pd.concat(dfs_list,axis=0,sort=True)
        
    #df.repartition(100).write.parquet('hdfs:/user/xh2112/train_validation_dataset.parquet')

    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user netID from the command line
    netID = getpass.getuser()
    
    main(spark,netID)

    # Call our main routine