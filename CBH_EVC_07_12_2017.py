# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:08:46 2017

@author: dkashi200
"""

import sys # Load a library module
import csv # CSV module
import cx_Oracle
import pandas.io.sql as psql
from pandas import *
import numpy as np
import pandas
import matplotlib.pyplot as plt
import urllib
from io import StringIO
import time
import unicodedata
import codecs
import paramiko
import gzip
import zipfile
from datacon.database_ext import *

paramiko.util.log_to_file('/tmp/paramiko.log')

def Gzip_Reader(memberf, columns):
    f=gzip.open(memberf,'rb')
    file_content=f.read()
    return Str_to_df_Parser( file_content, columns)

def Str_to_df_Parser( string, col_list):
    respstr = str(string).strip("b'")
    count = 0
    for i in col_list:
        if count == 0:            
            colum =i
            count+=1
        else:
            colum+=','+i
    colum+='$'
    respstr = colum+ respstr.replace('\\n','$')
    #print(respstr)
    df_DExt=read_csv(StringIO(respstr), lineterminator='$')
    return df_DExt

col_list = ['SOAM_MAP_CLLI_PORT','COS', 'KI', 'MI', 'AGG', 'TS', 'VALUE']
my_file=r'C:\Users\dkashi200\Yawaris\Temp\CBH_EVCS_07_12_2017.csv'
df_list=read_csv(my_file,names = ['SOAM_MAP_CLLI_PORT','COS', 'KI', 'MI', 'AGG', 'TS', 'VALUE'])

df_list['SOAM_MA']=df_list['SOAM_MAP_CLLI_PORT'].apply(lambda x: x[:x.rfind('UNI')+3])
df_list['SOAM_MAID']=df_list['SOAM_MAP_CLLI_PORT'].apply(lambda x: x[x.rfind('UNI')+4:x.find('^')])
df_list['MEP_PAIR']=df_list['SOAM_MAP_CLLI_PORT'].apply(lambda x: x[x.find('^')+1:x.find(':',x.find('^'))])
df_list['CLLI']=df_list['SOAM_MAP_CLLI_PORT'].apply(lambda x: x[x.find(':',x.find('^'))+1:x.find('.',x.find('^'))])
df_list['Port']=df_list['SOAM_MAP_CLLI_PORT'].apply(lambda x: x[x.find('.',x.find('^'))+1:])
Number_of_misconf_EVC=len(df_list[df_list['SOAM_MAID'].str.rfind('V') < 0]['SOAM_MAID'].unique())*100/len(df_list['SOAM_MAID'].unique())
Number_of_misconf_conn=len(df_list[df_list['SOAM_MAID'].str.rfind('V') < 0]['SOAM_MAID'])*100/len(df_list['SOAM_MAID'])

print('Number_of_misconf_EVC :',Number_of_misconf_EVC)
print('Number_of_misconf_conn :',Number_of_misconf_conn)

EVC=[]
Serv = ['Ethernet Network Service', 'Ethernet Virtual Private Line', 'Ethernet Private Line','Ethernet Dedicated Internet Access']
CR = data_base('DHIRENK', 'dH1reN','clipdb-po-ap-sc.sys.comcast.net', 1555, 'CRMRPR0_EXT_RO_SVC', False)


############################Cramer EVC##########################################################################
CR.sql_file = r"C:\Users\dkashi200\Yawaris\python\cbh.sql"

df = query.Exp(CR)
df.index.name = 'S.No'

final=merge(df_list,df,how='inner',left_on='SOAM_MAID', right_on='EVCID')

fin=final[['EVCID','CUSTOMERNAME','LINEOFBUSINESS', 'MARKETID','DEVSITENAME']].drop_duplicates()

fin_=fin[['DEVSITENAME','CUSTOMERNAME']].drop_duplicates()

my_file=r'C:\Users\dkashi200\Yawaris\Temp\CBH_EVCS_07_12_2017_Final.csv'

fin_.to_csv(my_file)
