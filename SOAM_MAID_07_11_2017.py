# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:19:07 2016

@author: dkashi001c

Extract CBH SOAM data from files Files
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
import datetime
import pickle




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

def dataframe_expand(dframe):
    df_temp1=DataFrame(columns=['Host Name', 'Device IP','Device Model', 'Configuration Text','mep_count'])
    for p in range(int(dframe['mep_count'])+1):
        df_temp1=df_temp1.append(DataFrame(data=[[dframe['Host Name'],dframe['Device IP'],dframe['Device Model'],dframe['Configuration Text'].split('#')[p],dframe['mep_count']]],columns=['Host Name', 'Device IP','Device Model', 'Configuration Text','mep_count']))
    return df_temp1

def spdb_data_Extractor( stime, Etime, MAID_List, interval,metric, col_list):
    try:
        #url = r'http://spdb.oss.comcast.net/GetBulkData?nodetype_id=15&attribute_ids=' + dev +'&attribute_ids=' + interface +'&metric_ids=' + metric_id+ '&range='+interval
        #http://spdb.io.comcast.net/GetTextBulkDataBySearch?nodetype=net_soam&start_time=1500595200&end_time=1500595201&service=^(ME-)&metric=packetsSent,900,raw        
        url = r'http://spdb.io.comcast.net/GetTextBulkDataBySearch?nodetype=net_soam&start_time=' + stime + '&end_time=' + Etime + '&service=^(' + MAID_List + ')&metric=' + metric + ','+ interval + ',raw'
        print(url)        
        resp = urllib.request.urlopen(url).read()
        respstr = str(resp).strip("b'")
        count = 0        
        for i in col_list:
            if count == 0:            
                colum =i
                count+=1
            else:
                colum+=','+i
        colum+='$'
        respstr = colum+ respstr.replace('\\n','$')
        print(respstr)
        df_DExt=read_csv(StringIO(respstr), lineterminator='$')
        return df_DExt
    except urllib.error.HTTPError:
        my_file = r"C:\Users\dkashi200\Yawaris\python\my_ME_SOAM_MAID_log.csv"
        with open(my_file, "w") as the_file:
            line='HTTPError on' 
            the_file.write(line + "\n")
            
EVC=[]
Serv = ['Ethernet Network Service', 'Ethernet Virtual Private Line', 'Ethernet Private Line','Ethernet Dedicated Internet Access']
CR = data_base('DHIRENK', 'dH1reN','clipdb-po-ap-sc.sys.comcast.net', 1555, 'CRMRPR0_EXT_RO_SVC', False)


############################Cramer EVC##########################################################################
CR.sql_file = r"C:\Users\dkashi200\Yawaris\python\Cramer_CPE.sql"

df = query.Exp(CR)
df=df[df['SERVICETYPE']!='Ethernet Dedicated Internet Access']
df.index.name = 'S.No'
df['PVLAN']=df['PVLAN'].astype('str')
df['DEVPORTCNAME']=df['DEVPORTCNAME'].astype('str')
'''
col_list = ['SOAM_MAP_CLLI_PORT','COS', 'KI', 'MI', 'AGG', 'TS', 'VALUE']
my_file=r'C:\Users\dkashi200\Yawaris\Temp\ME_connection_07_11_2017.csv'
df_list=read_csv(my_file,names = ['SOAM_MAP_CLLI_PORT','COS', 'KI', 'MI', 'AGG', 'TS', 'VALUE'])
'''

#hpna cfm cinfigured cpe processing

col_list = ['Host Name','Device IP','Device Type','Device Status','Device Vendor','Device Model','Configuration Text','Partition']
my_file=r'C:\Users\dkashi200\Downloads\search_result1.csv'
df_hpna_raw=read_csv(my_file)
df_hpna_raw['Configuration Text']= df_hpna_raw['Configuration Text'].astype('str')
df_hpna_raw['mep_count']=df_hpna_raw['Configuration Text'].apply(lambda x: x.count('#'))
df_hpna_raw_compressed=df_hpna_raw[['Host Name', 'Device IP','Device Model', 'Configuration Text','mep_count']]
df_temp2=DataFrame(columns=['Host Name', 'Device IP','Device Model', 'Configuration Text','mep_count'])
for i in df_hpna_raw_compressed.index.values:
    df_temp2=df_temp2.append(dataframe_expand(df_hpna_raw_compressed.ix[i]))
df_temp2['MAID_full']=df_temp2['Configuration Text'].apply(lambda x: x[x.rfind('service')+8 :x.rfind('port')])
df_temp2['port']=df_temp2['Configuration Text'].apply(lambda x: x[x.rfind('port')+5 : x.rfind('type')].replace(' ',''))
df_temp2['port']=df_temp2['port'].astype('str')
df_temp2['mepid']=df_temp2['Configuration Text'].apply(lambda x: x[x.rfind('mepid')+6 : ])
df_hpna=df_temp2
df_hpna['LOB']=df_hpna['MAID_full'].apply(lambda x: x[:3])
df_hpna[df_hpna['LOB'] == 'ME-']
df_hpna_ME=df_hpna[df_hpna['LOB'] == 'ME-']
df_hpna_ME['SOAM_MA']=df_hpna_ME['MAID_full'].apply(lambda x: x[:x.rfind('NNI')+3])
df_hpna_ME['SOAM_MAID']=df_hpna_ME['MAID_full'].apply(lambda x: x[x.rfind('NNI')+4:])
df_hpna_ME['VLAN']=df_hpna_ME['SOAM_MAID'].apply(lambda x: x.split('_')[-1].replace(' ','') if x.find('_') > 0 else 'NA')
df_hpna_ME['CLLI']=df_hpna_ME['Host Name'].apply(lambda x: x[x.find('cpe')-12:x.find('cpe')-1].upper() if x.find('cpe') > 0 else 'NA')
number_of_cpe_w_cfm=len(merge(df_hpna_ME,df,how = 'inner',left_on='CLLI', right_on='DEVCLLI')['CLLI'].unique())

#hpna cpe witout cfm configuration processing
my_file=r'C:\Users\dkashi200\Downloads\Devices_wo_CFM_CONF.csv'
df_hpna_raw=read_csv(my_file)
df_hpna_cpe_wo_cfm=df_hpna_raw
df_hpna_cpe_wo_cfm['CLLI']=df_hpna_cpe_wo_cfm['Host Name'].apply(lambda x: x[x.find('cpe')-12:x.find('cpe')-1].upper() if x.find('cpe') > 0 else 'NA')

df_hpna_cpe_wo_cfm=df_hpna_cpe_wo_cfm[df_hpna_cpe_wo_cfm['CLLI']!='NA']

number_of_cpe_wo_cfm=len(merge(df_hpna_cpe_wo_cfm,df,how = 'inner',left_on='CLLI', right_on='DEVCLLI')['CLLI'].unique())

df_hpna_me_cpe_wo_cfm=merge(df_hpna_cpe_wo_cfm,df,how = 'inner',left_on='CLLI', right_on='DEVCLLI')

Columns_List=['SOAM_MAP_CLLI_PORT','COS','KI','MI','AGG','TS','VALUE']
MAID_EVC='ME-'
Start_Time=str(int(time.time()/300)*300-3601)
End_Time = str(int(time.time()/300)*300-2701)
mi='900'
ki='packetsSent'
df_list=spdb_data_Extractor( Start_Time, End_Time, MAID_EVC, mi,ki, Columns_List)
my_file = r"C:\Users\dkashi200\Yawaris\python\ME_SOAM_spdb"+Start_Time+".csv"
df_list.to_csv(my_file)
'''
my_file = r"C:\Users\dkashi200\Yawaris\python\CBH_MACRO_SOAM_spdb.csv"
df_list=read_csv(my_file,index_col=0)

'''
df_list['SOAM_MA']=df_list['SOAM_MAP_CLLI_PORT'].apply(lambda x: x[:x.rfind('NNI')+3])
df_list['SOAM_MAID']=df_list['SOAM_MAP_CLLI_PORT'].apply(lambda x: x[x.rfind('NNI')+4:x.find('^')])
df_list['MEP_PAIR']=df_list['SOAM_MAP_CLLI_PORT'].apply(lambda x: x[x.find('^')+1:x.find(':',x.find('^'))])
df_list['CLLI']=df_list['SOAM_MAP_CLLI_PORT'].apply(lambda x: x[x.find(':',x.find('^'))+1:x.find('.',x.find('^'))])
df_list['Port']=df_list['SOAM_MAP_CLLI_PORT'].apply(lambda x: x[x.find('.',x.find('^'))+1:])
df_list['VLAN']=df_list['SOAM_MAID'].apply(lambda x: x.split('_')[-1] if x.find('_') > 0 else 'NA')
Number_of_misconf_EVC=len(df_list[df_list['SOAM_MAID'].str.rfind('V') < 0]['SOAM_MAID'].unique())*100/len(df_list['SOAM_MAID'].unique())
Number_of_misconf_conn=len(df_list[df_list['SOAM_MAID'].str.rfind('V') < 0]['SOAM_MAID'])*100/len(df_list['SOAM_MAID'])

print('Number_of_misconf_EVC :',Number_of_misconf_EVC)
print('Number_of_misconf_conn :',Number_of_misconf_conn)



'''
df_EOHFC1=merge(df_list,df,how='inner',left_on='SOAM_MAID',right_on='EVCID')
df_EOHFC2=merge(df_list,df,how='inner',left_on='SOAM_MAID',right_on='OVCID')
df_EOHFC1[['EVCID', 'OVCID']].drop_duplicates()
df_EOHFC2[['EVCID', 'OVCID']].drop_duplicates()
'''

