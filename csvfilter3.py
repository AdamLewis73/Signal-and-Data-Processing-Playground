### CSV Extraction Code written by: Jim Todd of Stack Overflow
import csv
import pandas as pd
temp = []
#with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_STATIC_TR01.csv', 'r') as csvfile:
with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_STATIC_TR02.csv', 'r') as csvfile:
     csvreader = csv.reader(csvfile, delimiter=',')
     for row in csvreader:
         if csvreader.line_num == 3:
             temp.append(row)
         if csvreader.line_num >= 6:
            if row:
                temp.append(row)
            else:
                break
df = pd.DataFrame(temp)
df.columns = df.iloc[0]
print(df)
print('BREAK BREAK BREAK')
df = df.drop(0)
print(df)
print('BREAK BREAK BREAK3')
#df.reindex(df.index.drop(1))
#print(df.columns)
#print(len(df))
df.reset_index(drop = True, inplace = True)
print(df)
print('BREAK BREAK BREAK')
print(df['Noraxon Desk Receiver - EMG1'])