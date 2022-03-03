### CSV Extraction Code written by: Jim Todd of Stack Overflow
i=0
import csv
import pandas as pd
df = pd.DataFrame()
with open('C:/Users/sword/Anaconda3/envs/exceltest/RF_SubjP02_Free_STATIC_TR01.csv', 'r') as csvfile:
     csvreader = csv.reader(csvfile, delimiter=',')
     for row in csvreader:
         if csvreader.line_num == 3:
             print(row)
             # df = pd.DataFrame(columns = row)
             # df.columns = row
         if csvreader.line_num >= 6:
            if row:
                if i<10:
                    print(i)
                    i+=1
                    df.append(row)
            else:
                break
print(df)