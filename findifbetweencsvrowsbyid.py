import pandas as pd
import csv
def finddifferences():
    file1 = "C:\\Users\\Pocky\\Desktop\\midtermcs506\\finalizedediteddata\\FinalSubmission.csv"
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv("C:\\Users\\Pocky\\Desktop\\midtermcs506\\finalizedediteddata\\xp.csv")
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    #print(df1)
    #print(df2)
    row = 1
    ct = 300001 #for looking for 300,000 rows
    while(row < ct):
        if (df1.loc[row, "Id"] != df2.loc[row, "Id"]):
            break
        row += 1
    return row


print(finddifferences())