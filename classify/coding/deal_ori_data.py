#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np


if __name__ == '__main__':
    sva_info_data=pd.read_csv("../ori_data/amplify_sva_info.csv").sort_values("contribution",ascending=False)
    sva_all_data = pd.read_csv("../ori_data/amplify_sva_all.csv")
    snp_sort_list=np.array(sva_info_data["ID_REF"])
    snp_size=len(snp_sort_list)
    for i in range(1,11):
        percent=i/float(10)
        select_number=int(snp_size*percent)
        select_list=snp_sort_list[0:select_number]
        data = sva_all_data[sva_all_data["ID_REF"].isin(select_list)].drop("ID_REF", axis=1)
        data.to_csv("../ori/"+str(percent)+"_ori.csv",index=False)



