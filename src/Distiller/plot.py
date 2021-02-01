import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()

    df_list = []
    step = []
    f1 = []
    em = []
    file_list = os.listdir(args.dir)
    file_list.sort(reverse=False)
    for file in file_list:
        if 'eval' in file:
            try:
                step = int(file.split('_')[0])
            except Exception as e:
                print(e)
                continue
            with open(os.path.join(args.dir, file)) as f:
                lines = f.readlines()[:8]
                em = round(float(lines[1].split(":")[1].lstrip().rstrip().replace(",","").strip("\n")), 2)
                f1 = round(float(lines[2].split(":")[1].lstrip().rstrip().replace(",","").strip("\n")), 2)
                df_list.append(pd.DataFrame({"step":[step],"F1":[f1],"EM":[em]}))
    df = pd.melt(pd.concat(df_list), ["step"]).rename({"variable": "metrics"}, axis=1)
    pic = sns.lineplot(x='step', y='value', hue="metrics", data=df)

    print(f"save to {list(filter(None,args.dir.split('/'))).pop()}.jpg")
    # plt.show()
    pic.get_figure().savefig(f'{list(filter(None,args.dir.split("/"))).pop()}.jpg')