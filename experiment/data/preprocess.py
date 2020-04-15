#%%
import pandas as pd
import re

#%%
df = pd.read_csv("./apis.csv")
del df["API Provider"]
df = df.dropna()

#%%
# df.describe()
# df.iloc[108]['Description']
df.iloc[108]["Description"] = df.iloc[108]["Description"][107:]

#%%
filter_url = r"http[s]?://.*[\s\\,.]"
f_url = re.compile(filter_url)

filter_deprecated = r"^\[.*This profile is being maintained purely for historical and research purposes.\]"
f_depre = re.compile(filter_deprecated)

# sep two words first, 00andMe->00and Me
sep_words = r"([a-z]+)([A-Z]+[a-z]*)"
sep_w = re.compile(sep_words)

# sep num and word 00and -> 00 and
sep_num1 = r"([0-9]+)([a-zA-Z]+)"
sep_num2 = r"([a-zA-Z]+)([0-9]+)"
sep_n_before = re.compile(sep_num1)
sep_n_after = re.compile(sep_num2)

filter_no_eng = r"[^a-zA-Z0-9]"
f_no_eng = re.compile(filter_no_eng)


def split_w(matched):
    return matched.group(1) + " " + matched.group(2)


#%%
# f_no_eng.sub('url','https://askquickly.org/ ')

#%%
des = df["Description"]
df["Description"] = des.apply(
    lambda x: f_no_eng.sub(" ",
        sep_n_after.sub(split_w,
            sep_n_before.sub(split_w, \
                sep_w.sub(split_w, \
                    f_depre.sub("", \
                        f_url.sub("", x)))),
        ),
    )
)

#%%
lens = df["Description"].str.len()
lens.describe()
#%%
df = df[df["Description"].str.len() > 100]
#%%
df.to_csv("filtered_apis.csv", index=False)

