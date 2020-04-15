#%%
from eda import synonym_replacement, random_deletion, lem,random_swap, random_insertion

#%%
print(synonym_replacement(["this", "is", "a", "test", "case"], 2))
print(random_deletion(["this", "is", "a", "test", "case"], 2))
print(random_swap(["this", "is", "a", "test", "case"], 2))
print(random_insertion(["this", "is", "a", "test", "case"], 2))

#%%
import pandas as pd
from functools import partial

df = pd.read_csv("./selected_docs.csv")

#%%
des = df["Description"].str.lower()
des = des.apply(lambda x: lem(x))

sr_alpha = 0.1
ri_alpha = 0.1
rs_alpha = 0.1
rd_alpha = 0.1




#%%
sr = des.apply(
    lambda x: " ".join(
        synonym_replacement(
            x.strip().split(), 
            n=int(sr_alpha * len(x)
            )
            )
            )
)
ri = des.apply(
    lambda x: " ".join(
        random_insertion(
            x.strip().split(), 
            n=int(ri_alpha * len(x)
            )
            )
            )
)
rs = des.apply(
    lambda x: " ".join(
        random_swap(
            x.strip().split(), 
            n=int(rs_alpha * len(x)
            )
            )
            )
)
rd = des.apply(
    lambda x: " ".join(
        random_deletion(
            x.strip().split(), 
            rd_alpha
            )
            )
)
#%%
df['sr'] = sr
df['ri'] = ri
df['rs'] = rs
df['rd'] = rd
df.to_csv('eda_dataset.csv',index=False)