import os
import pandas as pd
import glob

Capoff_path = os.getcwd() + "/images/labeloncapoff"
CapON_path = os.getcwd() + "/images/labeloncapon"


os.chdir(Capoff_path)

ImageName = []
for x in glob.glob("*.jpg"):
	ImageName.append(x)


df = pd.DataFrame({'image' : ImageName , 'label' : "2"})

df = df.sample(frac=1).reset_index(drop=True)
print(df)


df.to_csv("../labeloncapoff.csv", index = False)


os.chdir(CapON_path)

ImageName = []
for x in glob.glob("*.jpg"):
	ImageName.append(x)


df = pd.DataFrame({'image' : ImageName , 'label' : "3"})
df = df.sample(frac=1).reset_index(drop=True)
print(df)

df.to_csv("../labenoncapoN.csv", index = False)

