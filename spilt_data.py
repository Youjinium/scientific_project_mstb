import splitfolders

# https://pypi.org/project/split-folders/
DIR_NAME = "cvia2"
splitfolders.ratio(DIR_NAME, output="dataset", seed=1337, ratio=(.8, 0.1, 0.1))