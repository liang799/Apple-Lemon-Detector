import splitfolders

splitfolders.ratio("C:/Python/Apple-Lemon-Detector/raw", output="datasets",
                   seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False)  # default values
