# Information

Group Name: BD06

Project Name: BD06: Advanced Clustering in Big Data

Group Members:
1. WANG Zeyu shyzw6@nottingham.edu.cn
2. YUAN Yuze shyyy1@nottingham.edu.cn
3. HE Yuxin scyyh6@nottingham.edu.cn
4. LI Minxiao scyml2@nottingham.edu.cn

# How to run the code?

Environment: CSLinux Server

Steps:
1. `cd` into the root directory of this project
2. uncomment the experiments you want to run in `MR-OPTICS.py` (our contribution as propsed in paper) or `MR-DBSCAN.py`.
3. run `/usr/bin/python MR-OPTICS.py` or `/usr/bin/python MR-DBSCAN.py`

This code has been tested on CSLinux Server. As the cluster labels are calibrated manually as other papers, labels may be different when testing on other machines (this basicly won't happen as no randomness is involved in clustering algorithm). But please contact us if code does not work correctly!!!

# Note:
In this repository, we only provide 4 test cases because the 10MB limitation is very strict. We also provide a version with full test cases through email.


# Code Structure

```
BD06
│  .gitignore
│  MR-DBSCAN.py # experiments for MR-DBSCAN
│  MR-OPTICS.py # experiments for MR-OPTICS
│  README.md
│
├─common # common function for MR-DBSCAN and MR-OPTICS
│  EvenSplitPartitioner.py # partation the space
│  Graph.py # Graph for final merge
│  Heap.py # Heap to store Points
│  LabeledPoint.py # LabeledPoint in the dataset
│  Point.py # Point in the dataset
│  Rectangle.py  # Rectangle split the space
│  TypedRDD.py # add type for RDD in python
│  utils.py # tools for the functions
│  Verbose.py # generate logs
│  __init__.py
│
├─dataset
│  │  labeled_data.csv # file to test algorithms
│  └─sklearn # dataset generated from sklearn
│  blobs10000.csv
│  blobs100000.csv
│  blobs500000.csv
│  circles10000.csv
│  circles100000.csv
│  circles500000.csv
│  clusters10.csv # 10-cluster Blobs datasets with 100000 points
│  clusters6.csv # 6-cluster Blobs datasets with 100000 points
│  moons10000.csv
│  moons100000.csv
│  moons500000.csv
│
├─DBSCAN
│  DBSCAN.py # MR-DBSCAN core file 
│  LocalDBSCANNaive.py # Classic DBSCAN 
│  __init__.py
│
└─OPTICS
    LocalOPTICSNaive.py # Classic OPTICS 
    OPTICS.py # MR-OPTICS core file 
    __init__.py
```

# Code References

MR-DBSCAN: https://github.com/irvingc/dbscan-on-spark
