

# CLEANED DATA

## HD
```
>>> RF
valid score = 0.8273030018761727
test accuracy = 0.8518518518518519
test f1 = 0.7999999999999999
[[30  2]
 [ 6 16]]
{'max_depth': 6, 'n_estimators': 75}

>>> LR_L1
valid score = 0.6070284406668846
test accuracy = 0.7592592592592593
test f1 = 0.6976744186046512
[[26  6]
 [ 7 15]]
{'C': 0.7, 'penalty': 'l2'}

>>> MLP
valid score = 0.6328697834371224
test accuracy = 0.6111111111111112
test f1 = 0.6557377049180326
[[13 19]
 [ 2 20]]
{'hidden_layer_sizes': 50}

```

## CC
```
>>> RF
valid score = 0.6112987012987012
test accuracy = 0.9461077844311377
test f1 = 0.4705882352941177
[[154   2]
 [  7   4]]
{'max_depth': 8, 'n_estimators': 25}

>>> LR_L1
valid score = 0.737835497835498
test accuracy = 0.9640718562874252
test f1 = 0.7500000000000001
[[152   4]
 [  2   9]]
{'C': 0.7, 'penalty': 'l1'}

>>> MLP
valid score = 0.12091454644646134
test accuracy = 0.0658682634730539
test f1 = 0.12359550561797754
[[  0 156]
 [  0  11]]
{'hidden_layer_sizes': (10, 10)}
```

# CLEANED + BALANCED DATA

## HD
```
>>> RF
valid score = 0.8472106002517347
test accuracy = 0.8148148148148148
test f1 = 0.761904761904762
[[28  4]
 [ 6 16]]
{'max_depth': 6, 'n_estimators': 25}

>>> LR_L1
valid score = 0.639105151016369
test accuracy = 0.7777777777777778
test f1 = 0.7272727272727273
[[26  6]
 [ 6 16]]
{'C': 1.0, 'penalty': 'l2'}

>>> MLP
valid score = 0.6799232577567952
test accuracy = 0.7222222222222222
test f1 = 0.6938775510204083
[[22 10]
 [ 5 17]]
{'hidden_layer_sizes': 150}
```

## CC
```
>>> RF
valid score = 0.9793823256146844
test accuracy = 0.9640718562874252
test f1 = 0.7500000000000001
[[152   4]
 [  2   9]]
{'max_depth': 8, 'n_estimators': 75}

>>> LR_L1
valid score = 0.933034204339575
test accuracy = 0.9640718562874252
test f1 = 0.7500000000000001
[[152   4]
 [  2   9]]
{'C': 0.9, 'penalty': 'l1'}

>>> MLP
valid score = 0.7240147355684323
test accuracy = 0.49101796407185627
test f1 = 0.15841584158415842
[[74 82]
 [ 3  8]]
{'hidden_layer_sizes': 100}

```

# CLEANED + BALANCED + FEATURE SELECTED DATA

## HD
```
>>> RF
valid score = 0.8574747765279408
test accuracy = 0.8148148148148148
test f1 = 0.7499999999999999
[[29  3]
 [ 7 15]]
{'max_depth': 4, 'n_estimators': 10}

>>> LR_L1
valid score = 0.639105151016369
test accuracy = 0.7777777777777778
test f1 = 0.7272727272727273
[[26  6]
 [ 6 16]]
{'C': 0.9, 'penalty': 'l2'}

>>> MLP
valid score = 0.6832946001367055
test accuracy = 0.7777777777777778
test f1 = 0.7272727272727273
[[26  6]
 [ 6 16]]
{'hidden_layer_sizes': (50, 50)}
```

## CC
```
>>> RF
valid score = 0.9777610104163121
test accuracy = 0.9640718562874252
test f1 = 0.7272727272727273
[[153   3]
 [  3   8]]
{'max_depth': 6, 'n_estimators': 10}

>>> LR_L1
valid score = 0.9366865942814734
test accuracy = 0.9640718562874252
test f1 = 0.7500000000000001
[[152   4]
 [  2   9]]
{'C': 1.0, 'penalty': 'l1'}


>>> MLP
valid score = 0.8467671339824451
test accuracy = 0.4251497005988024
test f1 = 0.1724137931034483
[[61 95]
 [ 1 10]]
{'hidden_layer_sizes': (50, 50)}
```

# Random model 

## HD
```
>>> Random
valid score = 0.5813900226757369
test accuracy = 0.5740740740740741
test f1 = 0.46511627906976744
[[21 11]
 [12 10]]
{}

>>> Baseline
valid score = 0.26285714285714284
test accuracy = 0.5925925925925926
test f1 = nan
[[32  0]
 [22  0]]
{}
```

## CC 
```
>>> Random
valid score = 0.5012584953454615
test accuracy = 0.5149700598802395
test f1 = 0.14736842105263157
[[79 77]
 [ 4  7]]
{}

>>> Baseline
valid score = 0.0
test accuracy = 0.9341317365269461
test f1 = nan
[[156   0]
 [ 11   0]]
{}
```
