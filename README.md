<div class="cell code" data-execution_count="1">

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings('ignore')
color= sns.color_palette()
```

</div>

<div class="cell code" data-execution_count="2">

``` python
df=pd.read_csv('/home/tmahajan/default.csv')
```

</div>

<div class="cell code" data-execution_count="3">

``` python
df
```

<div class="output execute_result" data-execution_count="3">

``` 
   default  student   balance   income
0        No      Yes    520.15   20000
1        No       No    453.12   23000
2        No       No   1023.66   12000
3        No      Yes   1444.11   56000
4        No      Yes    320.03   34000
5        No       No    432.90   43000
6       Yes       No   2300.00   22000
7        No       No   1300.75   25000
8       Yes       No     65.00   31000
9        No       No    345.32   35000
10       No       No    654.77   36000
11       No      Yes    213.00   37000
12       No      Yes    544.66   38000
13      Yes       No    765.77   39000
14      Yes       No    900.66   42000
15       No       No    211.54   41000
16       No       No   2067.00   33000
17       No       No   1200.23   15000
18       No      Yes   3400.00   18000
19       No       No   1100.00    5644
20       No      Yes    533.00   18900
21       No       No    213.00   23455
22      Yes      Yes    345.00   35345
23       No       No    786.88   24534
24       No       No    221.00   34256
25       No      Yes    432.00   75883
26      Yes      Yes     56.00   23455
27       No       No     89.00   97666
28       No      Yes    999.00   54457
29      Yes       No    444.00   23443
30      Yes      Yes    222.00   54647
31       No       No    777.00   43235
32      Yes       No    994.00    2532
33       No       No    111.00    6564
34       No      Yes    543.00   23255
35       No       No    654.75   23435
36       No       No   1244.66   23546
37       No      Yes    244.88   23235
38       No       No    457.77   76754
39       No       No    653.77   23245
40      Yes      Yes    232.11   23546
41      Yes       No    887.66   35657
42      Yes      Yes    657.90   34532
43       No       No    372.00   23243
44       No       No    644.00    4657
45       No      Yes    112.00   34566
46       No       No    113.00   45646
47       No       No    443.00   34436
48      Yes       No    221.00   97665
```

</div>

</div>

<div class="cell code" data-execution_count="4">

``` python
df.shape
```

<div class="output execute_result" data-execution_count="4">

    (49, 4)

</div>

</div>

<div class="cell code" data-execution_count="5">

``` python
df.describe()
```

<div class="output execute_result" data-execution_count="5">

``` 
          balance         income
count    49.000000     49.000000
mean    672.796939  33702.734694
std     625.016206  20409.293964
min      56.000000   2532.000000
25%     232.110000  23243.000000
50%     520.150000  33000.000000
75%     887.660000  39000.000000
max    3400.000000  97666.000000
```

</div>

</div>

<div class="cell code" data-execution_count="6">

``` python
df
```

<div class="output execute_result" data-execution_count="6">

``` 
   default  student   balance   income
0        No      Yes    520.15   20000
1        No       No    453.12   23000
2        No       No   1023.66   12000
3        No      Yes   1444.11   56000
4        No      Yes    320.03   34000
5        No       No    432.90   43000
6       Yes       No   2300.00   22000
7        No       No   1300.75   25000
8       Yes       No     65.00   31000
9        No       No    345.32   35000
10       No       No    654.77   36000
11       No      Yes    213.00   37000
12       No      Yes    544.66   38000
13      Yes       No    765.77   39000
14      Yes       No    900.66   42000
15       No       No    211.54   41000
16       No       No   2067.00   33000
17       No       No   1200.23   15000
18       No      Yes   3400.00   18000
19       No       No   1100.00    5644
20       No      Yes    533.00   18900
21       No       No    213.00   23455
22      Yes      Yes    345.00   35345
23       No       No    786.88   24534
24       No       No    221.00   34256
25       No      Yes    432.00   75883
26      Yes      Yes     56.00   23455
27       No       No     89.00   97666
28       No      Yes    999.00   54457
29      Yes       No    444.00   23443
30      Yes      Yes    222.00   54647
31       No       No    777.00   43235
32      Yes       No    994.00    2532
33       No       No    111.00    6564
34       No      Yes    543.00   23255
35       No       No    654.75   23435
36       No       No   1244.66   23546
37       No      Yes    244.88   23235
38       No       No    457.77   76754
39       No       No    653.77   23245
40      Yes      Yes    232.11   23546
41      Yes       No    887.66   35657
42      Yes      Yes    657.90   34532
43       No       No    372.00   23243
44       No       No    644.00    4657
45       No      Yes    112.00   34566
46       No       No    113.00   45646
47       No       No    443.00   34436
48      Yes       No    221.00   97665
```

</div>

</div>

<div class="cell code" data-execution_count="7">

``` python
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(y= df['balance '])
plt.show()


plt.figure(figsize=(15,5))
plt.subplot(1,2,2)
sns.boxplot(y= df['income'])
plt.show()
```

<div class="output display_data">

![](1b25fe27ffe977a65029f45ba08e0fb7c57d1975.png)

</div>

<div class="output display_data">

![](5693c2eae4f7028be8e0d2a2b463f866beed878e.png)

</div>

</div>

<div class="cell code" data-execution_count="8">

``` python
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(df['student '])

plt.figure(figsize=(15,5))
plt.subplot(1,2,2)
sns.countplot(df['default '])
plt.show()
```

<div class="output display_data">

![](ebbc2a7c65c6316b9be414ed858bd4228eae3ba7.png)

</div>

<div class="output display_data">

![](5e6c545959717e0d80f3618a8339d5b2d5225203.png)

</div>

</div>

<div class="cell code" data-execution_count="9">

``` python
df['default '].value_counts()
```

<div class="output execute_result" data-execution_count="9">

    No     36
    Yes    13
    Name: default , dtype: int64

</div>

</div>

<div class="cell code" data-execution_count="10">

``` python
df
```

<div class="output execute_result" data-execution_count="10">

``` 
   default  student   balance   income
0        No      Yes    520.15   20000
1        No       No    453.12   23000
2        No       No   1023.66   12000
3        No      Yes   1444.11   56000
4        No      Yes    320.03   34000
5        No       No    432.90   43000
6       Yes       No   2300.00   22000
7        No       No   1300.75   25000
8       Yes       No     65.00   31000
9        No       No    345.32   35000
10       No       No    654.77   36000
11       No      Yes    213.00   37000
12       No      Yes    544.66   38000
13      Yes       No    765.77   39000
14      Yes       No    900.66   42000
15       No       No    211.54   41000
16       No       No   2067.00   33000
17       No       No   1200.23   15000
18       No      Yes   3400.00   18000
19       No       No   1100.00    5644
20       No      Yes    533.00   18900
21       No       No    213.00   23455
22      Yes      Yes    345.00   35345
23       No       No    786.88   24534
24       No       No    221.00   34256
25       No      Yes    432.00   75883
26      Yes      Yes     56.00   23455
27       No       No     89.00   97666
28       No      Yes    999.00   54457
29      Yes       No    444.00   23443
30      Yes      Yes    222.00   54647
31       No       No    777.00   43235
32      Yes       No    994.00    2532
33       No       No    111.00    6564
34       No      Yes    543.00   23255
35       No       No    654.75   23435
36       No       No   1244.66   23546
37       No      Yes    244.88   23235
38       No       No    457.77   76754
39       No       No    653.77   23245
40      Yes      Yes    232.11   23546
41      Yes       No    887.66   35657
42      Yes      Yes    657.90   34532
43       No       No    372.00   23243
44       No       No    644.00    4657
45       No      Yes    112.00   34566
46       No       No    113.00   45646
47       No       No    443.00   34436
48      Yes       No    221.00   97665
```

</div>

</div>

<div class="cell code" data-execution_count="11">

``` python
df['student '].value_counts()
```

<div class="output execute_result" data-execution_count="11">

    No     32
    Yes    17
    Name: student , dtype: int64

</div>

</div>

<div class="cell code" data-execution_count="12">

``` python
df['student '].value_counts(normalize=True)
```

<div class="output execute_result" data-execution_count="12">

    No     0.653061
    Yes    0.346939
    Name: student , dtype: float64

</div>

</div>

<div class="cell code" data-execution_count="13">

``` python
df['default '].value_counts(normalize=True)
```

<div class="output execute_result" data-execution_count="13">

    No     0.734694
    Yes    0.265306
    Name: default , dtype: float64

</div>

</div>

<div class="cell code" data-execution_count="14">

``` python
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(df['default '], df['balance '])

plt.subplot(1,2,2)
sns.boxplot(df['default '], df['income'])
plt.show()
```

<div class="output display_data">

![](5ce1abd9893992fb082252d2c465c7f7b16ba0a1.png)

</div>

</div>

<div class="cell code" data-execution_count="15">

``` python
pd.crosstab(df['student '],df['default '],normalize='index').round(2)
```

<div class="output execute_result" data-execution_count="15">

    default     No   Yes
    student             
    No        0.75  0.25
    Yes       0.71  0.29

</div>

</div>

<div class="cell code" data-execution_count="16">

``` python
sns.heatmap(df[['balance ','income']].corr(),annot=True)
plt.show()
```

<div class="output display_data">

![](984303057b215997a6629ca3f60249398387b190.png)

</div>

</div>

<div class="cell code" data-execution_count="17">

``` python
df.isnull().sum()
```

<div class="output execute_result" data-execution_count="17">

    default     0
    student     0
    balance     0
    income      0
    dtype: int64

</div>

</div>

<div class="cell code" data-execution_count="18">

``` python
Q1,Q3=df['balance '].quantile([.25,.75])
IQR =Q3-Q1
LL =Q1- 1.5*(IQR)
UL =Q3+ 1.5*(IQR)
```

</div>

<div class="cell code" data-execution_count="19">

``` python
UL
```

<div class="output execute_result" data-execution_count="19">

    1870.985

</div>

</div>

<div class="cell code" data-execution_count="20">

``` python
df1=df[df['balance ']> UL]
```

</div>

<div class="cell code" data-execution_count="21">

``` python
df1
```

<div class="output execute_result" data-execution_count="21">

``` 
   default  student   balance   income
6       Yes       No    2300.0   22000
16       No       No    2067.0   33000
18       No      Yes    3400.0   18000
```

</div>

</div>

<div class="cell code" data-execution_count="22">

``` python
df['balance '] =np.where(df['balance ']> UL, UL , df['balance '])
```

</div>

<div class="cell code" data-execution_count="23">

``` python
sns.boxplot(y= df['balance '])
plt.show()
```

<div class="output display_data">

![](abd1e8e8e3db9a126081f17f59fc72ebe227700c.png)

</div>

</div>

<div class="cell code" data-execution_count="24">

``` python
df=pd.get_dummies(df,drop_first=True)
```

</div>

<div class="cell code" data-execution_count="25">

``` python
df
```

<div class="output execute_result" data-execution_count="25">

``` 
    balance   income  default _Yes  student _Yes
0    520.150   20000             0             1
1    453.120   23000             0             0
2   1023.660   12000             0             0
3   1444.110   56000             0             1
4    320.030   34000             0             1
5    432.900   43000             0             0
6   1870.985   22000             1             0
7   1300.750   25000             0             0
8     65.000   31000             1             0
9    345.320   35000             0             0
10   654.770   36000             0             0
11   213.000   37000             0             1
12   544.660   38000             0             1
13   765.770   39000             1             0
14   900.660   42000             1             0
15   211.540   41000             0             0
16  1870.985   33000             0             0
17  1200.230   15000             0             0
18  1870.985   18000             0             1
19  1100.000    5644             0             0
20   533.000   18900             0             1
21   213.000   23455             0             0
22   345.000   35345             1             1
23   786.880   24534             0             0
24   221.000   34256             0             0
25   432.000   75883             0             1
26    56.000   23455             1             1
27    89.000   97666             0             0
28   999.000   54457             0             1
29   444.000   23443             1             0
30   222.000   54647             1             1
31   777.000   43235             0             0
32   994.000    2532             1             0
33   111.000    6564             0             0
34   543.000   23255             0             1
35   654.750   23435             0             0
36  1244.660   23546             0             0
37   244.880   23235             0             1
38   457.770   76754             0             0
39   653.770   23245             0             0
40   232.110   23546             1             1
41   887.660   35657             1             0
42   657.900   34532             1             1
43   372.000   23243             0             0
44   644.000    4657             0             0
45   112.000   34566             0             1
46   113.000   45646             0             0
47   443.000   34436             0             0
48   221.000   97665             1             0
```

</div>

</div>

<div class="cell code" data-execution_count="26">

``` python
df.head()
```

<div class="output execute_result" data-execution_count="26">

``` 
   balance   income  default _Yes  student _Yes
0    520.15   20000             0             1
1    453.12   23000             0             0
2   1023.66   12000             0             0
3   1444.11   56000             0             1
4    320.03   34000             0             1
```

</div>

</div>

<div class="cell code" data-execution_count="27">

``` python
from sklearn.model_selection import train_test_split
```

</div>

<div class="cell code" data-execution_count="28">

``` python
df.columns=['balance','income','default','student']
```

</div>

<div class="cell code" data-execution_count="29">

``` python
x=df.drop('default',axis=1)
y=df['default']
```

</div>

<div class="cell code" data-execution_count="30">

``` python
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=5, stratify=y)
```

</div>

<div class="cell code" data-execution_count="31">

``` python
print(x_train.shape)
print(x_test.shape)
```

<div class="output stream stdout">

    (34, 3)
    (15, 3)

</div>

</div>

<div class="cell code" data-execution_count="32">

``` python
print(y_train.value_counts(normalize=True).round(2))
print(y_test.value_counts(normalize=True).round(2))
```

<div class="output stream stdout">

    0    0.74
    1    0.26
    Name: default, dtype: float64
    0    0.73
    1    0.27
    Name: default, dtype: float64

</div>

</div>

<div class="cell code" data-execution_count="33">

``` python
from sklearn.linear_model import LogisticRegression
```

</div>

<div class="cell code" data-execution_count="34">

``` python
lr = LogisticRegression()
```

</div>

<div class="cell code" data-execution_count="35">

``` python
lr.fit(x_train,y_train)
```

<div class="output execute_result" data-execution_count="35">

    LogisticRegression()

</div>

</div>

<div class="cell code" data-execution_count="36">

``` python
y_pred= lr.predict(x_test)
```

</div>

<div class="cell code" data-execution_count="37">

``` python
from sklearn.metrics import confusion_matrix, classification_report
```

</div>

<div class="cell code">

``` python
confusion_matrix(y_test, y_pred)
```

</div>

<div class="cell code" data-execution_count="39">

``` python
acc=(11+0)/(11+4+0+0)*100
```

</div>

<div class="cell code" data-execution_count="40">

``` python
print(acc,'%'.format(float(acc)))
```

<div class="output stream stdout">

    73.33333333333333 %

</div>

</div>

<div class="cell code">

``` python
```

</div>
