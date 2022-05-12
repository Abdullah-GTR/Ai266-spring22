import  pandas as pd
import numpy as np
testDF = pd.read_csv('test.csv')
idDF = testDF[['id']]
idDF.insert(1,"target",0)
idDF['target'] = np.random.rand(700000,1)
print(idDF)
idDF.to_csv('out.CSV', index=False)
