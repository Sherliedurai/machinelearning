import numpy as np
import matplotlib.pyplot as plt

greyhounds=500
laborer=500
grey_height=28+4*np.random.randn(greyhounds)
lab_height=20+4*np.random.randn(laborer)

plt.hist([grey_height,lab_height],stacked=False,color=['r','b'])
plt.show()
