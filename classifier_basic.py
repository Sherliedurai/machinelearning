from sklearn import tree
#1-smooth,0-bumpy
features = [[140,1],[150,0],[120,1],[170,0]]
#1-apple,0-orange
labels = [1,0,1,0]
clf=tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)
print(clf.predict([[150,0]]))

