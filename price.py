from sklearn import tree
#nt-1,ht-2,sr-3
features = [[3,2000,1],[2,800,2],[2,850,1],[1,550,1],[4,2000,3]]
labels=[250000,300000,150000,78000,150000]
determiner=tree.DecisionTreeClassifier()
determiner=determiner.fit(features,labels)

print(determiner.predict([[3,2000,2]]))