from sklearn import tree

features = [[100, 2 ],[102 , 3],[300,4],[400,4]]
lables = [ 'tfa7A', 'tfa7A' , 'limouna' , 'limouna']
clsf = tree.DecisionTreeClassifier()
clf = clsf.fit(features,lables)

print (clf.predict([[ 160 , 5 ]]))
