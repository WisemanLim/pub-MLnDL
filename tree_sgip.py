#-*- coding: utf-8 -*-
# Ref : https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn import svm, metrics, model_selection

def  main() :
    csv = pd.read_csv('./sgip/form_vn2020f_v22.2.csv')
    feature_names = csv.columns.tolist()
    '''
    기초화장품선호브랜드        aq3(
    선호하는향               aq41~48
    화장품성분               aq51~59
    클렌저제형종류            cq1
    브랜드(7종)             cq2lp1_cq2~cq2lp7_cq2
    선호클렌저제형            cq4
    보습제제형               cq6
    브랜드(5종)             cq7lp1_cq7~cq7lp5_cq7
    기초화장품제형            cq15
    '''
    class_names = ['1', '2', '3', '4', '5', '6', '7', '99'] # cq4
    features = csv[feature_names].to_numpy()
    targets = ('cq4')

    target_feature_names = ['sq2', 'sq4open', 'sq4', 'bq1', 'bq1x1_1', 'bq1x1_2', 'bq1x1_3', 'bq1x1_4', 'bq1x2', 'bq1x3', 'dq2']
    target_features = csv.loc[:, target_feature_names]
    criterion = 'log_loss' # gini(default), entropy, log_loss

    # 의사결정 모델 클래스 생성 (3)
    dtSGIP = DecisionTreeClassifier(criterion=criterion, max_depth=3)
    #모델을 훈련 (4)
    dtSGIP.fit(target_features, targets)

    filename = 'sgip-dtree_' + criterion
    """
    # DOT 언어의 형식으로 결정 나무의 형태를 출력한다.
    with open(filename + ".dot", mode = 'w') as f:
        tree.export_graphviz(cIris, out_file = f)
    f.close() """
    import graphviz
    """
    # DOT 형태를 이미지로 출력
    with open(filename + ".dot") as f:
        dot_graph = f.read()
    f.close()
    graph = graphviz.Source(dot_graph)
    graph.format = "png"
    graph.render(filename)
    """
    # Print text
    dot_data = tree.export_text(dtSGIP)
    print(dot_data)

    # graphviz
    dot_data = tree.export_graphviz(dtSGIP, out_file=None
                                    , feature_names=target_feature_names, class_names=class_names
                                    , filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.render(filename)

    # dtreeviz
    from dtreeviz.trees import dtreeviz  # remember to load the package
    viz = dtreeviz(dtSGIP, target_features, targets, target_name="target"
                   , feature_names=target_feature_names, class_names=list(class_names))
    viz.save(filename + ".svg")
    # delete temporary file : filename, i.e 03iris-dtress
    import os
    os.system("rm -Rf " + filename)

if __name__ == '__main__':
    main()
