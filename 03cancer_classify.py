#-*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def main():
    # 유방암 데이터 로드 및 화일 생성
    cancer = datasets.load_breast_cancer()
    import pandas as pd
    df = pd.DataFrame(data=cancer['data'], columns=cancer['feature_names'])
    df.to_csv('03cancer.csv', sep=',', index=False)
    # 생성된 유방암 데이터 로드
    # data = pd.read_csv('03cancer.csv')
    # print(data.columns, data.values)

    # 데이터를 훈련, 테스트 데이터로 나누기 (2)
    x_train, x_test, y_train, y_test = train_test_split(cancer.data,
                cancer.target,
                stratify=cancer.target,
                test_size=0.2,
                random_state=42)

    # 의사결정모델 클래스를 생성
    cancer_model = DecisionTreeClassifier(criterion='entropy', max_depth=3)

    # 모델을 학습 (3)
    cancer_model.fit(x_train, y_train)
    print("훈련 점수: {:.3f}".format(cancer_model.score(x_train, y_train)))
    print("테스트 점수: {:.3f}".format(cancer_model.score(x_test, y_test)))

    # DOT 언어의 형식으로 결정 나무의 형태를 출력
    with open('03cancer-dtree.dot', mode='w') as f:
        tree.export_graphviz (cancer_model, out_file=f
                              , feature_names=cancer.feature_names
                              , class_names=["cancer", "not cancer"])
    import os
    os.system("dot -T png 03cancer-dtree.dot -o 03cancer-dtree.png")

    # DOT 언어의 형식으로 결정 나무의 형태를 출력하면서 이미지 변환
    # import graphviz
    # graph = graphviz.Source(tree.export_graphviz(cancer_model, out_file=None
    #                                              , feature_names=cancer.feature_names
    #                                              , class_names=["cancer", "not cancer"]
    #                         , filled=True, rounded=True, special_characters=True
    #                         , proportion=False, impurity=True))
    # graph.format = "png"
    # graph.render("03cancer-dtree", view=False)

if __name__ == '__main__':
    main()
