import numpy as np
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
choice=0
num=0
data=np.genfromtxt('./winequality-red.csv', dtype=np.float32, delimiter=";", skip_header=1)
X=data[:,0:11]
Y=data[:,11]
        
#round(num1,num2) : num1을 소수점아래 num2째자리까지 반올림하여 표현
classifier_t=tree.DecisionTreeClassifier(random_state=0)
classifier_t=classifier_t.fit(X,Y)
score_t=str(round(sum(cross_val_score(classifier_t, X,Y,cv=5))/5*100,1))

classifier_s=svm.SVC(random_state=0)
classifier_s=classifier_s.fit(X,Y)
score_s=str(round(sum(cross_val_score(classifier_s, X,Y,cv=5))/5*100,1))

classifier_l=linear_model.LogisticRegression(random_state=0)
classifier_l=classifier_l.fit(X,Y)
score_l=str(round(sum(cross_val_score(classifier_l, X,Y,cv=5))/5*100,1))

classifier_k=neighbors.KNeighborsClassifier(n_neighbors=5)
classifier_k=classifier_k.fit(X,Y)
score_k=str(round(sum(cross_val_score(classifier_k, X,Y,cv=5))/5*100,1))

while choice!=4:
    print("[ MENU ]")
    print("\n1.Estimate the accuracy of classifiers\n2.Input the informatin about a wine\n3.Predict wine quality\n4.Quit")
    
    choice=input("메뉴 선택:")
    if choice=='1':
        print("[ Accuracy estimation ]")
        print("1. Decision tree: "+score_t+"%")
        print("2. Support vector machine: "+score_s+"%")
        print("3. Logistic regression:  "+score_l+"%")
        print("4. k-NN classifier:  "+score_k+"%")

    elif choice=='2':
        test=[]
        fix=float(input("1. fixed acidity :"))
        vol=float(input("2. volatiled acidity :"))
        cit=float(input("3. citric acid :"))
        res=float(input("4. residual sugar:"))
        chl=float(input("5. chlorides:"))
        free=float(input("6. free sultur dioxide:"))
        total=float(input("7. total sulfur dioxide:"))
        den=float(input("8. density:"))
        ph=float(input("9. pH:"))
        sul=float(input("10. sulphates:"))
        alc=float(input("11. alchol:"))

    elif choice=='3':
        test=[[fix, vol, cit, res, chl, free, total, den, ph, sul, alc]]
        print("[ Predicted wine quality]")
        predicted_class_tr=classifier_t.predict(test)
        predicted_class_sr=classifier_s.predict(test)
        predicted_class_lr=classifier_l.predict(test)
        predicted_class_kr=classifier_k.predict(test)

        print("1. Decision tree: %.1f" %predicted_class_tr)
        print("2. Support vector machine:%.1f" %predicted_class_sr)
        print("3. Logistic regression:%.1f" %predicted_class_lr)
        print("4. k-NN classifier:%.1f" %predicted_class_kr)

    elif choice=='4':
        break

    else:
        print("잘못된 입력")
        

            
