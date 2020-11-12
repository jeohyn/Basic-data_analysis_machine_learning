import warnings
warnings.filterwarnings(action='ignore')  #warning을 출력하지 않기 위한 코드
import numpy as np
from sklearn import tree, svm, linear_model, neighbors
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.cluster import AgglomerativeClustering, KMeans

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
y_pred_t=classifier_t.predict(X)
score_s=str(round(sum(cross_val_score(classifier_s, X,Y,cv=5))/5*100,1))

classifier_l=linear_model.LogisticRegression(random_state=0)
classifier_l=classifier_l.fit(X,Y)
score_l=str(round(sum(cross_val_score(classifier_l, X,Y,cv=5))/5*100,1))

classifier_k=neighbors.KNeighborsClassifier(n_neighbors=5)
classifier_k=classifier_k.fit(X,Y)
score_k=str(round(sum(cross_val_score(classifier_k, X,Y,cv=5))/5*100,1))

y_true=Y

y_pred_t=classifier_t.predict(X)
y_pred_s=classifier_s.predict(X)
y_pred_l=classifier_l.predict(X)
y_pred_k=classifier_k.predict(X)

cm_t=confusion_matrix(y_true, y_pred_t)
cm_s=confusion_matrix(y_true, y_pred_s)
cm_l=confusion_matrix(y_true, y_pred_l)
cm_k=confusion_matrix(y_true, y_pred_k)

ps_t=precision_score(y_true, y_pred_t, average=None)
ps_s=precision_score(y_true, y_pred_s, average=None)
ps_l=precision_score(y_true, y_pred_l, average=None)
ps_k=precision_score(y_true, y_pred_k, average=None)

re_t=recall_score(y_true, y_pred_t, average=None)
re_s=recall_score(y_true, y_pred_s, average=None)
re_l=recall_score(y_true, y_pred_l, average=None)
re_k=recall_score(y_true, y_pred_k, average=None)


while choice!=4:
    print("[ MENU ]")
    print("\n1.Evaluate classifiers\n2.Input the informatin about a wine\n3.Predict wine quality\n4.Cluster wines\n5.Quit")
    
    choice=input("메뉴 선택:")
    if choice=='1':
        print("[ Accuracy estimation ]")
        print("1. Decision tree: "+score_t+"%")
        print("2. Support vector machine: "+score_s+"%")
        print("3. Logistic regression:  "+score_l+"%")
        print("4. k-NN classifier:  "+score_k+"%\n")

        print("[ Confusion matrix ]")
        print("1. Decision tree")
        print(cm_t)
        print("2. Support vector machine")
        print(cm_s)
        print("3. Logistic regression")
        print(cm_l)
        print("4. k-NN classifier")
        print(cm_k)
        print("\n")

        print("[ Precision ]")
        print("1. Decision tree : " +str(ps_t))
        print("2. Support vector machine : " +str(ps_s))
        print("3. Logistic regression : "+ str(ps_l))
        print("4. k-NN classifier : "+str(ps_k))
        print("\n")

        print("[ Recall ]")
        print("1. Decision tree : "+str(re_t))
        print("2. Support vector machine : "+str(re_s))
        print("3. Logistic regression : "+str(re_l))
        print("4. k-NN classifier : "+str(re_k))
        print("\n")

    elif choice=='2':
        test=[]
        print("\n[ Wine information ]\n")
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
        choice=input("Select the algorithm ((h)ierarchical or (k)-means)) : ")
        c_num=int(input("Input the number of clusters : "))

        if choice=='h':
            model_h=AgglomerativeClustering(n_clusters=c_num)
            model_h.fit(X)
            model_h_r=model_h.labels_
            
            f_w_num=int(input("Input the number of the first wine : "))
            s_w_num=int(input("Input the number of the second wine : "))

            if model_h.labels_[f_w_num]==model_h.labels_[s_w_num] :
               print("Result: "+str(f_w_num)+" and "+str(s_w_num)+" are in the same cluster\n")
            else:
               print("Result: "+str(f_w_num)+" and "+str(s_w_num)+" are in the different cluster\n")

        elif choice=='k':
            model_k=KMeans(n_clusters=c_num, random_state=0)
            model_k.fit(X)
            model_k_r=model_k.labels_
            
            f_w_num=int(input("Input the number of the first wine : "))
            s_w_num=int(input("Input the number of the second wine : "))

            if model_k.labels_[f_w_num]==model_k.labels_[s_w_num]:
               print("Result: "+str(f_w_num)+" and "+str(s_w_num)+" are in the same cluster\n")
            else:
               print("Result: "+str(f_w_num)+" and "+str(s_w_num)+" are in the different cluster\n")

    elif choice=='5':
        break

    else:
        print("잘못된 입력")
        

            
