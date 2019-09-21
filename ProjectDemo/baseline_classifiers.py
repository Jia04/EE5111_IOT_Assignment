from sklearn import svm, tree, naive_bayes,  ensemble, neighbors
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split

import numpy as np

def svm_classifier(df, label, runs):

    clf_svm = svm.SVC(kernel='linear', probability=True)
    scores_svm_list = []
    cm_svm_list = []
    auc_svm_list = []
    f1_svm_list = []
    cv_svm_mean_list = []

    # svm
    for i in range(0, runs):
        print ("****************************************************************This is run: ",i)
        train_data, test_data, train_label, test_label = train_test_split(df, label, test_size=0.20)

        clf_svm.fit(train_data, train_label)
        scores_svm = clf_svm.score(test_data, test_label)
        pred_svm = clf_svm.predict(test_data)

        cm_svm = confusion_matrix(test_label, pred_svm)

        prob_svm = clf_svm.predict_proba(test_data)
        auc_svm = roc_auc_score(test_label, prob_svm[:,1], average='micro')

        f1_svm = f1_score(pred_svm, test_label, average='weighted')


        # svm lists
        scores_svm_list.append(scores_svm)
        auc_svm_list.append(auc_svm)
        f1_svm_list.append(f1_svm)
        cm_svm_list.append(cm_svm)

    cv_value = StratifiedKFold(n_splits=5, shuffle=True)

    # average confusion matrix
    svm_0_0 = []
    svm_0_1 = []
    svm_1_0 = []
    svm_1_1 = []
    for j in range(0, runs):
        svm_0_0.append(cm_svm_list[j][0][0])
        svm_0_1.append(cm_svm_list[j][0][1])
        svm_1_0.append(cm_svm_list[j][1][0])
        svm_1_1.append(cm_svm_list[j][1][1])
        a = round(np.mean(svm_0_0), 2)
        b = round(np.mean(svm_0_1), 2)
        c = round(np.mean(svm_1_0), 2)
        d = round(np.mean(svm_1_1), 2)
        svm_cnf = np.array([[a, b], [c, d]])

    print ("svm accuracy: ", np.mean(scores_svm_list))
    print ('StdDev accuracy is: ', np.std(np.asarray(scores_svm_list)))
    print ("svm AUC: ", np.mean(auc_svm_list))
    print ('StdDev AUC is: ', np.std(np.asarray(auc_svm_list)))
    print ("svm f1 score: ", np.mean(f1_svm_list))
    print ('StdDev f1 is: ', np.std(np.asarray(f1_svm_list)))
    print ("svm confusion matrix: \n", svm_cnf)
    print ("svm cv: ", np.mean(cross_val_score(clf_svm, df, label, cv=cv_value)))

def dt_classifier(df, label, runs):

    clf_dt = tree.DecisionTreeClassifier(criterion='entropy', splitter='random')
    scores_dt_list = []
    cm_dt_list = []
    auc_dt_list = []
    f1_dt_list = []
    cv_dt_mean_list = []

    # dt
    for i in range(0, runs):
        print ("****************************************************************This is run: ",i)
        train_data, test_data, train_label, test_label = train_test_split(df, label, test_size=0.20)
        clf_dt.fit(train_data, train_label)
        scores_dt = clf_dt.score(test_data, test_label)
        pred_dt = clf_dt.predict(test_data)

        cm_dt = confusion_matrix(test_label, pred_dt)

        prob_dt = clf_dt.predict_proba(test_data)
        auc_dt = roc_auc_score(test_label, prob_dt[:,1], average='micro')

        f1_dt = f1_score(pred_dt, test_label, average='weighted')

        # dt lists
        scores_dt_list.append(scores_dt)
        auc_dt_list.append(auc_dt)
        f1_dt_list.append(f1_dt)
        cm_dt_list.append(cm_dt)

    cv_value = StratifiedKFold(n_splits=5, shuffle=True)

    # average confusion matrix
    dt_0_0 = []
    dt_0_1 = []
    dt_1_0 = []
    dt_1_1 = []
    for j in range(0, runs):
        dt_0_0.append(cm_dt_list[j][0][0])
        dt_0_1.append(cm_dt_list[j][0][1])
        dt_1_0.append(cm_dt_list[j][1][0])
        dt_1_1.append(cm_dt_list[j][1][1])
        a = round(np.mean(dt_0_0), 2)
        b = round(np.mean(dt_0_1), 2)
        c = round(np.mean(dt_1_0), 2)
        d = round(np.mean(dt_1_1), 2)
        dt_cnf = np.array([[a, b], [c, d]])

    print ("dt accuracy: ", np.mean(scores_dt_list))
    print ('StdDev accuracy is: ', np.std(np.asarray(scores_dt_list)))
    print ("dt AUC: ", np.mean(auc_dt_list))
    print ('StdDev AUC is: ', np.std(np.asarray(auc_dt_list)))
    print ("dt f1 score: ", np.mean(f1_dt_list))
    print ('StdDev f1 is: ', np.std(np.asarray(f1_dt_list)))
    print ("dt confusion matrix: \n", dt_cnf)
    print ("dt cv: ", np.mean(cross_val_score(clf_dt, df, label, cv=cv_value)))

def gb_classifier(df, label, runs):

    clf_gb = ensemble.GradientBoostingClassifier(loss='exponential', learning_rate=0.2, n_estimators=150)
    scores_gb_list = []
    cm_gb_list = []
    auc_gb_list = []
    f1_gb_list = []
    cv_gb_mean_list = []

    # gb
    for i in range(0, runs):
        print ("****************************************************************This is run: ",i)
        train_data, test_data, train_label, test_label = train_test_split(df, label, test_size=0.20)
        clf_gb.fit(train_data, train_label)
        scores_gb = clf_gb.score(test_data, test_label)
        pred_gb = clf_gb.predict(test_data)

        cm_gb = confusion_matrix(test_label, pred_gb)

        prob_gb = clf_gb.predict_proba(test_data)
        auc_gb = roc_auc_score(test_label, prob_gb[:,1], average='micro')

        f1_gb = f1_score(pred_gb, test_label, average='weighted')

        # gb lists
        scores_gb_list.append(scores_gb)
        auc_gb_list.append(auc_gb)
        f1_gb_list.append(f1_gb)
        cm_gb_list.append(cm_gb)

    cv_value = StratifiedKFold(n_splits=5, shuffle=True)

    # average confusion matrix
    gb_0_0 = []
    gb_0_1 = []
    gb_1_0 = []
    gb_1_1 = []
    for j in range(0, runs):
        gb_0_0.append(cm_gb_list[j][0][0])
        gb_0_1.append(cm_gb_list[j][0][1])
        gb_1_0.append(cm_gb_list[j][1][0])
        gb_1_1.append(cm_gb_list[j][1][1])
        a = round(np.mean(gb_0_0), 2)
        b = round(np.mean(gb_0_1), 2)
        c = round(np.mean(gb_1_0), 2)
        d = round(np.mean(gb_1_1), 2)
        gb_cnf = np.array([[a, b], [c, d]])

    print ("gb accuracy: ", np.mean(scores_gb_list))
    print ('StdDev accuracy is: ', np.std(np.asarray(scores_gb_list)))
    print ("gb AUC: ", np.mean(auc_gb_list))
    print ('StdDev AUC is: ', np.std(np.asarray(auc_gb_list)))
    print ("gb f1 score: ", np.mean(f1_gb_list))
    print ('StdDev f1 is: ', np.std(np.asarray(f1_gb_list)))
    print ("gb confusion matrix: \n", gb_cnf)
    print ("gb cv: ", np.mean(cross_val_score(clf_gb, df, label, cv=cv_value)))

def knn_classifier(df, label, runs):

    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
    scores_knn_list = []
    cm_knn_list = []
    auc_knn_list = []
    f1_knn_list = []
    cv_knn_mean_list = []

    # knn
    for i in range(0, runs):
        print ("****************************************************************This is run: ",i)
        train_data, test_data, train_label, test_label = train_test_split(df, label, test_size=0.20)
        clf_knn.fit(train_data, train_label)
        scores_knn = clf_knn.score(test_data, test_label)
        pred_knn = clf_knn.predict(test_data)

        cm_knn = confusion_matrix(test_label, pred_knn)

        prob_knn = clf_knn.predict_proba(test_data)
        auc_knn = roc_auc_score(test_label, prob_knn[:,1], average='micro')

        f1_knn = f1_score(pred_knn, test_label, average='weighted')

        # knn lists
        scores_knn_list.append(scores_knn)
        auc_knn_list.append(auc_knn)
        f1_knn_list.append(f1_knn)
        cm_knn_list.append(cm_knn)

    cv_value = StratifiedKFold(n_splits=5, shuffle=True)

    # average confusion matrix
    knn_0_0 = []
    knn_0_1 = []
    knn_1_0 = []
    knn_1_1 = []
    for j in range(0, runs):
        knn_0_0.append(cm_knn_list[j][0][0])
        knn_0_1.append(cm_knn_list[j][0][1])
        knn_1_0.append(cm_knn_list[j][1][0])
        knn_1_1.append(cm_knn_list[j][1][1])
        a = round(np.mean(knn_0_0), 2)
        b = round(np.mean(knn_0_1), 2)
        c = round(np.mean(knn_1_0), 2)
        d = round(np.mean(knn_1_1), 2)
        knn_cnf = np.array([[a, b], [c, d]])

    print ("knn accuracy: ", np.mean(scores_knn_list))
    print ('StdDev accuracy is: ', np.std(np.asarray(scores_knn_list)))
    print ("knn AUC: ", np.mean(auc_knn_list))
    print ('StdDev AUC is: ', np.std(np.asarray(auc_knn_list)))
    print ("knn f1 score: ", np.mean(f1_knn_list))
    print ('StdDev f1 is: ', np.std(np.asarray(f1_knn_list)))
    print ("knn confusion matrix: \n", knn_cnf)
    print ("knn cv: ", np.mean(cross_val_score(clf_knn, df, label, cv=cv_value)))
