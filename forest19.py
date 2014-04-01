""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September, 2012
please see packages.python.org/milk/randomforests.html for more

""" 

def train():

    import pdb
    import copy
    import numpy as np
    import csv as csv
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.grid_search import GridSearchCV
    from sklearn import svm    
    from sklearn import tree, ensemble
    from sklearn.cross_validation import cross_val_score
    from sklearn import cross_validation
    from sklearn.metrics import classification_report
    from sklearn import metrics
    
    
    csv_file_object = csv.reader(open('csv/train.csv', 'rb')) #Load in the training csv file
    header = csv_file_object.next() #Skip the fist line as it is a header
    #reshape to put survived at the end
    header.append(header.pop(1))
    train_data=[] #Create a variable called 'train_data'
    for row in csv_file_object: #Skip through each row in the csv file
        row.append(row.pop(1))
        train_data.append(row) #adding each row to the data variable
    train_data = np.array(train_data) #Then convert from a list to an array
    
    test_file_object = csv.reader(open('csv/test.csv', 'rb')) #Load in the test csv file
    test_header = test_file_object.next() #Skip the fist line as it is a header
    test_data=[] #Create a variable called 'test_data'
    for row in test_file_object: #Skip through each row in the csv file
        row.append("")
        test_data.append(row) #adding each row to the data variable
    test_data = np.array(test_data) #Then convert from a list to an array
    
    
    n_train = len(train_data)
    n_test = len(test_data)

    #combine both data sets to do a data munging    
    joined_data =np.concatenate((train_data,test_data), axis = 0)
    
    
    #normalize class
    joined_data[joined_data[0::,1] == '1',1] = 1.0
    joined_data[joined_data[0::,1] == '2',1] = 2.0
    joined_data[joined_data[0::,1] == '3',1] = 3.0
    
    #I need to convert all strings to integer classifiers:
    #Male = 1, female = 0:   
    joined_data[joined_data[0::,3]=='male',3] = 1.0
    joined_data[joined_data[0::,3]=='female',3] = 0.0
    
    #embark c=0, s=1, q=2
    joined_data[joined_data[0::,10] =='C',10] = 1.0
    joined_data[joined_data[0::,10] =='S',10] = 2.0
    joined_data[joined_data[0::,10] =='Q',10] = 3.0
      
    
    #Create 2 more columns from name: family name, title
    header.insert(11,"Family name")
    header.insert(12,"Title")
    joined_data = np.insert(joined_data, 11, values = 0,axis = 1)
    joined_data = np.insert(joined_data, 12, values = 0,axis = 1)
    
    test_header.extend(["Family name","Title"])
    test_data = np.append(test_data,test_data[:,[0,1]],1)
    

    last_names = {}
    tickets = {}
    titles = {"Mr":0,"Mrs":1,"Miss":2,"Master":3}
    name_count = 0
    ticket_count = 0
    
    for i in range(joined_data.shape[0]):
        name = joined_data[i,2]
        val = name.replace(".",",").split(",")
        if val[1].strip() == "Don" or val[1].strip() == "Rev" or val[1].strip() == "Dr"  or \
                                      val[1].strip() == "Major"  or val[1].strip() == "Sir" or \
                                      val[1].strip() == "Col"   or val[1].strip() == "Capt" or \
                                      val[1].strip() == "Jonkheer": val[1] = "Mr"
        if val[1].strip() == "Mme" or val[1].strip() == "Ms"  or val[1].strip() == "Lady" or \
                                      val[1].strip() == "the Countess"  or val[1].strip() == "Dona": val[1] = "Mrs"
        if val[1].strip() == "Mlle": val[1] = "Miss"  
        
        if val[0].strip() not in last_names.keys():
            last_names[val[0].strip()] = name_count
            name_count += 1
        #titles
        joined_data[i,12] = titles[val[1].strip()]#/3.0
        #ticket
        if joined_data[i,7] not in tickets.keys():
            tickets[joined_data[i,7]] = ticket_count
            ticket_count += 1    
    
    for i in range(joined_data.shape[0]):
        name = joined_data[i,2]
        val = name.replace(".",",").split(",")     
        joined_data[i,11] = last_names[val[0].strip()]
        joined_data[i,7] = tickets[joined_data[i,7]]
   
    

    #I need to fill in the gaps of the data and make it complete.
    #So where there is no price, I will assume price on median of that class
    #Where there is no age I will give median of all ages

    #All the ages with no data. make a prediction based on the passengers with the same sex, number of sibsp, parch, family_name and title
    #decision to predict age
    age_exist = joined_data[(joined_data[:,4] != ''),:]
    age_exist = age_exist[:,[4,3,5,6,12]].astype('float32')
    age_missing = joined_data[(joined_data[:,4] == ''),:]
    age_missing = age_missing[:,[3,5,6,12]].astype('float32')
    
    n_age = len(age_exist)
    
    print 'Training missing age values'

    x_age = age_exist[0::,1:]
    y_age = age_exist[0::,0]

   
    
    #K-fold cross-validation
    
    reg = tree.DecisionTreeRegressor(random_state = 0)
    reg = ensemble.RandomForestRegressor(n_estimators = 100, random_state = 0)
    #scores = cross_val_score(reg, x_age, y_age, cv=10, scoring='mean_squared_error')
    
    #output = reg.predict(train_age_missing)
    
    
    
    
    kf = cross_validation.KFold(n=n_age, n_folds=10)
    scores = []
    for train_index, test_index in kf:
        x_train, x_test = x_age[train_index], x_age[test_index]
        y_train, y_test = y_age[train_index], y_age[test_index]
        reg.fit(x_train,y_train)
        y_pred = reg.predict(x_test)
        scores.append(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
    
    scores = np.array(scores)
    print scores
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    

    # Set the parameters by cross-validation
    tuned_parameters = {'max_features': [3],
                     'max_depth': [4],
                    'min_samples_split': [2],
                    'min_samples_leaf': [5]}

    
    print("# Tuning hyper-parameters for MSE")
    print()

    clf = GridSearchCV(reg, tuned_parameters, cv=10, scoring='mean_squared_error')
    clf.fit(x_age, y_age)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (np.sqrt(-mean_score), np.sqrt(-scores).std(), params))
    print()

    

    print 'Predicting missing age values'

    output = reg.predict(age_missing)

    joined_data[(joined_data[:,4] == ''),4] = ["%.1f" %x for x in output]

    
    #All missing ebmbarks just make them embark from most common place among the passengers of the same class

    for i in range(joined_data.shape[0]):
        if joined_data[i,10] == "":
            joined_data[i,10] = np.round(np.mean(joined_data[(joined_data[0::,10]\
                                                       != '') * (joined_data[0::,1] == joined_data[i,1]),10].astype(np.float)))
  
 
    #All the missing prices assume median of their respective class
    for i in xrange(np.size(joined_data[0::,0])):
        if joined_data[i,8] == '':
            joined_data[i,8] = np.median(joined_data[(joined_data[0::,8] != '') &\
                                                 (joined_data[0::,0] == joined_data[i,1])\
                ,8].astype(np.float))   


    with open("csv/train_data_work.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n_train):
            writer.writerow(joined_data[i,:])
            
    with open("csv/test_data_work.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(header[0:-1])
        for i in range(n_test):
            writer.writerow(joined_data[n_train+i,0:-1])

    #dum = [header.pop(i) for i in [2,7,9,11,12]]
    joined_data = np.delete(joined_data,[0,2,7,8,9,10,11],1) #remove the id, name, cabin,ticket, price, family_name


    #The data is now ready to go. So lets train then test!
    
    # Here we will train two random forests, once for each sex
    
    train_data = joined_data[0:n_train,:]
    test_data = joined_data[n_train:,0:-1]
    

    print 'Training '
    forest_m = RandomForestClassifier(n_estimators=20, random_state = 0)
    forest_f = RandomForestClassifier(n_estimators=20, random_state = 0)
    forest = RandomForestClassifier(n_estimators=20, random_state = 0)
    #forest = ExtraTreesClassifier(n_estimators=500, random_state = 0)
    #forest = DecisionTreeClassifier(random_state = 0)    
    #forest = GradientBoostingClassifier(n_estimators=500, random_state = 0, learning_rate=0.5, max_depth=1)
    #forest = SGDClassifier() 
    #forest = svm.SVC()  

    
    #split train data into two subsets
    train_data_m = np.delete(train_data[(train_data[:,1] == '1.0'),:],[1],1).astype('float32')
    train_data_f = np.delete(train_data[(train_data[:,1] == '0.0'),:],[1],1).astype('float32')
    
    # First train a tree for males
       
    # k-fold cross-validation
    
    score_m = cross_val_score(forest_m, train_data_m[:,0:-1],train_data_m[:,-1], cv=10)
          
    print "Cross-validation error (males): ", np.mean(score_m)

    score_f = cross_val_score(forest_m, train_data_f[:,0:-1],train_data_f[:,-1], cv=10)
          
    print "Cross-validation error (females): ", np.mean(score_f)
        
 
    print "Cross-validation error (total): ", (np.mean(score_f)*len(train_data_f) + np.mean(score_m)*len(train_data_m))/(len(train_data_f) + len(train_data_m))   
 

    score = cross_val_score(forest, train_data[:,0:-1],train_data[:,-1], cv=10)
          
    print "Cross-validation error (all): ", np.mean(score)
 
 
    # Set the parameters by cross-validation
    tuned_parameters = {'n_estimators':[70],
                    'max_features': [3],
                    'max_depth': [4],
                    'min_samples_split': [1],
                    'min_samples_leaf': [1]}

    
    print("# Tuning hyper-parameters for score")
    print()

    clf = GridSearchCV(forest, tuned_parameters, cv=20)
    clf.fit(train_data[:,0:-1], train_data[:,-1])


    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std(), params))
    print()
 
 
    # Set the parameters by cross-validation
    tuned_parameters = {'n_estimators':[70],
                    'max_features': [3],
                    'max_depth': [3],
                    'min_samples_split': [1],
                    'min_samples_leaf': [1]}

    
    print("# Tuning hyper-parameters for score")
    print()

    clf_m = GridSearchCV(forest_m, tuned_parameters, cv=20)
    clf_m.fit(train_data_m[:,0:-1], train_data_m[:,-1])

    print("Best parameters set found on development set:")
    print()
    print(clf_m.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf_m.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std(), params))
    print()
    
        # Set the parameters by cross-validation
    tuned_parameters = {'n_estimators':[50],
                    'max_features': [3],
                    'max_depth': [3],
                    'min_samples_split': [1],
                    'min_samples_leaf': [1]}

    
    print("# Tuning hyper-parameters for score")
    print()

    clf_f = GridSearchCV(forest_f, tuned_parameters, cv=20)
    clf_f.fit(train_data_f[:,0:-1], train_data_f[:,-1])

    print("Best parameters set found on development set:")
    print()
    print(clf_f.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf_f.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std(), params))
    print()
    
    

    
    print 'Final predicting'


    forest_m = forest_m.fit(train_data_m[0::,0:-1],\
                        train_data_m[0::,-1])
                        
    forest_f = forest_f.fit(train_data_f[0::,0:-1],\
                        train_data_f[0::,-1])                       

    
                        
    #split test data into two subsets
    test_data_m = np.delete(test_data[(test_data[:,1] == '1.0'),:],[1],1).astype('float32')
    test_data_f = np.delete(test_data[(test_data[:,1] == '0.0'),:],[1],1).astype('float32')                        
                                             
    output_m = clf_m.predict(test_data_m)
    output_f = clf_f.predict(test_data_f)


 
    #restore output
    output = test_data[:,0] # initialize
    output[(test_data[:,1] == '1.0')] = output_m.astype('uint8')
    output[(test_data[:,1] == '0.0')] = output_f.astype('uint8') 

    output = clf.predict(test_data)
    
    #get train error

    #train_output = forest.predict(train_data[:,1::])

    #print " my score: ", sum(train_data[:,0] == train_output)/float(train_data.shape[0])

    #mean_accuracy = forest.score(train_data[:,1::],train_data[0::,0])

    #print "mean accuracy train data: ", mean_accuracy
    open_file_object = csv.writer(open("csv/forest19.csv", "wb"))
    test_file_object = csv.reader(open('csv/test.csv', 'rb')) #Load in the csv file


    test_file_object.next()
    open_file_object.writerow(["PassengerId","Survived"])
    i = 0
    for row in test_file_object:
        open_file_object.writerow([row[0],output[i]])
        i += 1



if __name__ == '__main__':
    train()