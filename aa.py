from flask import Flask,render_template,request,redirect,url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import csv    
from geopy.geocoders import Nominatim
from geopy.distance import distance
from geopy.distance import geodesic

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('start.html')


@app.route('/login',methods=['POST','GET'])
def login():
    return render_template('login.html')

@app.route('/register',methods=['POST','GET'])
def register():
    return render_template('register.html')

@app.route("/intro",methods=['POST','GET'])
def intro():
    return render_template('intro.html')

@app.route('/profile',methods=['POST','GET'])
def profile():
    return render_template('profile.html')

@app.route("/check",methods=['POST','GET'])
def check():
    return render_template('check.html')


@app.route("/hi",methods=['POST','GET'])
def hi():
    return render_template('hi.html')

@app.route("/display",methods=['POST','GET'])
def display():
    d={}
    hi={}
    if request.method =='POST':
        preg=request.form.get("pregnancy")
        d['Pregnancies']=int(preg)
        glu=request.form.get("glucose")
        d['Glucose']=int(glu)
        blood=request.form.get("blood")
        d['BloodPressure']=int(blood)
        skin=request.form.get("skin")
        d['SkinThickness']=int(skin)  
        inslu=request.form.get("insulin")
        d['Insulin']=int(inslu)
        hei=request.form.get("height")
        hei=int(hei)
        wei=request.form.get("weight")
        wei=int(wei)
        bmi=wei/(hei*hei)
        r=float(bmi)
        d['BMI']=r
        pedi=request.form.get("ped")
        d['DiabetesPedigreeFunction']=int(pedi)
        age=request.form.get("age")
        e=int(age)
        d['Age']=e

        hi['Age']=e
        gen=request.form.get("gender")
        b=int(gen)
        hi['Sex']=b
        chol=request.form.get("highchol")
        hi['HighChol']=int(chol)
        checkchol=request.form.get("CholCheck")
        hi['CholCheck']=int(checkchol)
        hi['BMI']=r
        smokie=request.form.get('smoki')
        hi['Smoker']=int(smokie)
        phy=request.form.get('physical')
        hi['PhysActivity']=int(phy)
        fruit=request.form.get('fruit')
        hi['Fruits']=int(fruit)
        vege=request.form.get('vege')
        hi['Veggies']=int(vege)
        drink=request.form.get('drinker')
        hi['HvyAlcoholConsump']=int(drink)
        health=request.form.get('health')
        hi['GenHlth']=int(health)
        stress=request.form.get('stress')
        hi['MentHlth']=int(stress)
        phyill=request.form.get('phy ill')
        hi['PhysHlth']=int(phyill)
        diffwlk=request.form.get('diffwalk')
        hi['DiffWalk']=int(diffwlk)
        bp=request.form.get('highbp')
        hi['HighBP']=int(bp)
        
        

        d1=pd.read_csv('diabetesdata.csv')
        d1=d1.dropna()
        y1=d1['Outcome']
        x1=d1.iloc[:,1:9]
        x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.3)
        knn1= KNeighborsClassifier(n_neighbors = 28)
        gnb1=GaussianNB()
        svm1=SVC()
        knn1.fit(x1,y1)
        gnb1.fit(x1,y1)
        svm1.fit(x1,y1)
        est1=[('knn',knn1),('gnb',gnb1),('svm',svm1)]
        model1 = VotingClassifier(estimators = est1, voting ='hard')
        model1.fit(x1,y1)
        d=pd.DataFrame(d,index=[0])
        pred1=model1.predict(d)
        knn1.fit(x1_train,y1_train)
        gnb1.fit(x1_train,y1_train)
        svm1.fit(x1_train,y1_train)
        est1=[('knn',knn1),('gnb',gnb1),('svm',svm1)]
        model1 = VotingClassifier(estimators = est1, voting ='hard')
        model1.fit(x1_train,y1_train)
        accpred1=model1.predict(x1_test)
        acc1=metrics.accuracy_score(accpred1,y1_test)*100
        cm1=metrics.confusion_matrix(accpred1,y1_test)
        pipe='No chance for diabetes'
        vv=' %.2f'%acc1
        hi['Diabetes']=int(pred1[0])
        d=pd.read_csv('heartattackdataa.csv')
        d=d.dropna()
        x=d.iloc[:,[1,2,3,4,5,6,8,9,10,11,12,13,14,15,17,18]]
        y=d['HeartDiseaseorAttack']
        z=d['Stroke']
        rs=RandomOverSampler()
        xrs,yrs=rs.fit_resample(x,y)
        x_train,x_test,y_train,y_test=train_test_split(xrs,yrs,test_size=0.3,random_state=1)
        knn= KNeighborsClassifier(n_neighbors = 266)
        gnb=GaussianNB()
        svm=SVC()
        hi=pd.DataFrame(hi,index=[0])
        knn.fit(x_train,y_train)
        gnb.fit(x_train,y_train)
        svm.fit(x_train,y_train)
        est=[('knn',knn),('gnb',gnb),('svm',svm)]
        modelheart = VotingClassifier(estimators = est, voting ='hard')
        modelheart.fit(x_train,y_train)
        predheart=modelheart.predict(hi) 
        accpred=modelheart.predict(x_test)
        acch=metrics.accuracy_score(accpred,y_test)*100
        cmh=metrics.confusion_matrix(accpred,y_test)
        ll=' %.2f'%acch
        xrs,zrs=rs.fit_resample(x,z)
        x_train,x_test,z_train,z_test=train_test_split(xrs,zrs,test_size=0.3,random_state=1) 
        knn.fit(x_train,z_train)
        gnb.fit(x_train,z_train)
        svm.fit(x_train,z_train)
        est=[('knn',knn),('gnb',gnb),('svm',svm)]
        modelstroke = VotingClassifier(estimators = est, voting ='hard')
        modelstroke.fit(x_train,z_train)
        predstroke=modelheart.predict(hi)
        accpred=modelstroke.predict(x_test)
        acc=metrics.accuracy_score(accpred,z_test)*100
        cm=metrics.confusion_matrix(accpred,z_test)
        kk=' %.2f'%acc
        hi['Diabetes']=int(pred1[0])
        if pred1==1 and predheart==1 and predstroke==1:
            return render_template('display.html',dis=vv,c=cm1,dis1=kk,c1=cm,dis2=ll,c2=cmh)
        elif pred1==1 and predheart==1:
            return render_template('display.html',dis=vv,c=cm1,dis2=ll,c2=cmh)
        elif pred1==1 and predstroke==1:
            return render_template('display.html',dis=vv,c=cm1,dis1=kk,c1=cm)
        elif predstroke==1 and predheart==1:
            return render_template('display.html',dis1=kk,c1=cm,dis2=ll,c2=cmh)
        elif pred1==1 :
            return render_template('display.html',dis=vv,c=cm1)
        elif predheart==1 :
            return render_template('display.html',dis2=ll,c2=cmh)
        elif predstroke==1 :
            return render_template('display.html',dis1=kk,c1=cm)
        
    
        
       
@app.route('/lock1')
def lock1():
    return render_template('lock1.html')

@app.route("/locke",methods=['POST','GET'])
def locke():
    if request.method =='POST':
        dis=request.form.get("disease")
        cityy=request.form.get("city")
        df=pd.read_csv("doctors.csv",encoding='windows-1254')
        grp = df.groupby('FIELD')
        
        if dis=="DIABETES":
            a=grp.get_group('DIABETES')
            def city_distance(city1, city2):
                geolocator = Nominatim(user_agent="geoapiExercises")
                location1 = geolocator.geocode(city1)
                location2 = geolocator.geocode(city2)
                return geodesic((location1.latitude, location1.longitude), (location2.latitude, location2.longitude)).km
            
            min_distance = float('inf')
            min_distance_city = None

            # iterate through the rows in the csv
            for index, row in a.iterrows():
                city = row['LOCATION']
                distance = city_distance(cityy, city)
                if distance < min_distance:
                    min_distance = distance
                    min_distance_city = city
                    min_distance_row = row
        
            target_city = min_distance_city 
            for index, row in a.iterrows():
                if row['LOCATION'] == target_city:
                    a="VISIT :\t"+row["NAME"]+" , "+row["ADDRESS"]
                    return render_template('locke.html',lo=a)
        
        elif dis=="STROKE":
            a=grp.get_group('STROKE')

            def city_distance(city1, city2):
                geolocator = Nominatim(user_agent="geoapiExercises")
                location1 = geolocator.geocode(city1)
                location2 = geolocator.geocode(city2)
                return geodesic((location1.latitude, location1.longitude), (location2.latitude, location2.longitude)).km


            

            # initialize variables for minimum distance and city
            min_distance = float('inf')
            min_distance_city = None

            # iterate through the rows in the csv
            for index, row in a.iterrows():
                city = row['LOCATION']
                distance = city_distance(cityy, city)
                if distance < min_distance:
                    min_distance = distance
                    min_distance_city = city
                    min_distance_row = row
        
            target_city = min_distance_city 
            for index, row in a.iterrows():
                if row['LOCATION'] == target_city:
                    b="VISIT :\t"+row["NAME"]+" , "+row["ADDRESS"]
                    return render_template('locke.html',lol=b)
        else:
            a=grp.get_group('HEART DISEASE')

            def city_distance(city1, city2):
                geolocator = Nominatim(user_agent="geoapiExercises")
                location1 = geolocator.geocode(city1)
                location2 = geolocator.geocode(city2)
                return geodesic((location1.latitude, location1.longitude), (location2.latitude, location2.longitude)).km


            

            # initialize variables for minimum distance and city
            min_distance = float('inf')
            min_distance_city = None

            # iterate through the rows in the csv
            for index, row in a.iterrows():
                city = row['LOCATION']
                distance = city_distance(cityy, city)
                if distance < min_distance:
                    min_distance = distance
                    min_distance_city = city
                    min_distance_row = row
        
            target_city = min_distance_city 
            for index, row in a.iterrows():
                if row['LOCATION'] == target_city:
                    c="VISIT :\t"+row["NAME"]+" , "+row["ADDRESS"]
                    return render_template('locke.html',loll=c)

@app.route('/lock2')
def lock2():
    return render_template('lock2.html')

@app.route("/lock",methods=['POST','GET'])
def lock():
    if request.method =='POST':
        geolocator = Nominatim(user_agent="myGeocoder")

        # Get the latitude and longitude of two cities
        dis=request.form.get("disease")
        cityy=request.form.get("city")
        loc1 = geolocator.geocode(cityy)
        lat1, lon1 = loc1.latitude, loc1.longitude

        if(dis=="diabetes"):   
            loc2 = geolocator.geocode("chennai")
            lat2, lon2 = loc2.latitude, loc2.longitude

            loc3 = geolocator.geocode("tuticorin")
            lat3, lon3 = loc3.latitude, loc3.longitude

            loc4 = geolocator.geocode("pondicherry")
            lat4, lon4 = loc4.latitude, loc4.longitude

        
            d1 = distance((lat1, lon1), (lat2, lon2)).km
            d2 = distance((lat1, lon1), (lat3, lon3)).km
            d3 = distance((lat1, lon1), (lat4, lon4)).km
            a="vist dr.Mohanlal diabetes at chennai"
            b="visit narayan at tuticorin"
            c="visit government hospital of pondicherry"
            if d1<d2 and d1<d3:
                
                return render_template('lock.html',lo=a)
            elif d2<d1 and d2<d3:
                return render_template('lock.html',lo=b)
            else:
                return render_template('lock.html',lo=c)
        elif(dis=="strokes"):
                loc4 = geolocator.geocode("kovai")
                lat5, lon5 = loc4.latitude, loc4.longitude

                loc5 = geolocator.geocode("trichy")
                lat6, lon6 = loc5.latitude, loc5.longitude

                

        
                d4 = distance((lat1, lon1), (lat5, lon5)).km
                d5 = distance((lat1, lon1), (lat6, lon6)).km
                d="vist The KG Medical Trust ,Kovai"
                e="visit Kauvery hospital at trichy"
                if d4<d5 :
                    return render_template('lock.html',lol=d)
                else :
                    return render_template('lock.html',lol=e)
        elif(dis=="heart attack"):
                loc6 = geolocator.geocode("bangalore")
                lat7, lon7 = loc6.latitude, loc6.longitude
                loc7 = geolocator.geocode("chennai")
                lat8, lon8 = loc7.latitude, loc7.longitude
                loc8 = geolocator.geocode("kovai")
                lat9, lon9 = loc8.latitude, loc8.longitude


                d6 = distance((lat1, lon1), (lat7, lon7)).km
                d7 = distance((lat1, lon1), (lat8, lon8)).km
                d8 = distance((lat1, lon1), (lat9, lon9)).km
                f="Sri Sathya Sai Central Trust,Bangalore"
                g="visit government hospital,chennai"
                h="visit kovai heart foundation"
                if d6<d7 and d6<d8 :
                    return render_template('lock.html',loll=f)
                elif d7<d6 and d7<d8 :
                    return render_template('lock.html',loll=g)
                else:
                    return render_template('lock.html',loll=h)  
        else:
            print("you're fine")


if __name__ == "__main__":
    app.run(debug=True)