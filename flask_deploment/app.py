from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
app = Flask(__name__)
@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/prediction',methods=['POST'])
def prediction():
    name = request.form["name"]
    idd = 00000
    gender = str(request.form["gender"])
    age = float(request.form["age"])
    hypertension = int(request.form["hypertension"])
    heartdisease = int(int(request.form["heartdisease"]))
    evermarried = request.form["evermarried"]
    worktype = request.form["worktype"]
    residencetype = request.form["residencetype"]
    avg_glucose_level = float(request.form["avg_glucose_level"])
    bmi = float(request.form["bmi"])
    smoking_status = request.form["smokingstatus"]

    # Pickle Extraction
    with open('..\\data\\trainedstrokemodel.pkl','rb') as mf:
        pickle_file = pickle.load(mf)
    with open('..\\data\\trainedlabelencoders.pkl','rb') as ef:
        label_encoders = pickle.load(ef)
    encoder = label_encoders
    model = pickle_file['model']
    scaler = pickle_file['scaler']

    # Model Initialization and Pre-processing
    columns = ['id','gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']
    df = pd.DataFrame([[idd,gender,age,hypertension,heartdisease,evermarried,worktype,residencetype,avg_glucose_level,bmi,smoking_status]],columns=columns)
    dtype_dict = {'id':int,'age':float,'hypertension':int,'heart_disease':int,'avg_glucose_level':float,'bmi':float} 
    df = df.astype(dtype_dict)
    for col in df.select_dtypes(include=['object']):
        df[col] = encoder[col].transform(df[col])
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)
    dicto = {
    'gender':gender,
    'age':age,
    'hypertension':hypertension,
    'heartdisease':heartdisease,
    'ever_married':evermarried,
    'worktype':worktype,
    'residencetype':residencetype,
    'avg_glucose,level':avg_glucose_level,
    'bmi':bmi,
    'smoking_status' : smoking_status
    }

    return render_template('prediction.html',res=dicto,predicted=prediction)

if __name__ == "__main__":
    app.run(debug=True)