import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
lr = pickle.load(open('model_save', 'rb'))

@app.route('/',methods=['GET','POST'])
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	'''
	for rendering results on HTML
	'''	
	print(request.form.values())
	features = [float(x) for x in request.form.values()]
	print(features)
	# re-arranging the list as per data set
	feature_list = ["grade","annual_inc","short_emp","emp_length_num","home_ownership","dti","purpose","term","last_delinq_none","revol_util","total_rec_late_fee","od_ratio"]
	

	prediction = lr.predict(np.array(features).reshape(1,-1))

	
	print("prediction value: ", prediction)

	result = ""
	if prediction == 0:
		result = "non defaulter"
	else:
		result = "defaulter"

	return  result#render_template('index.html', prediction_text = result) 


if __name__ == '__main__':
	app.run(debug=True)





