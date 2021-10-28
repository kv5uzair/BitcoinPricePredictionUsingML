from flask import Flask, render_template, request
import pickle
from flask.templating import _default_template_ctx_processor
import numpy as np

filename = 'model.pkl'
model = pickle.load(open(filename, 'rb'))

app =  Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    #temp_array =list()
    if request.method == 'POST':

        Model1=float(request.form.get('Model1',False))
        Model2=float(request.form.get('Model2',False))
        Model3=float(request.form.get('Model3',False))
        Model4=float(request.form.get('Model4',False))
        Model5=float(request.form.get('Model5',False))
        
       
        '''sentinusdUSD = float(request.form['sentinusdUSD'])
        difficulty = float(request.form['difficulty'])
        difficulty90mom =float(request.form['difficulty90mom']) 
        price30momUSD = float(request.form['price30momUSD'])
        sentinusd90momUSD = float(request.form['sentinusd90momUSD'])
        transactions = float(request.form['transactions'])
        difficulty14std = float(request.form['difficulty14std'])
        transactionvalue3stdUSD=float(request.form['transactionvalue3stdUSD'])
        mining_profitability90trx = float(request.form['mining_profitability90trx'])
        activeaddresses7std	 = float(request.form['activeaddresses7std'])
        fee_to_reward3stdUSD = float(request.form['fee_to_reward3stdUSD'])
        activeaddresses3std = float(request.form['activeaddresses3std'])
        transactions3std = float(request.form['transactions3std'])
        difficulty7std = float(request.form['difficulty7std'])
        price14momUSD = float(request.form['price14momUSD'])
        mining_profitability30trx = float(request.form['mining_profitability30trx'])
        transactionvalue30momUSD =float(request.form['transactionvalue30momUSD']) 
        sentinusd30momUSD = float(request.form['sentinusd30momUSD'])
        difficulty30mom = float(request.form['difficulty30mom'])
        hashrate90mom = float(request.form['hashrate90mom'])'''

        #temp_data = [ [
        #                        sentinusdUSD, difficulty, difficulty90mom, 
        #                        price30momUSD, sentinusd90momUSD, transactions, 
        #                        difficulty14std, transactionvalue3stdUSD, mining_profitability90trx,
        #                         activeaddresses7std, hashrate90mom, fee_to_reward3stdUSD, activeaddresses3std, 
        #                         transactions3std, difficulty7std,price14momUSD,mining_profitability30trx, 
        #                         difficulty30mom,transactionvalue30momUSD,sentinusd30momUSD]]
                                 
        
        data= np.array([[Model1,Model2,Model3,Model4,Model5]])

    '''data = np.array([[sentinusd90momUSD,hashrate90mom,difficulty90mom,
       activeaddresses7std, difficulty7std,price14momUSD,
       sentinusdUSD, transactionvalue3stdUSD, activeaddresses3std,
       transactions3std, price30momUSD, fee_to_reward3stdUSD,
       mining_profitability90trx,sentinusd30momUSD,
       transactionvalue30momUSD, transactions, difficulty,
       difficulty14std, difficulty30mom, mining_profitability30trx,
       Change, expanding_mean, lag_1, lag_2, 'lag_3, lag_4, lag_5,
       lag_6, lag_7, Return, Mean, difference, 30day_WMA,
       30_day_EMA]])'''
    my_prediction = model.predict(data)
    return render_template('result.html',prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)