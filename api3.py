from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
 
 
app = Flask(__name__)
CORS(app)  
 
 
 
model = joblib.load('modelo_preco_pecas.pkl')
 
 
encoder = joblib.load('onehot_encoder.pkl')
 
 
pecas = ['Para-choque dianteiro (R$)', 'Farol dianteiro (R$)', 'Capô (R$)', 'Grade dianteira (R$)',
         'Para-lama dianteiro (R$)', 'Espelho retrovisor (R$)', 'Porta dianteira (R$)',
         'Vidro da porta (R$)', 'Painel frontal (R$)', 'Para-choque traseiro (R$)',
         'Lanterna traseira (R$)', 'Porta-malas (R$)', 'Spoiler traseiro (R$)',
         'Pára-brisa (R$)', 'Rodas (R$)', 'Assentos/Bancos (R$)', 'Motor (R$)', 'Bateria (R$)']
 
@app.route('/predict', methods=['GET'])
def predict():
   
    marca = request.args.get('marca').lower()
    peca = request.args.get('peca')
   
   
    if peca not in pecas:
        return jsonify({"error": f"A peça '{peca}' não é válida. Escolha uma das seguintes: {pecas}"}), 400
   
   
    marca_encoded = encoder.transform([[marca]])
    marca_df = pd.DataFrame(marca_encoded, columns=encoder.get_feature_names_out(['Marca/Modelo']))
   
   
    peca_valores = [0] * len(pecas)
    peca_index = pecas.index(peca)
    peca_valores[peca_index] = 1  
    peca_df = pd.DataFrame([peca_valores], columns=pecas)
 
   
    input_data = pd.concat([marca_df, peca_df], axis=1)
   
   
    predicted_price = model.predict(input_data)[0][peca_index]
 
   
    return jsonify({
        "marca": marca,
        "peca": peca,
        "preco_estimado": predicted_price
    })
 

 
 
if __name__ == '__main__':
    app.run(debug=True, port=5001)
