from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import os
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:newpassword123@127.0.0.1/retail'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.String(50))
    cluster = db.Column(db.Integer)

with app.app_context():
    db.create_all()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'kmeans_model.pkl')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
ONLINE_RETAIL_PATH = os.path.join(BASE_DIR, 'OnlineRetail.csv')
os.makedirs(STATIC_DIR, exist_ok=True)
model = None

def load_and_clean_data(file_path):
    # Load data
    retail = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)

    # Convert CustomerID to string and create Amount column
    retail['CustomerID'] = retail['CustomerID'].astype(str)
    retail['Amount'] = retail['Quantity']*retail['UnitPrice']

    # Compute RFM metrics
    rfm_m = retail.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']
    
    retail['InvoiceDate'] = pd.to_datetime(
        retail['InvoiceDate'],
        dayfirst=True,
        errors='coerce'
    )
    retail = retail.dropna(subset=['InvoiceDate'])
    max_date = max(retail['InvoiceDate'])
    retail['Diff'] = max_date - retail['InvoiceDate']
    
    rfm_p = retail.groupby('CustomerID')['Diff'].min().reset_index()
    rfm_p['Diff'] = rfm_p['Diff'].dt.days
    
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

    # Remove outliers
    numeric_rfm = rfm[['Amount', 'Frequency', 'Recency']]
    Q1 = numeric_rfm.quantile(0.05)
    Q3 = numeric_rfm.quantile(0.95)
    IQR = Q3 - Q1
    
    rfm = rfm[(rfm.Amount >= Q1['Amount'] - 1.5*IQR['Amount']) & (rfm.Amount <= Q3['Amount'] + 1.5*IQR['Amount'])]
    rfm = rfm[(rfm.Recency >= Q1['Recency'] - 1.5*IQR['Recency']) & (rfm.Recency <= Q3['Recency'] + 1.5*IQR['Recency'])]
    rfm = rfm[(rfm.Frequency >= Q1['Frequency'] - 1.5*IQR['Frequency']) & (rfm.Frequency <= Q3['Frequency'] + 1.5*IQR['Frequency'])]

    return rfm

def preprocess_data(file_path):
    rfm = load_and_clean_data(file_path)
    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]
    
    # Instantiate
    scaler = StandardScaler()
    
    # fit_transform
    rfm_df_scaled = scaler.fit_transform(rfm_df)
    rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
    
    # rfm_df_scaled
    rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']
    
    return rfm, rfm_df_scaled


def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass

    if not os.path.exists(ONLINE_RETAIL_PATH):
        raise FileNotFoundError(
            f'No model found and sample data missing at {ONLINE_RETAIL_PATH}'
        )

    _, df_scaled = preprocess_data(ONLINE_RETAIL_PATH)
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(df_scaled)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    return model


model = load_model()

@app.route('/')
def home():
    predictions = Prediction.query.all()
    return render_template('index.html', predictions=predictions)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(BASE_DIR, filename)
    file.save(file_path)

    rfm, df_scaled = preprocess_data(file_path)
    results_df = model.predict(df_scaled)
    rfm['Cluster_Id'] = results_df

    for index, row in rfm.head(10).iterrows():
        entry = Prediction(
            customer_id=str(row['CustomerID']), 
            cluster=int(row['Cluster_Id'])
        )
        db.session.add(entry)
    db.session.commit()

    amount_img_path = os.path.join(STATIC_DIR, 'ClusterId_Amount.png')
    freq_img_path = os.path.join(STATIC_DIR, 'ClusterId_Frequency.png')
    recency_img_path = os.path.join(STATIC_DIR, 'ClusterId_Recency.png')

    plt.figure()
    sns.boxplot(x='Cluster_Id', y='Amount', data=rfm, hue='Cluster_Id')
    plt.savefig(amount_img_path)
    plt.close()

    plt.figure()
    sns.boxplot(x='Cluster_Id', y='Frequency', data=rfm, hue='Cluster_Id')
    plt.savefig(freq_img_path)
    plt.close()

    plt.figure()
    sns.boxplot(x='Cluster_Id', y='Recency', data=rfm, hue='Cluster_Id')
    plt.savefig(recency_img_path)
    plt.close()

    response = {
            'amount_img': '/static/ClusterId_Amount.png',
            'freq_img': '/static/ClusterId_Frequency.png',
            'recency_img': '/static/ClusterId_Recency.png'
    }

    return jsonify(response)

# for local
if __name__ == "__main__":
    app.run(debug=True)

# for cloud
# if __name__ == "__main__":
#     app.run(host = '0.0.0.0', port=8080)