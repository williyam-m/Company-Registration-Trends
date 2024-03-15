import pickle
import joblib
#import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
from flask import Flask, render_template, request, Response, send_file
from sklearn.linear_model import LinearRegression


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html', val = '')

@app.route('/company-registration-trends')
def company():
    return render_template('company.html', val = '')

@app.route('/company-registration-trends/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        inputyear = request.form['year']
        if inputyear == '':
            return render_template('company.html', val = 'Please enter year')
        if int(inputyear) < 2025:
            return render_template('company.html', val = 'Please enter future year')

        data = pd.read_csv("registrations.csv", sep = ',', on_bad_lines = 'skip', index_col = False, dtype = 'unicode')
        data['DATE_OF_REGISTRATION'] = pd.to_datetime(data['DATE_OF_REGISTRATION'], errors = 'coerce')

        data = data.dropna(subset=['DATE_OF_REGISTRATION'])
        years = data['DATE_OF_REGISTRATION'].dt.year.astype(int)

        """year_counts = {}
        for year in np.unique(years):
            count = np.sum(years == year)
            year_counts[year] = count"""

        year_counts = years.value_counts().to_dict()

        # Prepare data for training
        year_values = list(year_counts.keys())
        registration_counts = list(year_counts.values())

        """year_values = np.array(year_values).reshape(-1, 1)
        registration_counts = np.array(registration_counts)
        """

        year_values = pd.DataFrame(year_values, columns = ['year'])  # Convert to DataFrame
        registration_counts = pd.DataFrame(registration_counts, columns = ['count'])  # Convert to DataFrame

        # Initialize and train the model
        model = LinearRegression()
        model.fit(year_values, registration_counts)

        # Predict registrations for 2025
        predicted_count = model.predict([[int(inputyear)]])

        return render_template('company.html', val = f' In {inputyear} Company registration count will be around \n {int(predicted_count)-5} to {int(predicted_count)+10}.')



@app.route('/plot')
def plot():
    """data = pd.read_csv("registrations.csv" , sep=',',on_bad_lines='skip', index_col=False, dtype='unicode')

    # Convert 'year' column to datetime
    data['DATE_OF_REGISTRATION'] = pd.to_datetime(data['DATE_OF_REGISTRATION'], errors='coerce')

    # Remove rows with missing 'year' values
    data = data.dropna(subset=['DATE_OF_REGISTRATION'])

    # Extract year from datetime and convert to integers
    years = data['DATE_OF_REGISTRATION'].dt.year.astype(int)

    year_counts = {}
    for year in np.unique(years):
        count = np.sum(years == year)
        year_counts[year] = count

    # Prepare data for training
    year_values = list(year_counts.keys())
    registration_counts = list(year_counts.values())

    year_values = np.array(year_values).reshape(-1, 1)
    registration_counts = np.array(registration_counts)

    model = LinearRegression()
    model.fit(year_values, registration_counts)

    # Generate a plot
    plt.figure(figsize=(10, 7))
    plt.plot(year_values, registration_counts, label='Actual Data')
    plt.plot(year_values, model.predict(year_values), label='Predicted Data', linestyle='dashed')
    plt.xlabel('Year')
    plt.ylabel('Registration Counts')
    plt.title('Company Registration Counts Over the Years')
    plt.legend()

    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return Response(img, mimetype='image/png')"""
    return render_template('plot.html')

@app.route('/download')
def download():
    data = pd.read_csv("registrations.csv", sep = ',', on_bad_lines = 'skip', index_col = False, dtype = 'unicode')
    data.to_csv("company_registration_dataset.csv", index = False)

    return send_file("company_registration_dataset.csv",as_attachment = True)

@app.route('/dataset')
def dataset():
    # Read the CSV file
    data = pd.read_csv("registrations.csv" , nrows=150)  # Replace with your CSV file path
    data = data.dropna(subset=['DATE_OF_REGISTRATION'])
    return render_template('dataset.html', data = data)

if __name__ == '__main__':
    app.run(debug=True)