from flask import Flask, request, jsonify
    import os
    import joblib
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    import requests
    from bs4 import BeautifulSoup
    import firebase_admin
    from firebase_admin import credentials, firestore
    import json

    # Initialize Firebase Admin SDK (replace with your credentials)
    firebase_credentials_json = os.environ.get('FIREBASE_CREDENTIALS')

    if firebase_credentials_json:
        try:
            cred = credentials.Certificate(json.loads(firebase_credentials_json))
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            WIKI_URL_COLLECTION = 'fish_wiki_urls'
        except Exception as e:
            print(f"Error initializing Firebase Admin SDK: {e}")
            db = None
            WIKI_URL_COLLECTION = None
    else:
        print("Firebase credentials not found in environment variables.")
        db = None
        WIKI_URL_COLLECTION = None

    # App definition
    app = Flask(__name__)

    # Helper function definitions
    def _fetch_and_parse_wiki_page(wiki_url: str) -> BeautifulSoup | None:
        """Fetches and parses a Wikipedia page."""
        try:
            response = requests.get(wiki_url)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {wiki_url}: {e}")
            return None
        except Exception as e:
            print(f"Error parsing URL {wiki_url}: {e}")
            return None

    def _get_wiki_url_from_firestore(fish_name: str) -> str | None:
        """Retrieves the Wikipedia URL for a fish from Firestore."""
        if db is None:
            print("Firestore not initialized.")
            return None
        try:
            doc_ref = db.collection(WIKI_URL_COLLECTION).document(fish_name)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict().get('url')
            return None
        except Exception as e:
            print(f"Error accessing Firestore for {fish_name}: {e}")
            return None

    def get_binomial_name(fish_name: str) -> str | None:
        """Retrieves the binomial name of a fish from its Wikipedia page (Firestore URL)."""
        wiki_url = _get_wiki_url_from_firestore(fish_name)
        if wiki_url:
            soup = _fetch_and_parse_wiki_page(wiki_url)
            if soup:
                binomial_name_span = soup.find('span', class_='binomial')
                if binomial_name_span:
                    binomial_name_i = binomial_name_span.find('i')
                    if binomial_name_i:
                        return binomial_name_i.text.strip()
        return None

    def get_scientific_classification(fish_name: str) -> dict[str, str] | None:
        """Retrieves the scientific classification of a fish from its Wikipedia page (Firestore URL)."""
        wiki_url = _get_wiki_url_from_firestore(fish_name)
        if wiki_url:
            soup = _fetch_and_parse_wiki_page(wiki_url)
            if soup:
                classification_table = soup.find('table', class_='infobox biota')
                if classification_table:
                    classification = {}
                    rows = classification_table.find_all('tr')
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) == 2:
                            rank = cols[0].text.strip(':').split(':')[0]
                            value_element = cols[1].find('a')
                            value = value_element.text.strip() if value_element else cols[1].text.strip()
                            classification[rank] = value
                        return classification
        return None

    def get_image_url(fish_name: str) -> str | None:
        """Retrieves the image URL of a fish from its Wikipedia page (Firestore URL)."""
        wiki_url = _get_wiki_url_from_firestore(fish_name)
        if wiki_url:
            soup = _fetch_and_parse_wiki_page(wiki_url)
            if soup:
                infobox = soup.find('table', class_='infobox biota')
                if infobox:
                    image_td = infobox.find('td')
                    if image_td:
                        image_element = image_td.find('img', class_='mw-file-element', width=lambda w: w and int(w) > 50)
                        if not image_element:
                            image_element = image_td.find('img', class_='mw-file-element') # Fallback
                        if image_element and 'src' in image_element.attrs and not 'icon' in image_element['src'].lower():
                            if image_element['src'].startswith('//'):
                                return 'https:' + image_element['src']
                            else:
                                return 'https://en.wikipedia.org' + image_element['src']
        return None

    def predict_specific_model(model_name: str, fish_data) -> jsonify:
        """Loads a specific trained model and predicts recommendations based on Firestore data."""
        try:
            model = joblib.load(f'{model_name}.pkl')
            # Create a list to store predictions
            predictions = []
            for fish in fish_data:
                # Extract relevant features for prediction
                features = [fish['Temperature'], fish['pH'], fish['GH'], fish['KH']]
                prediction = model.predict([features])[0]  # Predict for each fish
                predictions.append({'Name': fish['Name'], 'prediction': prediction})

            # Sort fish based on prediction
            sorted_predictions = sorted(predictions, key=lambda x: x['prediction'])
            recommendations = [item['Name'] for item in sorted_predictions]

            return jsonify({
                "model": model_name,
                "recommendations": recommendations
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    # Endpoints
    @app.route('/upload', methods=['GET'])  # Keep GET method
    def upload_csv() -> jsonify:
        """Endpoint to acknowledge the removal of CSV upload functionality."""
        return jsonify({"message": "CSV upload functionality has been removed. Data is now fetched from Firestore."}), 200


    @app.route('/train', methods=['GET'])
    def train_models() -> jsonify:
        """Endpoint to train machine learning models using data from Firestore."""
        if db is None:
            return jsonify({"error": "Firestore not initialized."}), 500

        try:
            fish_data_ref = db.collection('fish_data')
            fish_docs = fish_data_ref.get()
            fish_data = [doc.to_dict() for doc in fish_docs]

            # Prepare data for training
            X = [[fish['Temperature'], fish['pH'], fish['GH'], fish['KH']] for fish in fish_data]
            y = [fish['Nitrate'] for fish in fish_data]

            models = {
                "Linear": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "RandomForest": RandomForestRegressor()
            }
            results = []
            for name, model in models.items():
                model.fit(X, y)
                y_pred = model.predict(X)
                acc = r2_score(y, y_pred)
                joblib.dump(model, f'{name}.pkl')
                results.append({"model": name, "accuracy": acc})
            sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
            return jsonify({"results": sorted_results})
        except Exception as e:
             return jsonify({"error": f"Error fetching or processing data from Firestore: {str(e)}"}), 500


    @app.route('/predict', methods=['GET'])
    def predict() -> jsonify:
        """Endpoint to predict top 10 most suitable fish based on Firestore data."""
        if db is None:
            return jsonify({"error": "Firestore not initialized."}), 500

        try:
            model_name = request.args.get('model')
            model = joblib.load(f'{model_name}.pkl')
            temp = float(request.args.get('temp'))
            ph = float(request.args.get('ph'))
            gh = float(request.args.get('gh'))
            kh = float(request.args.get('kh'))

            fish_data_ref = db.collection('fish_data')
            fish_docs = fish_data_ref.get()
            fish_data = [doc.to_dict() for doc in fish_docs]

            # Calculate scores based on Firestore data
            scored_fish = []
            for fish in fish_data:
                score = ((fish['Temperature'] - temp)**2 +
                         (fish['pH'] - ph)**2 +
                         (fish['GH'] - gh)**2 +
                         (fish['KH'] - kh)**2)**0.5
                scored_fish.append({'fish': fish, 'score': score})

            # Calculate percentages
            max_score = max(f['score'] for f in scored_fish) if scored_fish else 0
            min_score = min(f['score'] for f in scored_fish) if scored_fish else 0

            for fish_data in scored_fish:
                percentage = 100 * (max_score - fish_data['score']) / (max_score - min_score) if max_score > min_score else 100
                fish_data['percentage'] = round(percentage, 2)

            # Sort and get top 10
            top10_scored = sorted(scored_fish, key=lambda x: x['percentage'], reverse=True)[:10]

            results = []
            for item in top10_scored:
                fish = item['fish']
                name = fish['Name'].split(',')[0].strip()
                binomial_name = get_binomial_name(name)
                scientific_classification = get_scientific_classification(name) or {}
                image_url = get_image_url(name)

                fish_data = {
                    "name": name,
                    "percentage": item['percentage'],
                    "image_url": image_url,
                    "binomial_name": binomial_name,
                    "scientific_classification": scientific_classification
                }
                results.append(fish_data)

            return jsonify({"model": model_name, "top_10": results})
        except Exception as e:
            return jsonify({"error": f"Error fetching or processing data from Firestore: {str(e)}"}), 500

    # Individual Model Endpoints
    @app.route('/predict/linear', methods=['GET'])
    def predict_linear() -> jsonify:
        """Endpoint to specifically use the 'Linear' model for prediction."""
        if db is None:
            return jsonify({"error": "Firestore not initialized."}), 500

        try:
            fish_data_ref = db.collection('fish_data')
            fish_docs = fish_data_ref.get()
            fish_data = [doc.to_dict() for doc in fish_docs]
            return predict_specific_model('Linear', fish_data)
        except Exception as e:
            return jsonify({"error": f"Error fetching data from Firestore: {str(e)}"}), 400


    @app.route('/predict/knn', methods=['GET'])
    def predict_knn() -> jsonify:
        """Endpoint to specifically use the 'KNN' model for prediction."""
        if db is None:
            return jsonify({"error": "Firestore not initialized."}), 500

        try:
            fish_data_ref = db.collection('fish_data')
            fish_docs = fish_data_ref.get()
            fish_data = [doc.to_dict() for doc in fish_docs]
            return predict_specific_model('KNN', fish_data)
        except Exception as e:
             return jsonify({"error": f"Error fetching data from Firestore: {str(e)}"}), 400

    @app.route('/predict/randomforest', methods=['GET'])
    def predict_rf() -> jsonify:
        """Endpoint to specifically use the 'RandomForest' model for prediction."""
        if db is None:
            return jsonify({"error": "Firestore not initialized."}), 500
        try:
            fish_data_ref = db.collection('fish_data')
            fish_docs = fish_data_ref.get()
            fish_data = [doc.to_dict() for doc in fish_docs]
            return predict_specific_model('RandomForest', fish_data)
        except Exception as e:
            return jsonify({"error": f"Error fetching data from Firestore: {str(e)}"}), 400

    # if __name__ code
    if __name__ == '__main__':
        app.run(debug=True)
