# Smishing Detection System: Feature-Engineered Ensemble Architecture

Brought to you by:
1. Chris Jewel Barcebal
2. Cris Javellana
3. Joshua Ticot
___
This project provides a robust, modular pipeline for detecting SMS phishing (smishing). It leverages an ensemble of machine learning models that combine textual analysis (via TF-IDF) and structural metadata features to provide a high-accuracy, low-latency detection service.

## **System Architecture**
The system is built on a modular pipeline approach:
1.  **Feature Engineering:** Extracts linguistic cues, structural markers, and sender metadata.
2.  **Vectorization:** Transforms raw text into high-dimensional space using optimized TF-IDF.
3.  **Base-Model Ensemble:** Trains specialized estimators for both text and metadata.
4.  **Stacked Meta-Classifier:** Learns the optimal combination of base predictions for final classification.

## **Key Components**
* **`src/preprocessing/text_cleaner.py`**: The "engine room" for feature generation. Handles the extraction of meta-features and ensures uniform data processing.
* **`src/models/ensemble.py`**: Orchestrates model training, stacking logic, and hyperparameter optimization.
* **`src/evaluation/metrics.py`**: Automated backend for plotting performance matrices and validation grids.
* **`config/threat_keywords.py`**: Configuration file for defining custom urgency triggers and smishing-related vocabulary.

## **Quick Start**
1.  **Environment Setup**: Install dependencies via `pip install -r requirements.txt`.
2.  **Data Processing**: Run the feature extraction pipeline to generate the structured dataset.
3.  **Model Training**: Execute the experimental pipeline to train base models and select the champion ensemble configuration.
4.  **Deployment**: Export model artifacts to the `models/` directory for use in the production environment.


## **Evaluation & Performance**
The system utilizes **5-fold Cross-Validation** and **GridSearch** to ensure stability. The "Champion" configuration is automatically identified by evaluating the F1-Score of various ensemble combinations, ensuring that the model maintains peak performance against both False Positives and False Negatives.

## **License & Documentation**
Developed as a specialized smishing detection research framework. For further implementation details, please refer to the technical modules within the `src/` directory.