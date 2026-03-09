# Customer Support Ticket Classification and Priority Prediction System

## Project Overview
This is an end-to-end Machine Learning project that automates the handling of customer support tickets. The system reads incoming text tickets, identifies their intent by classifying them into specific categories, and predicts their severity/priority levels. 

By utilizing Natural Language Processing (NLP) techniques and Machine Learning algorithms, this tool reduces manual triage efforts and ensures urgent issues receive immediate attention.

## Business Impact Explanation
Implementing this automated ticket classification system provides multiple advantages to a company:
* **Automated Routing:** Tickets are instantly sent to the correct department (e.g., Billing issues to Finance, Technical Problems to IT), reducing transfer bounce rates.
* **Prioritize Urgent Problems:** Identifying high-priority tickets early means crucial issues (like payment failures or server crashes) are fast-tracked for resolution.
* **Reduce Response Time:** Removing the manual review stage for ticket triage saves valuable hours, drastically cutting down the First Response Time (FRT).
* **Increase Support Team Efficiency:** Agents can focus on resolving issues rather than categorizing them, boosting their output and improving overall customer satisfaction.

## Dataset Description
The model is trained on a simulated customer support ticket dataset (located in `data/raw_tickets.csv`). The data contains the following columns:
* `ticket_id`: Unique identifier for the ticket.
* `ticket_text`: The raw text of the customer's query or problem.
* `category`: The categorized intent of the ticket. Classes include: Billing Issue, Technical Problem, Account Access, Refund Request, and General Inquiry.
* `priority`: The urgency of the ticket. Classes include: High, Medium, and Low.

## NLP Pipeline Explanation
We utilized standard Natural Language Processing (NLP) routines to prepare unstructured text data for machine learning:
1. **Lowercasing:** Enhances word matching consistency.
2. **Punctuation & Number Removal:** Removes irrelevant characters that don't contribute to sentiment or category identity.
3. **Stop Words Removal:** Discards common contextual words (e.g., "the", "and", "is") utilizing the NLTK library.
4. **Lemmatization:** Reduces words to their absolute root dictionary format (e.g., 'running' becomes 'run'), reducing the overall vocabulary size while maintaining semantics.
5. **TF-IDF Vectorization:** Converts our clean tokens into a Term Frequency-Inverse Document Frequency matrix mapping text significance to numerical values. 

## Model Performance
The system evaluates multiple classical machine learning models for predictions. 

**Category Prediction:**
* **Models Evaluated:** Logistic Regression, Random Forest Classifier, Multinomial Naive Bayes.
* **Selection:** Based on accuracy metrics, the strongest generalizing model is retained.

**Priority Prediction:**
* **Models Evaluated:** Logistic Regression, Random Forest Classifier.
* **Selection:** Captures the complex relationship between urgency indicators in the text to correctly predict output priority.

## Instructions to Run the System

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the necessary requirements:
```bash
pip install -r requirements.txt
```

### 2. Generate Data
Create the simulated dataset by running:
```bash
cd src
python data_generation.py
```

### 3. Run Pipeline 
Execute the training script. This script automatically performs Data Cleaning, Text Preprocessing, Model Training, and saves the final `.pkl` models to the `models/` directory:
```bash
python train_models.py
```

### 4. Evaluate Models & Generate Visualizations
Evaluate the performance on your test split and generate confusion matrices:
```bash
python evaluate_models.py
```
*Screenshots of the CMs and distributions will be saved in the `visualizations/` folder.*

### 5. Prediction System (Inference)
Use the interactive predicting script to evaluate live customer queries:
```bash
python predict_ticket.py
```
*Example usage:* Enter "My payment failed but money was deducted" -> It will output "**Category:** Billing Issue | **Priority:** High".

### 6. Exploratory Data Analysis
You can open the Jupyter Notebook for further visual insights:
```bash
jupyter notebook ../notebooks/ticket_classification_analysis.ipynb
```
