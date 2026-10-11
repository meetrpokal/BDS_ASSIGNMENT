import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys
from datetime import datetime

import config

# Set environment for PySpark
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

st.set_page_config(
    page_title="ML Models Dashboard - Amazon Reviews",
    page_icon="🤖",
    layout="wide"
)

# Light theme CSS
st.markdown("""
<style>
    .main {background-color: #FFFFFF;}
    h1 {color: #FF9900;}
    h2 {color: #232F3E;}
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E0E0E0;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border: 2px solid;
    }
    .positive-box {
        background-color: #E8F5E9;
        border-color: #00cc96;
    }
    .negative-box {
        background-color: #FFEBEE;
        border-color: #ef553b;
    }
    .neutral-box {
        background-color: #FFF3E0;
        border-color: #ffa15a;
    }
</style>
""", unsafe_allow_html=True)

# Get current date/time in exact format
current_datetime = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

st.title("🤖 Machine Learning Models Dashboard")
st.markdown("**Amazon Reviews Sentiment Analysis** ")

# Locate training artifacts (cluster shared storage first, then local folders)
artifacts = config.locate_artifacts()
if artifacts is None:
    st.error("❌ No ML models found!")
    st.info("Please run: `python ml_sentiment_model.py` to train models first")
    st.stop()

MODELS_DIR, OUTPUT_DIR = artifacts

# Load metadata
with open(os.path.join(MODELS_DIR, "model_metadata.json"), "r") as f:
    metadata = json.load(f)

# Initialize Spark session for model loading (cached)
@st.cache_resource
def init_spark():
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder \
        .appName("MLDashboard") \
        .master("local[2]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    return spark

# Load ML model (cached)
@st.cache_resource
def load_ml_model():
    from pyspark.ml import PipelineModel
    
    try:
        spark = init_spark()
        model = PipelineModel.load(os.path.join(MODELS_DIR, "sentiment_lr_model"))
        return model, spark
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Prediction function using actual model
def predict_sentiment_real(review_text, model, spark):
    """Predict sentiment using the actual trained model"""
    try:
        review_length = len(review_text)
        word_count = len(review_text.split())
        
        data = [(
            review_text,
            review_length,
            word_count,
            0.5,
            1.0
        )]
        
        df = spark.createDataFrame(
            data, 
            ["review_body", "review_length", "word_count", "helpfulness_ratio", "is_verified"]
        )
        
        prediction = model.transform(df)
        result = prediction.select("prediction", "probability").first()
        
        sentiment = "Positive" if result.prediction == 1.0 else "Negative"
        probability_array = result.probability.toArray()
        confidence = float(probability_array[int(result.prediction)])
        
        prob_negative = float(probability_array[0])
        prob_positive = float(probability_array[1])
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'prob_negative': prob_negative,
            'prob_positive': prob_positive
        }
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Display model info
st.header("📊 Model Training Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 40px; margin-bottom: 10px;'>🎯</div>
        <div style='font-size: 28px; font-weight: 700; color: #232F3E;'>{len(metadata['models'])}</div>
        <div style='font-size: 14px; color: #666; margin-top: 5px;'>ML Models Trained</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    training_date = metadata['training_date'].split()[0]
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 40px; margin-bottom: 10px;'>📅</div>
        <div style='font-size: 18px; font-weight: 700; color: #232F3E;'>{training_date}</div>
        <div style='font-size: 14px; color: #666; margin-top: 5px;'>Training Date</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 40px; margin-bottom: 10px;'>📝</div>
        <div style='font-size: 28px; font-weight: 700; color: #232F3E;'>{metadata['total_samples']:,}</div>
        <div style='font-size: 14px; color: #666; margin-top: 5px;'>Training Samples</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    best_auc = max(m['auc'] for m in metadata['models'].values() if 'auc' in m)
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size: 40px; margin-bottom: 10px;'>🏆</div>
        <div style='font-size: 28px; font-weight: 700; color: #00cc96;'>{best_auc:.4f}</div>
        <div style='font-size: 14px; color: #666; margin-top: 5px;'>Best AUC Score</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Model comparison
st.header("📈 Model Performance Comparison")

if os.path.exists(os.path.join(OUTPUT_DIR, "model_comparison.csv")):
    comparison_df = pd.read_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))
    
    models_data = []
    for model_key, model_data in metadata['models'].items():
        if 'auc' in model_data:
            models_data.append({
                'Model': model_key.replace('_', ' ').title(),
                'AUC': model_data['auc'],
                'Accuracy': model_data['accuracy'],
                'F1': model_data['f1']
            })
    
    col1, col2 = st.columns(2)
    
    with col1:
        if models_data:
            class_df = pd.DataFrame(models_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=class_df['Model'],
                y=class_df['AUC'],
                name='AUC',
                marker_color='#FF9900',
                text=class_df['AUC'].round(4),
                textposition='outside'
            ))
            fig.add_trace(go.Bar(
                x=class_df['Model'],
                y=class_df['Accuracy'],
                name='Accuracy',
                marker_color='#232F3E',
                text=class_df['Accuracy'].round(4),
                textposition='outside'
            ))
            
            fig.update_layout(
                title='<b>Classification Models Performance</b>',
                barmode='group',
                template='plotly_white',
                height=400,
                yaxis_range=[0, 1],
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📋 Model Metrics")
        display_data = []
        for model_key, model_data in metadata['models'].items():
            if 'auc' in model_data:
                display_data.append({
                    'Model': model_key.replace('_', ' ').title(),
                    'AUC': f"{model_data['auc']:.4f}",
                    'Accuracy': f"{model_data['accuracy']:.4f}",
                    'F1 Score': f"{model_data['f1']:.4f}"
                })
        
        if display_data:
            st.dataframe(
                pd.DataFrame(display_data),
                use_container_width=True,
                hide_index=True
            )

# Feature importance

# Live Prediction
st.header("🎯 Live Sentiment Prediction")

st.markdown("""
<div style='background-color: #E3F2FD; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3;'>
    <p style='color: #1565C0; margin: 0;'>
        <strong>🚀 Real-time AI Prediction!</strong> This uses your actual trained Logistic Regression model 
        (AUC: {:.2%}) to predict sentiment. The model analyzes the review text and provides confidence scores.
    </p>
</div>
""".format(metadata['models']['logistic_regression']['auc']), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Example reviews
example_reviews = {
    "Positive Example": "This product is absolutely amazing! The quality exceeded my expectations and it works perfectly. Highly recommended!",
    "Negative Example": "Terrible product. Broke after one day of use. Complete waste of money. Very disappointed.",
    "Neutral Example": "The product is okay. It does what it's supposed to do but nothing special.",
}

col1, col2, col3 = st.columns(3)
with col1:
    if st.button(" Load Positive Example"):
        st.session_state.review_input = example_reviews["Positive Example"]

with col2:
    if st.button("Load Negative Example"):
        st.session_state.review_input = example_reviews["Negative Example"]

with col3:
    if st.button("Load Neutral Example"):
        st.session_state.review_input = example_reviews["Neutral Example"]

review_text = st.text_area(
    "Enter product review:",
    value=st.session_state.get('review_input', ''),
    placeholder="Example: This product is amazing! Works perfectly and exceeded my expectations.",
    height=150,
    key='review_area'
)

col1, col2 = st.columns([1, 3])

with col1:
    predict_button = st.button("🔮 Predict Sentiment", type="primary", use_container_width=True)

with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.review_input = ""
        st.rerun()

if predict_button and review_text:
    with st.spinner("🤖 Loading AI model..."):
        model, spark = load_ml_model()
    
    if model and spark:
        with st.spinner("🔍 Analyzing sentiment..."):
            result = predict_sentiment_real(review_text, model, spark)
        
        if result:
            sentiment = result['sentiment']
            confidence = result['confidence']
            prob_positive = result['prob_positive']
            prob_negative = result['prob_negative']
            
            if sentiment == "Positive":
                color = "#00cc96"
                emoji = "😊"
                box_class = "positive-box"
            else:
                color = "#ef553b"
                emoji = "😞"
                box_class = "negative-box"
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"""
                <div class='prediction-box {box_class}'>
                    <h2 style='color: {color}; margin: 0; display: flex; align-items: center;'>
                        <span style='font-size: 48px; margin-right: 15px;'>{emoji}</span>
                        Prediction: {sentiment}
                    </h2>
                    <p style='color: #666; margin: 15px 0 0 0; font-size: 18px;'>
                        <strong>Confidence:</strong> {confidence*100:.2f}%
                    </p>
                    <div style='margin-top: 20px;'>
                        <div style='background-color: rgba(0,0,0,0.1); border-radius: 10px; padding: 10px;'>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                                <span>Positive:</span>
                                <strong>{prob_positive*100:.1f}%</strong>
                            </div>
                            <div style='background-color: white; border-radius: 5px; height: 25px; overflow: hidden;'>
                                <div style='background-color: #00cc96; height: 100%; width: {prob_positive*100}%;'></div>
                            </div>
                        </div>
                        <div style='background-color: rgba(0,0,0,0.1); border-radius: 10px; padding: 10px; margin-top: 10px;'>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                                <span>Negative:</span>
                                <strong>{prob_negative*100:.1f}%</strong>
                            </div>
                            <div style='background-color: white; border-radius: 5px; height: 25px; overflow: hidden;'>
                                <div style='background-color: #ef553b; height: 100%; width: {prob_negative*100}%;'></div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = confidence * 100,
                    title = {'text': "Confidence Score", 'font': {'size': 24}},
                    delta = {'reference': 50, 'increasing': {'color': color}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1},
                        'bar': {'color': color, 'thickness': 0.75},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': '#FFE0E0'},
                            {'range': [50, 75], 'color': '#FFF4E0'},
                            {'range': [75, 100], 'color': '#E0FFE0'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(
                    height=300,
                    margin={'t': 50, 'b': 0, 'l': 0, 'r': 0}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 🔬 Analysis Details")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Review Length", f"{len(review_text)} chars")
            
            with col2:
                st.metric("Word Count", len(review_text.split()))
            
            with col3:
                st.metric("Model Used", "Logistic Regression")
            
            with col4:
                st.metric("Model AUC", f"{metadata['models']['logistic_regression']['auc']:.4f}")

elif predict_button and not review_text:
    st.warning("⚠️ Please enter a review to analyze")

# Sample predictions
if os.path.exists(os.path.join(OUTPUT_DIR, "lr_predictions_sample.csv")):
    st.header("📊 Sample Predictions from Test Set")
    
    with st.expander("View Sample Predictions", expanded=False):
        sample_df = pd.read_csv(os.path.join(OUTPUT_DIR, "lr_predictions_sample.csv"))
        sample_df = sample_df.head(20)
        
        sample_df['Predicted Sentiment'] = sample_df['prediction'].map({1.0: 'Positive ✅', 0.0: 'Negative ❌'})
        sample_df['Actual Sentiment'] = sample_df['sentiment'].map({'positive': 'Positive ✅', 'negative': 'Negative ❌'})
        
        sample_df['Match'] = sample_df.apply(
            lambda row: '✅ Correct' if (row['prediction'] == 1.0 and row['sentiment'] == 'positive') or 
                                        (row['prediction'] == 0.0 and row['sentiment'] == 'negative') 
            else '❌ Wrong', 
            axis=1
        )
        
        st.dataframe(
            sample_df[['review_body', 'Actual Sentiment', 'Predicted Sentiment', 'Match']].head(10),
            use_container_width=True,
            hide_index=True
        )
        
        correct = len(sample_df[sample_df['Match'] == '✅ Correct'])
        total = len(sample_df)
        st.success(f"Sample Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

# Confusion Matrix
if os.path.exists(os.path.join(OUTPUT_DIR, "confusion_matrix.csv")):
    st.header("📊 Confusion Matrix")
    
    confusion_df = pd.read_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"))
    confusion_pivot = confusion_df.pivot(index='label', columns='prediction', values='count').fillna(0)
    
    fig = px.imshow(
        confusion_pivot,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Negative (0)', 'Positive (1)'],
        y=['Negative (0)', 'Positive (1)'],
        title='<b>Confusion Matrix - Logistic Regression Model</b>',
        color_continuous_scale='Blues',
        text_auto=True,
        template='plotly_white'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 20px; background-color: #F8F9FA; border-radius: 10px;'>
    <h3 style='color: #232F3E; margin-bottom: 10px;'>🤖 ML Models Dashboard</h3>
    <p style='color: #666;'>
        <strong>Model:</strong> Logistic Regression | 
        <strong>AUC:</strong> {metadata['models']['logistic_regression']['auc']:.4f} | 
        <strong>Accuracy:</strong> {metadata['models']['logistic_regression']['accuracy']:.4f}
    </p>
    <p style='color: #666;'>
         {current_datetime} UTC
    </p>
</div>
""", unsafe_allow_html=True)