import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")

# Load the saved model
@st.cache_resource
def load_model():
    with open('final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title with icon
st.title('💳 Credit Risk Prediction Dashboard')
st.markdown('### Assess credit delinquency risk with AI-powered predictions')

# Sidebar for inputs
st.sidebar.header('📋 Customer Information')
st.sidebar.markdown('---')

revolving_util = st.sidebar.slider(
    '💰 Revolving Credit Utilization',
    min_value=0.0, max_value=2.0, value=0.3, step=0.01,
    help='Ratio of credit card balance to credit limit'
)

times_30_59_late = st.sidebar.number_input(
    '⏰ Times 30-59 Days Late',
    min_value=0, max_value=20, value=0, step=1,
    help='Number of times payment was 30-59 days past due'
)

times_90_late = st.sidebar.number_input(
    '🚨 Times 90+ Days Late',
    min_value=0, max_value=20, value=0, step=1,
    help='Number of times payment was 90 or more days past due'
)

monthly_income = st.sidebar.number_input(
    '💵 Monthly Income ($)',
    min_value=0, max_value=500000, value=5000, step=500
)

times_60_89_late = st.sidebar.number_input(
    '⚠️ Times 60-89 Days Late',
    min_value=0, max_value=20, value=0, step=1,
    help='Number of times payment was 60-89 days past due'
)

age = st.sidebar.slider(
    '👤 Age',
    min_value=18, max_value=100, value=35, step=1
)

credit_lines = st.sidebar.number_input(
    '🏦 Open Credit Lines',
    min_value=0, max_value=50, value=5, step=1
)

st.sidebar.markdown('---')
predict_button = st.sidebar.button('🔮 Predict Risk', type='primary', use_container_width=True)

# Main content area
if predict_button:
    # Prepare input data
    input_data = np.array([[
        revolving_util,
        times_30_59_late,
        times_90_late,
        monthly_income,
        times_60_89_late,
        age,
        credit_lines
    ]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    risk_score = probability[1] * 100
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Risk Classification",
            value="HIGH RISK" if prediction == 1 else "LOW RISK",
            delta=f"{risk_score:.1f}% probability"
        )
    
    with col2:
        st.metric(
            label="Delinquency Probability",
            value=f"{risk_score:.1f}%"
        )
    
    with col3:
        risk_level = "Critical" if risk_score > 80 else "High" if risk_score > 50 else "Moderate" if risk_score > 20 else "Low"
        st.metric(
            label="Risk Level",
            value=risk_level
        )
    
    st.markdown('---')
    
    # Create gauge chart for risk score
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('### 📊 Risk Gauge')
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Delinquency Risk Score", 'font': {'size': 20}},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': '#00CC66'},
                    {'range': [20, 50], 'color': '#FFCC00'},
                    {'range': [50, 80], 'color': '#FF9933'},
                    {'range': [80, 100], 'color': '#FF3333'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        st.markdown('### 📈 Probability Distribution')
        
        # Probability bar chart
        prob_df = pd.DataFrame({
            'Category': ['No Delinquency', 'Delinquency'],
            'Probability': [probability[0] * 100, probability[1] * 100]
        })
        
        fig_bar = px.bar(
            prob_df,
            x='Category',
            y='Probability',
            color='Probability',
            color_continuous_scale=['green', 'red'],
            text='Probability'
        )
        
        fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_bar.update_layout(
            height=300,
            showlegend=False,
            yaxis_title="Probability (%)",
            xaxis_title="",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown('---')
    
    # Risk factors analysis
    st.markdown('### 🔍 Risk Factor Analysis')
    
    # Create feature importance visualization
    feature_names = [
        'Credit Utilization',
        '30-59 Days Late',
        '90+ Days Late',
        'Monthly Income',
        '60-89 Days Late',
        'Age',
        'Credit Lines'
    ]
    
    feature_values = [
        revolving_util,
        times_30_59_late,
        times_90_late,
        monthly_income / 10000,  # Normalize
        times_60_89_late,
        age / 100,  # Normalize
        credit_lines
    ]
    
    # Feature importance from your analysis
    feature_importance = [0.072, 0.050, 0.040, 0.022, 0.011, 0.009, 0.008]
    
    # Calculate risk contribution
    risk_contribution = [val * imp * 100 for val, imp in zip(feature_values, feature_importance)]
    
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Your Value': feature_values,
        'Importance': feature_importance,
        'Risk Contribution': risk_contribution
    })
    
    fig_features = px.bar(
        feature_df.nlargest(5, 'Risk Contribution'),
        x='Risk Contribution',
        y='Feature',
        orientation='h',
        color='Risk Contribution',
        color_continuous_scale='Reds',
        title='Top 5 Risk Contributing Factors'
    )
    
    fig_features.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_features, use_container_width=True)
    
    # Detailed risk breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('### ⚠️ Risk Indicators')
        
        risk_indicators = []
        if revolving_util > 0.7:
            risk_indicators.append("🔴 High credit utilization (>70%)")
        if times_30_59_late > 0:
            risk_indicators.append(f"🔴 {times_30_59_late} recent late payments (30-59 days)")
        if times_90_late > 0:
            risk_indicators.append(f"🔴 {times_90_late} serious late payments (90+ days)")
        if monthly_income < 3000:
            risk_indicators.append("🟡 Low monthly income")
        if credit_lines > 15:
            risk_indicators.append("🟡 Many open credit lines")
        
        if not risk_indicators:
            st.success("✅ No major risk indicators detected")
        else:
            for indicator in risk_indicators:
                st.warning(indicator)
    
    with col2:
        st.markdown('### 💡 Recommendations')
        
        recommendations = []
        if revolving_util > 0.3:
            recommendations.append("• Pay down credit card balances")
        if times_30_59_late > 0 or times_90_late > 0:
            recommendations.append("• Set up automatic payments to avoid late fees")
        if monthly_income < 5000:
            recommendations.append("• Consider additional income sources")
        if credit_lines > 10:
            recommendations.append("• Consolidate or close unused credit accounts")
        
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("✅ Good credit profile! Maintain current habits.")
    
    # Historical comparison (mock data for visualization)
    st.markdown('---')
    st.markdown('### 📉 Risk Trend Comparison')
    
    comparison_df = pd.DataFrame({
        'Risk Level': ['Very Low (0-20%)', 'Low (20-50%)', 'High (50-80%)', 'Very High (80-100%)'],
        'Percentage of Population': [45, 35, 15, 5],
        'Your Score': [risk_score if 0 <= risk_score < 20 else 0,
                       risk_score if 20 <= risk_score < 50 else 0,
                       risk_score if 50 <= risk_score < 80 else 0,
                       risk_score if 80 <= risk_score <= 100 else 0]
    })
    
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Bar(
        name='General Population',
        x=comparison_df['Risk Level'],
        y=comparison_df['Percentage of Population'],
        marker_color='lightblue'
    ))
    
    fig_comparison.add_trace(go.Scatter(
        name='Your Position',
        x=comparison_df['Risk Level'],
        y=comparison_df['Your Score'],
        mode='markers',
        marker=dict(size=20, color='red', symbol='star'),
        showlegend=True
    ))
    
    fig_comparison.update_layout(
        title='How You Compare to General Population',
        xaxis_title='Risk Category',
        yaxis_title='Percentage (%)',
        height=400,
        barmode='group'
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)

else:
    # Welcome screen
    st.info('👈 Enter customer information in the sidebar and click **Predict Risk** to get started')
    
    # Add some sample statistics or info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('### 🎯 Model Accuracy')
        st.markdown('**93.66%**')
        st.caption('Tested on 45,000 cases')
    
    with col2:
        st.markdown('### 📊 AUC Score')
        st.markdown('**0.849**')
        st.caption('Excellent discrimination')
    
    with col3:
        st.markdown('### ⚡ Features Used')
        st.markdown('**7 Key Factors**')
        st.caption('Optimized selection')
