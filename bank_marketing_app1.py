"""
🏦 BANK MARKETING INTELLIGENCE PLATFORM
Application complète et fonctionnelle pour l'examen
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             roc_auc_score)
import pickle
from datetime import datetime

# Configuration
st.set_page_config(
    page_title="🏦 Bank Marketing Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Personnalisé
st.markdown("""
<style>
    .main-title {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e3a8a 100%);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-title">
    <h1>🏦 BANK MARKETING INTELLIGENCE PLATFORM</h1>
    <p style='font-size: 1.2rem; margin-top: 1rem;'>
        Plateforme d'IA Avancée pour Optimisation de Campagnes Marketing
    </p>
</div>
""", unsafe_allow_html=True)

# Chargement des données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('bank-full.csv', sep=';')
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.replace('"', '').str.strip()
        
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], 
                                 labels=['<30', '30-40', '40-50', '50-60', '60+'])
        df['balance_category'] = pd.cut(df['balance'], bins=[-10000, 0, 1000, 5000, 100000],
                                       labels=['Négatif', 'Bas', 'Moyen', 'Élevé'])
        return df
    except FileNotFoundError:
        st.error("⚠️ Fichier 'bank-full.csv' non trouvé. Uploadez-le.")
        uploaded = st.file_uploader("Upload bank-full.csv", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded, sep=';')
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].str.replace('"', '').str.strip()
            df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], 
                                     labels=['<30', '30-40', '40-50', '50-60', '60+'])
            df['balance_category'] = pd.cut(df['balance'], bins=[-10000, 0, 1000, 5000, 100000],
                                           labels=['Négatif', 'Bas', 'Moyen', 'Élevé'])
            return df
        return None

df = load_data()

if df is not None:
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Navigation")
        
        page = st.radio(
            "Sélectionnez:",
            ["🏠 Dashboard Exécutif",
             "🔍 Exploration",
             "🤖 ML Hub",
             "🎯 Prédicteur",
             "📊 Analytics",
             "🏆 Comparaison",
             "💾 Export"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.metric("Total Clients", f"{len(df):,}")
        conversion_rate = (df['y'] == 'yes').sum() / len(df) * 100
        st.metric("Taux Conversion", f"{conversion_rate:.1f}%")
        st.caption(f"🕒 {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # ==================== PAGE 1: DASHBOARD ====================
    if page == "🏠 Dashboard Exécutif":
        st.header("🏠 Dashboard Exécutif")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style='color: #3b82f6;'>👥 Clients</h3>
                <h2>{len(df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            conversion = (df['y'] == 'yes').sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style='color: #10b981;'>✅ Conversions</h3>
                <h2>{conversion:,}</h2>
                <p style='color: #64748b;'>{conversion/len(df)*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_duration = df['duration'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style='color: #f59e0b;'>⏱️ Durée Moy.</h3>
                <h2>{avg_duration:.0f}s</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_campaign = df['campaign'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style='color: #ef4444;'>📞 Contacts</h3>
                <h2>{avg_campaign:.1f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Conversion par Âge")
            age_conv = df.groupby('age_group')['y'].apply(
                lambda x: (x == 'yes').sum() / len(x) * 100
            ).reset_index()
            age_conv.columns = ['Groupe', 'Taux']
            
            fig = px.bar(age_conv, x='Groupe', y='Taux', color='Taux',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💼 Distribution Professions")
            job_dist = df['job'].value_counts().head(8)
            fig = px.pie(values=job_dist.values, names=job_dist.index, hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== PAGE 2: EXPLORATION ====================
    elif page == "🔍 Exploration":
        st.header("🔍 Exploration des Données")
        
        tab1, tab2 = st.tabs(["📋 Vue Générale", "🎨 Visualisations"])
        
        with tab1:
            st.dataframe(df.head(20), use_container_width=True)
            st.subheader("Statistiques")
            st.dataframe(df.describe())
        
        with tab2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_col = st.selectbox("Variable:", numeric_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=selected_col, color='y')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, x='y', y=selected_col, color='y')
                st.plotly_chart(fig, use_container_width=True)
    
    # ==================== PAGE 3: ML HUB ====================
    elif page == "🤖 ML Hub":
        st.header("🤖 Machine Learning Hub")
        
        # Préparation
        if 'ml_prepared' not in st.session_state:
            with st.spinner("🔄 Préparation..."):
                df_ml = df.copy()
                
                le_dict = {}
                for col in df_ml.select_dtypes(include=['object']).columns:
                    if col != 'y':
                        le = LabelEncoder()
                        df_ml[col] = le.fit_transform(df_ml[col])
                        le_dict[col] = le
                
                df_ml['y_encoded'] = (df_ml['y'] == 'yes').astype(int)
                
                X = df_ml.drop(['y', 'y_encoded', 'age_group', 'balance_category'], axis=1, errors='ignore')
                y = df_ml['y_encoded']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                st.session_state['X_train'] = X_train_scaled
                st.session_state['X_test'] = X_test_scaled
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['feature_names'] = X.columns.tolist()
                st.session_state['scaler'] = scaler
                st.session_state['le_dict'] = le_dict
                st.session_state['ml_prepared'] = True
        
        st.success("✅ Données prêtes!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎓 Train", f"{len(st.session_state['X_train']):,}")
        with col2:
            st.metric("🧪 Test", f"{len(st.session_state['X_test']):,}")
        with col3:
            st.metric("📊 Features", len(st.session_state['feature_names']))
        
        st.markdown("---")
        
        models_to_train = st.multiselect(
            "Modèles à entraîner:",
            ["Logistic Regression", "Random Forest", "Gradient Boosting", 
             "AdaBoost", "Ensemble"],
            default=["Random Forest", "Gradient Boosting"]
        )
        
        if st.button("🎯 LANCER L'ENTRAÎNEMENT"):
            X_train = st.session_state['X_train']
            X_test = st.session_state['X_test']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']
            
            results = []
            trained_models = {}
            
            models_dict = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42)
            }
            
            progress = st.progress(0)
            
            for i, model_name in enumerate(models_to_train):
                if model_name == "Ensemble":
                    estimators = [
                        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                        ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
                        ('lr', LogisticRegression(max_iter=1000, random_state=42))
                    ]
                    model = VotingClassifier(estimators=estimators, voting='soft')
                else:
                    model = models_dict[model_name]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc = roc_auc_score(y_test, y_pred_proba)
                
                results.append({
                    'Modèle': model_name,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1-Score': f1,
                    'ROC-AUC': roc
                })
                
                trained_models[model_name] = model
                progress.progress((i + 1) / len(models_to_train))
            
            results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
            
            st.markdown("### 🏆 Résultats")
            st.dataframe(
                results_df.style.format({
                    'Accuracy': '{:.2%}',
                    'Precision': '{:.2%}',
                    'Recall': '{:.2%}',
                    'F1-Score': '{:.2%}',
                    'ROC-AUC': '{:.4f}'
                }).background_gradient(subset=['ROC-AUC'], cmap='Greens')
            )
            
            # Graphique radar
            fig = go.Figure()
            for _, row in results_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['Accuracy'], row['Precision'], row['Recall'], 
                       row['F1-Score'], row['ROC-AUC']],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
                    fill='toself',
                    name=row['Modèle']
                ))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1])),
                            title="Comparaison Multi-Critères")
            st.plotly_chart(fig, use_container_width=True)
            
            # Sauvegarder
            best_model_name = results_df.iloc[0]['Modèle']
            st.session_state['best_model'] = trained_models[best_model_name]
            st.session_state['best_model_name'] = best_model_name
            st.session_state['all_models'] = trained_models
            
            st.success(f"🏆 Meilleur: {best_model_name} (ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f})")
    
    # ==================== PAGE 4: PRÉDICTEUR ====================
    elif page == "🎯 Prédicteur":
        st.header("🎯 Prédicteur en Temps Réel")
        
        if 'best_model' not in st.session_state:
            st.warning("⚠️ Entraînez d'abord un modèle dans le ML Hub")
        else:
            st.info("✨ Utilisez ce formulaire pour prédire la conversion")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 👤 Client")
                age = st.slider("Âge", 18, 95, 40)
                job = st.selectbox("Profession", sorted(df['job'].unique()))
                marital = st.selectbox("Situation", sorted(df['marital'].unique()))
                education = st.selectbox("Éducation", sorted(df['education'].unique()))
            
            with col2:
                st.markdown("#### 💰 Finances")
                balance = st.number_input("Solde (€)", -10000, 100000, 1000, 100)
                default = st.selectbox("Défaut?", ['no', 'yes'])
                housing = st.selectbox("Prêt Immo?", ['no', 'yes'])
                loan = st.selectbox("Prêt Perso?", ['no', 'yes'])
            
            with col3:
                st.markdown("#### 📞 Campagne")
                contact = st.selectbox("Contact", sorted(df['contact'].unique()))
                month = st.selectbox("Mois", ['jan','feb','mar','apr','may','jun',
                                            'jul','aug','sep','oct','nov','dec'])
                day = st.slider("Jour", 1, 31, 15)
                duration = st.slider("Durée (s)", 0, 5000, 300, 10)
            
            col1, col2 = st.columns(2)
            with col1:
                campaign = st.slider("Contacts", 1, 50, 2)
                pdays = st.number_input("Jours (-1=jamais)", -1, 999, -1)
            with col2:
                previous = st.slider("Campagnes préc.", 0, 50, 0)
                poutcome = st.selectbox("Résultat préc.", sorted(df['poutcome'].unique()))
            
            if st.button("🔮 PRÉDIRE", use_container_width=True):
                input_data = {
                    'age': age, 'job': job, 'marital': marital, 'education': education,
                    'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
                    'contact': contact, 'day': day, 'month': month, 'duration': duration,
                    'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
                }
                
                input_df = pd.DataFrame([input_data])
                
                le_dict = st.session_state['le_dict']
                for col in input_df.select_dtypes(include=['object']).columns:
                    if col in le_dict:
                        input_df[col] = le_dict[col].transform(input_df[col])
                
                scaler = st.session_state['scaler']
                input_scaled = scaler.transform(input_df)
                
                model = st.session_state['best_model']
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0]
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if prediction == 1:
                    st.balloons()
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                                padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
                        <h1>✅ CONVERSION PROBABLE!</h1>
                        <h2>Probabilité: {proba[1]*100:.1f}%</h2>
                        <p style='font-size: 1.2rem;'>Contact prioritaire recommandé!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                                padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
                        <h1>❌ CONVERSION PEU PROBABLE</h1>
                        <h2>Probabilité: {proba[1]*100:.1f}%</h2>
                        <p style='font-size: 1.2rem;'>Revoir le ciblage</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba[1]*100,
                    title={'text': "Score de Conversion"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#3b82f6"},
                        'steps': [
                            {'range': [0, 30], 'color': '#fee2e2'},
                            {'range': [30, 70], 'color': '#fef3c7'},
                            {'range': [70, 100], 'color': '#d1fae5'}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'value': 50}}))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # ==================== PAGE 5: ANALYTICS ====================
    elif page == "📊 Analytics":
        st.header("📊 Analytics Avancés")
        
        if 'best_model' not in st.session_state:
            st.warning("⚠️ Entraînez d'abord un modèle")
        else:
            model = st.session_state['best_model']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            feature_names = st.session_state['feature_names']
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(15)
                
                fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
                st.plotly_chart(fig, use_container_width=True)
            
            # Matrice de confusion
            st.subheader("📊 Matrice de Confusion")
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(cm, text_auto=True, 
                           labels=dict(x="Prédiction", y="Réel", color="Nombre"),
                           x=['No', 'Yes'], y=['No', 'Yes'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Rapport
            st.subheader("📋 Rapport Détaillé")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
    
    # ==================== PAGE 6: COMPARAISON ====================
    elif page == "🏆 Comparaison":
        st.header("🏆 Comparaison de Modèles")
        
        if 'all_models' not in st.session_state:
            st.warning("⚠️ Entraînez d'abord les modèles")
        else:
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            models = st.session_state['all_models']
            
            st.subheader("📊 Courbes ROC")
            
            fig = go.Figure()
            
            for name, model in models.items():
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f'{name} (AUC={roc_auc:.3f})',
                    mode='lines'
                ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                name='Random',
                mode='lines',
                line=dict(dash='dash', color='gray')
            ))
            
            fig.update_layout(
                title='Courbes ROC',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau AUC
            st.subheader("📈 Scores AUC")
            auc_scores = []
            for name, model in models.items():
                y_proba = model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_proba)
                auc_scores.append({'Modèle': name, 'AUC': auc_score})
            
            auc_df = pd.DataFrame(auc_scores).sort_values('AUC', ascending=False)
            st.dataframe(auc_df.style.format({'AUC': '{:.4f}'})
                        .background_gradient(subset=['AUC'], cmap='Greens'))
    
    # ==================== PAGE 7: EXPORT ====================
    elif page == "💾 Export":
        st.header("💾 Export & Déploiement")
        
        if 'best_model' not in st.session_state:
            st.warning("⚠️ Entraînez d'abord un modèle")
        else:
            model = st.session_state['best_model']
            model_name = st.session_state['best_model_name']
            
            st.subheader(f"🏆 Modèle Sélectionné: {model_name}")
            
            # Performances
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            score = model.score(X_test, y_test)
            st.metric("Accuracy", f"{score*100:.2f}%")
            
            # Sauvegarde
            filename = st.text_input("Nom du fichier", "bank_model.pkl")
            
            if st.button("💾 Sauvegarder le Modèle"):
                try:
                    with open(filename, 'wb') as f:
                        pickle.dump({
                            'model': model,
                            'scaler': st.session_state['scaler'],
                            'le_dict': st.session_state['le_dict'],
                            'features': st.session_state['feature_names']
                        }, f)
                    
                    st.success(f"✅ Modèle sauvegardé: {filename}")
                    
                    with open(filename, 'rb') as f:
                        st.download_button(
                            "📥 Télécharger",
                            f,
                            filename,
                            mime="application/octet-stream"
                        )
                    
                    st.info("""
                    **Utilisation:**
                    ```python
                    import pickle
                    with open('bank_model.pkl', 'rb') as f:
                        data = pickle.load(f)
                    model = data['model']
                    predictions = model.predict(X_new)
                    ```
                    """)
                except Exception as e:
                    st.error(f"Erreur: {e}")

st.markdown("---")
st.markdown("💻 **Bank Marketing Intelligence** | 🎓 Développé pour l'examen")
