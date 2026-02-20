"""
🏦 BANK MARKETING INTELLIGENCE PLATFORM
Application avancée de prédiction et d'analyse pour campagnes marketing bancaires
Développée pour l'examen - Version Originale et Professionnelle
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

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="🏦 Bank Marketing Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS - DESIGN UNIQUE ====================
st.markdown("""
<style>
    /* Couleurs personnalisées - Thème bancaire moderne */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3b82f6;
        --accent-color: #10b981;
        --danger-color: #ef4444;
        --bg-dark: #0f172a;
    }
    
    /* Titre principal stylisé */
    .main-title {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    }
    
    /* Cards modernes */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* Boutons personnalisés */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e3a8a 100%);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(59, 130, 246, 0.4);
    }
    
    /* Sidebar moderne */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e3a8a 0%, #0f172a 100%);
    }
    
    /* Success/Warning boxes */
    .success-box {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated {
        animation: fadeIn 0.5s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HEADER UNIQUE ====================
st.markdown("""
<div class="main-title animated">
    <h1>🏦 BANK MARKETING INTELLIGENCE PLATFORM</h1>
    <p style='font-size: 1.2rem; margin-top: 1rem;'>
        Plateforme d'IA Avancée pour Optimisation de Campagnes Marketing
    </p>
    <p style='font-size: 0.9rem; opacity: 0.9;'>
        ⚡ Prédictions ML • 📊 Analytics en temps réel • 🎯 Ciblage intelligent
    </p>
</div>
""", unsafe_allow_html=True)

# ==================== CHARGEMENT DES DONNÉES ====================
@st.cache_data
def load_data():
    """Chargement optimisé des données avec preprocessing"""
    try:
        df = pd.read_csv('bank-full.csv', sep=';')
        
        # Nettoyer les guillemets
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.replace('"', '').str.strip()
        
        # Créer des features enrichies
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100], 
                                 labels=['<30', '30-40', '40-50', '50-60', '60+'])
        df['balance_category'] = pd.cut(df['balance'], bins=[-10000, 0, 1000, 5000, 100000],
                                       labels=['Négatif', 'Bas', 'Moyen', 'Élevé'])
        df['contact_intensity'] = df['campaign'] * df['duration'] / 1000
        
        return df
    except FileNotFoundError:
        st.error("⚠️ Fichier 'bank-full.csv' non trouvé. Uploadez-le ci-dessous.")
        return None

# Chargement
df = load_data()

if df is not None:
    
    # ==================== SIDEBAR - NAVIGATION MODERNE ====================
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/bank-building.png", width=80)
        st.title("🎛️ Navigation")
        
        page = st.radio(
            "Sélectionnez un module:",
            ["🏠 Dashboard Exécutif",
             "🔍 Exploration Intelligente",
             "🤖 Machine Learning Hub",
             "🎯 Prédicteur en Temps Réel",
             "📊 Analytics Avancés",
             "🏆 Comparaison de Modèles",
             "💾 Export & Déploiement"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Statistiques rapides
        st.markdown("### 📈 Statistiques Rapides")
        total_clients = len(df)
        conversion_rate = (df['y'] == 'yes').sum() / len(df) * 100
        
        st.metric("Total Clients", f"{total_clients:,}")
        st.metric("Taux de Conversion", f"{conversion_rate:.1f}%")
        st.metric("Features", len(df.columns) - 1)
        
        st.markdown("---")
        st.caption(f"🕒 Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    
    # ==================== PAGE 1: DASHBOARD EXÉCUTIF ====================
    if page == "🏠 Dashboard Exécutif":
        st.header("🏠 Dashboard Exécutif - Vue d'Ensemble")
        
        # KPIs principaux
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style='color: #3b82f6; margin: 0;'>👥 Clients</h3>
                <h2 style='margin: 0.5rem 0;'>{:,}</h2>
                <p style='color: #64748b; margin: 0;'>Total dans la base</p>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            conversion = (df['y'] == 'yes').sum()
            st.markdown("""
            <div class="metric-card">
                <h3 style='color: #10b981; margin: 0;'>✅ Conversions</h3>
                <h2 style='margin: 0.5rem 0;'>{:,}</h2>
                <p style='color: #64748b; margin: 0;'>{:.1f}% du total</p>
            </div>
            """.format(conversion, conversion/len(df)*100), unsafe_allow_html=True)
        
        with col3:
            avg_duration = df['duration'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3 style='color: #f59e0b; margin: 0;'>⏱️ Durée Moy.</h3>
                <h2 style='margin: 0.5rem 0;'>{:.0f}s</h2>
                <p style='color: #64748b; margin: 0;'>Par appel</p>
            </div>
            """.format(avg_duration), unsafe_allow_html=True)
        
        with col4:
            avg_campaign = df['campaign'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3 style='color: #ef4444; margin: 0;'>📞 Contacts</h3>
                <h2 style='margin: 0.5rem 0;'>{:.1f}</h2>
                <p style='color: #64748b; margin: 0;'>Par client</p>
            </div>
            """.format(avg_campaign), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Graphiques interactifs Plotly
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Taux de Conversion par Âge")
            
            age_conversion = df.groupby('age_group')['y'].apply(
                lambda x: (x == 'yes').sum() / len(x) * 100
            ).reset_index()
            age_conversion.columns = ['Groupe d\'âge', 'Taux (%)']
            
            fig = px.bar(age_conversion, x='Groupe d\'âge', y='Taux (%)',
                        color='Taux (%)',
                        color_continuous_scale='Blues',
                        title="Performance par Groupe d'Âge")
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💼 Distribution par Profession")
            
            job_dist = df['job'].value_counts().head(8)
            fig = px.pie(values=job_dist.values, names=job_dist.index,
                        title="Top 8 Professions",
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Timeline des contacts
        st.subheader("📅 Timeline des Campagnes")
        
        month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        df['month_cat'] = pd.Categorical(df['month'], categories=month_order, ordered=True)
        
        timeline = df.groupby(['month_cat', 'y']).size().unstack(fill_value=0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timeline.index, y=timeline['yes'], 
                                mode='lines+markers', name='Succès',
                                line=dict(color='#10b981', width=3),
                                marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=timeline.index, y=timeline['no'],
                                mode='lines+markers', name='Échec',
                                line=dict(color='#ef4444', width=3),
                                marker=dict(size=10)))
        
        fig.update_layout(title="Évolution Mensuelle des Résultats",
                         xaxis_title="Mois", yaxis_title="Nombre de Contacts",
                         height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights automatiques
        st.markdown("---")
        st.subheader("🔍 Insights Clés Générés par l'IA")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_month = df[df['y']=='yes']['month'].value_counts().index[0]
            st.info(f"""
            **📅 Meilleur Mois**  
            **{best_month.upper()}** a le plus de conversions.  
            Recommandation: Intensifier les campagnes ce mois.
            """)
        
        with col2:
            best_job = df[df['y']=='yes']['job'].value_counts().index[0]
            st.success(f"""
            **💼 Segment Clé**  
            **{best_job}** convertit le mieux.  
            ROI potentiel élevé sur ce segment.
            """)
        
        with col3:
            optimal_duration = df[df['y']=='yes']['duration'].median()
            st.warning(f"""
            **⏱️ Durée Optimale**  
            **{optimal_duration:.0f} secondes** en moyenne.  
            Former les agents sur cette durée.
            """)
    
    # ==================== PAGE 2: EXPLORATION INTELLIGENTE ====================
    elif page == "🔍 Exploration Intelligente":
        st.header("🔍 Exploration Intelligente des Données")
        
        tab1, tab2, tab3 = st.tabs(["📋 Vue Générale", "🎨 Visualisations", "🔬 Analyse Bivariée"])
        
        with tab1:
            st.subheader("📊 Aperçu du Dataset")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df.head(20), use_container_width=True, height=400)
            
            with col2:
                st.markdown("**📈 Statistiques**")
                st.write(f"**Lignes:** {df.shape[0]:,}")
                st.write(f"**Colonnes:** {df.shape[1]}")
                st.write(f"**Valeurs manquantes:** {df.isnull().sum().sum()}")
                st.write(f"**Taille mémoire:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                
                st.markdown("**🎯 Variable Cible**")
                target_dist = df['y'].value_counts()
                for cat, count in target_dist.items():
                    pct = count / len(df) * 100
                    st.write(f"**{cat}:** {count:,} ({pct:.1f}%)")
            
            st.markdown("---")
            st.subheader("📊 Statistiques Descriptives")
            st.dataframe(df.describe(), use_container_width=True)
        
        with tab2:
            st.subheader("🎨 Visualisations Interactives")
            
            viz_type = st.selectbox(
                "Choisir le type de visualisation:",
                ["Distribution des Variables", "Corrélations", "Box Plots Comparatifs"]
            )
            
            if viz_type == "Distribution des Variables":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                selected_col = st.selectbox("Variable à analyser:", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(df, x=selected_col, color='y',
                                      title=f"Distribution de {selected_col}",
                                      marginal='box',
                                      color_discrete_map={'yes': '#10b981', 'no': '#ef4444'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(df, x='y', y=selected_col, color='y',
                                title=f"Comparaison {selected_col} par Cible",
                                color_discrete_map={'yes': '#10b981', 'no': '#ef4444'})
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Corrélations":
                numeric_df = df.select_dtypes(include=[np.number])
                corr = numeric_df.corr()
                
                fig = px.imshow(corr, 
                               text_auto='.2f',
                               aspect='auto',
                               color_continuous_scale='RdBu_r',
                               title="Matrice de Corrélation")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # Box Plots
                numeric_cols = ['age', 'balance', 'duration', 'campaign']
                
                fig = make_subplots(rows=2, cols=2,
                                   subplot_titles=numeric_cols)
                
                for i, col in enumerate(numeric_cols):
                    row = i // 2 + 1
                    col_pos = i % 2 + 1
                    
                    for y_val, color in [('yes', '#10b981'), ('no', '#ef4444')]:
                        fig.add_trace(
                            go.Box(y=df[df['y']==y_val][col], name=y_val,
                                  marker_color=color),
                            row=row, col=col_pos
                        )
                
                fig.update_layout(height=600, showlegend=False,
                                 title_text="Comparaison des Variables par Résultat")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("🔬 Analyse Bivariée Avancée")
            
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            cat_cols.remove('y')
            
            selected_cat = st.selectbox("Choisir une variable catégorielle:", cat_cols)
            
            # Cross-tab avec taux de conversion
            crosstab = pd.crosstab(df[selected_cat], df['y'], normalize='index') * 100
            crosstab['Total'] = df[selected_cat].value_counts()
            crosstab = crosstab.sort_values('yes', ascending=False).head(10)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"**📊 Taux de Conversion par {selected_cat}**")
                st.dataframe(crosstab.style.background_gradient(subset=['yes'], cmap='Greens')
                           .format({'yes': '{:.1f}%', 'no': '{:.1f}%', 'Total': '{:.0f}'}))
            
            with col2:
                fig = px.bar(crosstab.reset_index(), x=selected_cat, y='yes',
                            title=f"Top 10 - Taux de Succès par {selected_cat}",
                            color='yes',
                            color_continuous_scale='Greens')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # Continuer dans le prochain message...