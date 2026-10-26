import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import google.generativeai as genai

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="PEA Quant Lab & AI", page_icon="🏛️", layout="wide")

st.title("🏛️ PEA Quant Lab : Simulateur Boursier Réel & IA Académique")
st.markdown("Construisez votre portefeuille avec de **vraies actions/ETF éligibles au PEA**, backtestez vos stratégies sur les vraies crises passées et auditez vos risques avec l'IA.")

# --- CATALOGUE D'ACTIFS RÉELS ÉLIGIBLES PEA ---
ACTIFS_PEA = {
    "ETF S&P 500 (BNP Easy - ESE.PA)": {"ticker": "ESE.PA", "frais": 0.0015, "categorie": "ETF US (Synthétique PEA)"},
    "ETF MSCI World (Amundi - CW8.PA)": {"ticker": "CW8.PA", "frais": 0.0038, "categorie": "ETF Monde (Synthétique PEA)"},
    "ETF Nasdaq-100 (Amundi - PANX.PA)": {"ticker": "PANX.PA", "frais": 0.0030, "categorie": "ETF Tech US (Synthétique PEA)"},
    "LVMH (MC.PA)": {"ticker": "MC.PA", "frais": 0.0, "categorie": "Action Luxe (CAC 40)"},
    "TotalEnergies (TTE.PA)": {"ticker": "TTE.PA", "frais": 0.0, "categorie": "Action Énergie (CAC 40)"},
    "Air Liquide (AI.PA)": {"ticker": "AI.PA", "frais": 0.0, "categorie": "Action Industrie (CAC 40)"},
    "ASML Holding (ASML.AS)": {"ticker": "ASML.AS", "frais": 0.0, "categorie": "Action Semi-conducteurs (Europe)"},
    "Hermès (RMS.PA)": {"ticker": "RMS.PA", "frais": 0.0, "categorie": "Action Luxe (CAC 40)"},
    "Schneider Electric (SU.PA)": {"ticker": "SU.PA", "frais": 0.0, "categorie": "Action Industrie (CAC 40)"},
    "Sanofi (SAN.PA)": {"ticker": "SAN.PA", "frais": 0.0, "categorie": "Action Santé (CAC 40)"}
}

# --- FONCTION DE TÉLÉCHARGEMENT DES DONNÉES BOURSIÈRES EN DIRECT ---
@st.cache_data(ttl=3600)
def charger_donnees_historiques(tickers_list, periode_annees=10):
    data = yf.download(tickers_list, period=f"{periode_annees}y", interval="1mo", auto_adjust=True)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers_list[0])
    data = data.dropna(how='all').ffill().bfill()
    return data

# --- BARRE LATÉRALE : CONSTRUCTEUR DE PORTEFEUILLE ---
st.sidebar.header("🛠️ 1. Votre Portefeuille PEA")
choix_actifs = st.sidebar.multiselect(
    "Choisissez vos Actions / ETF réels :",
    options=list(ACTIFS_PEA.keys()),
    default=["ETF S&P 500 (BNP Easy - ESE.PA)", "LVMH (MC.PA)", "Air Liquide (AI.PA)"]
)

if not choix_actifs:
    st.warning("⚠️ Veuillez sélectionner au moins un actif dans le menu de gauche pour lancer le simulateur.")
    st.stop()

# Gestion des pondérations (%)
st.sidebar.subheader("⚖️ Répartition de votre panier")
poids_bruts = {}
for actif in choix_actifs:
    poids_bruts[actif] = st.sidebar.slider(f"Part de {actif.split(' (')[0]}", min_value=5, max_value=100, value=100 // len(choix_actifs), step=5)

total_poids = sum(poids_bruts.values())
poids_normalises = {k: v / total_poids for k, v in poids_bruts.items()}

st.sidebar.header("💶 2. Vos Versements")
capital_initial = st.sidebar.number_input("Capital de départ (€)", min_value=0, value=2000, step=500)
versement_mensuel = st.sidebar.number_input("Versement mensuel DCA (€)", min_value=50, value=300, step=50)
horizon_futur = st.sidebar.slider("Horizon de projection future (Années)", min_value=5, max_value=35, value=15)
annees_backtest = st.sidebar.selectbox("Période d'analyse historique réelle", [5, 8, 10, 12], index=2)

# --- CHARGEMENT ET CALCULS MATHÉMATIQUES EN DIRECT ---
tickers_selectionnes = [ACTIFS_PEA[a]["ticker"] for a in choix_actifs]
with st.spinner("Connexion aux marchés financiers (Yahoo Finance) en cours..."):
    df_prix = charger_donnees_historiques(tickers_selectionnes, annees_backtest)

# Calcul des rendements mensuels réels de chaque actif
rendements_actifs = df_prix.pct_change().dropna()

# Calcul du rendement mensuel pondéré du portefeuille global
rendement_portefeuille_mensuel = pd.Series(0.0, index=rendements_actifs.index)
frais_moyens_ponderees = 0.0

stats_individuelles = []
for actif in choix_actifs:
    tkr = ACTIFS_PEA[actif]["ticker"]
    w = poids_normalises[actif]
    frais_moyens_ponderees += ACTIFS_PEA[actif]["frais"] * w
    
    if tkr in rendements_actifs.columns:
        r_col = rendements_actifs[tkr]
        rendement_portefeuille_mensuel += r_col * w
        
        # CAGR réel (Croissance annuelle composée) & Volatilité
        nb_annees_reelles = len(df_prix[tkr].dropna()) / 12
        cagr_actif = (df_prix[tkr].iloc[-1] / df_prix[tkr].iloc[0]) ** (1 / nb_annees_reelles) - 1
        vol_actif = r_col.std() * np.sqrt(12)
        prix_spot = df_prix[tkr].iloc[-1]
        
        stats_individuelles.append({
            "Actif": actif,
            "Catégorie": ACTIFS_PEA[actif]["categorie"],
            "Poids (%)": f"{w*100:.1f} %",
            "Dernier Prix": f"{prix_spot:,.2f} €",
            "Croissance Moyenne Réelle (CAGR)": f"{cagr_actif*100:.2f} % / an",
            "Volatilité Annuelle": f"{vol_actif*100:.2f} %",
            "Frais (TER)": f"{ACTIFS_PEA[actif]['frais']*100:.2f} %"
        })

# Métriques globales réelles du portefeuille
cagr_portefeuille = (1 + rendement_portefeuille_mensuel.mean()) ** 12 - 1
vol_portefeuille = rendement_portefeuille_mensuel.std() * np.sqrt(12)
taux_sans_risque = 0.03 # Taux sans risque de référence (3%)

# Ratio de Sharpe & Ratio de Sortino (Ne pénalise que les baisses)
sharpe_ratio = (cagr_portefeuille - taux_sans_risque) / vol_portefeuille if vol_portefeuille > 0 else 0
rendements_negatifs = rendement_portefeuille_mensuel[rendement_portefeuille_mensuel < 0]
downside_dev = rendements_negatifs.std() * np.sqrt(12) if len(rendements_negatifs) > 0 else 0.001
sortino_ratio = (cagr_portefeuille - taux_sans_risque) / downside_dev

# Calcul du Max Drawdown (Pire chute historique)
cum_returns = (1 + rendement_portefeuille_mensuel).cumprod()
peak = cum_returns.cummax()
drawdown = (cum_returns - peak) / peak
max_drawdown = drawdown.min()

# --- INTERFACE EN ONGLETS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. Portefeuille & Marché Réel",
    "⏪ 2. Backtest Historique (DCA vs LSI vs SmartDCA)",
    "🚀 3. Projection, Fiscalité PEA & Plafond",
    "🏖️ 4. Simulateur de Rente (Monte Carlo)",
    "🤖 5. Audit IA (Cerveau RAG)"
])

# ==============================================================================
# ONGLET 1 : PORTEFEUILLE & STATS RÉELLES
# ==============================================================================
with tab1:
    st.subheader(f"📈 Performances historiques réelles sur les {annees_backtest} dernières années")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rendement Annuel Moyen Réel", f"{cagr_portefeuille*100:.2f} %", help="Calculé sur les vraies cotations boursières de votre sélection.")
    c2.metric("Volatilité Annuelle (Risque)", f"{vol_portefeuille*100:.2f} %", f"Max Crash: {max_drawdown*100:.1f} %", delta_color="inverse")
    c3.metric("Ratio de Sharpe", f"{sharpe_ratio:.2f}", help="Mesure la rentabilité par rapport au risque total (>1 = Très bon).")
    c4.metric("Ratio de Sortino", f"{sortino_ratio:.2f}", help="Issu de la théorie comportementale : ne pénalise que la volatilité à la baisse.")

    st.dataframe(pd.DataFrame(stats_individuelles), use_container_width=True, hide_index=True)
    
    st.subheader("Évolution comparée de vos actifs (Base 100)")
    df_base100 = (df_prix / df_prix.iloc[0]) * 100
    st.line_chart(df_base100)

# ==============================================================================
# ONGLET 2 : BACKTESTER HISTORIQUE & DUEL DE STRATÉGIES
# ==============================================================================
with tab2:
    st.subheader(f"⏪ Machine à remonter le temps : Si vous aviez investi il y a {annees_backtest} ans...")
    st.markdown("Comparaison sur les **vraies données du marché** entre 3 stratégies académiques : **DCA Classique**, **SmartDCA** (acheter +25% quand le marché baisse de plus de 3% dans le mois) et **Lump Sum** (investir la totalité du budget dès le 1er jour).")
    
    nb_mois_hist = len(rendement_portefeuille_mensuel)
    budget_total_hist = capital_initial + (versement_mensuel * nb_mois_hist)
    
    val_dca = capital_initial
    val_smart = capital_initial
    val_lsi = budget_total_hist # En Lump Sum, tout le capital est placé au jour 1
    
    inv_dca = capital_initial
    inv_smart = capital_initial
    
    hist_backtest = []
    
    for date, r_mois in rendement_portefeuille_mensuel.items():
        r_net = r_mois - (frais_moyens_ponderees / 12)
        
        # 1. DCA Classique
        val_dca = val_dca * (1 + r_net) + versement_mensuel
        inv_dca += versement_mensuel
        
        # 2. SmartDCA (Calvet et al. : ajuste le versement selon la baisse du marché)
        if r_mois < -0.03:
            apport_smart = versement_mensuel * 1.25 # On achète plus pendant les soldes boursières
        elif r_mois > 0.04:
            apport_smart = versement_mensuel * 0.85 # On freine quand le marché s'emballe
        else:
            apport_smart = versement_mensuel
        val_smart = val_smart * (1 + r_net) + apport_smart
        inv_smart += apport_smart
        
        # 3. Lump Sum (LSI)
        val_lsi = val_lsi * (1 + r_net)
        
        hist_backtest.append({
            "Date": date,
            "Argent Investi (DCA)": round(inv_dca, 0),
            "Portefeuille DCA Classique": round(val_dca, 0),
            "Portefeuille SmartDCA": round(val_smart, 0),
            "Lump Sum (Tout au Jour 1)": round(val_lsi, 0)
        })
        
    df_bt = pd.DataFrame(hist_backtest).set_index("Date")
    
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Total Versé (DCA)", f"{inv_dca:,.0f} €".replace(",", " "))
    b2.metric("Valeur Finale DCA", f"{val_dca:,.0f} €".replace(",", " "), f"+{val_dca - inv_dca:,.0f} €".replace(",", " "))
    b3.metric("Valeur Finale SmartDCA", f"{val_smart:,.0f} €".replace(",", " "), f"Gain net: +{val_smart - inv_smart:,.0f} €".replace(",", " "))
    b4.metric("Valeur Finale Lump Sum", f"{val_lsi:,.0f} €".replace(",", " "), f"+{val_lsi - budget_total_hist:,.0f} €".replace(",", " "))
    
    st.line_chart(df_bt)

# ==============================================================================
# ONGLET 3 : PROJECTION FUTURE, SCANNER FISCAL PEA & PLAFOND 150 000 €
# ==============================================================================
with tab3:
    st.subheader(f"🚀 Projection sur {horizon_futur} ans (Basée sur le rendement réel de votre panier : {cagr_portefeuille*100:.2f}%/an)")
    
    col_inf, col_frais = st.columns(2)
    inflation_annuelle = col_inf.slider("Taux d'inflation annuel estimé (%)", 0.0, 6.0, 2.0, 0.5) / 100
    frais_banque_trad = col_frais.slider("Frais d'une banque classique à comparer (%)", 0.5, 3.5, 2.0, 0.25) / 100

    taux_mensuel_net = (1 + (cagr_portefeuille - frais_moyens_ponderees)) ** (1/12) - 1
    taux_mensuel_banque = (1 + (cagr_portefeuille - frais_banque_trad)) ** (1/12) - 1
    
    cap_pea_brut = capital_initial
    cap_banque = capital_initial
    cumul_versements = capital_initial
    annee_plafond_atteint = None
    
    proj_data = []
    
    for m in range(1, horizon_futur * 12 + 1):
        # Vérification de la règle légale du plafond PEA (150 000 € de versements max)
        if cumul_versements + versement_mensuel <= 150000:
            apport_effectif = versement_mensuel
        else:
            apport_effectif = max(0, 150000 - cumul_versements)
            if annee_plafond_atteint is None and apport_effectif == 0:
                annee_plafond_atteint = round(m / 12, 1)
                
        cumul_versements += apport_effectif
        cap_pea_brut = cap_pea_brut * (1 + taux_mensuel_net) + apport_effectif
        cap_banque = cap_banque * (1 + taux_mensuel_banque) + apport_effectif
        
        if m % 12 == 0:
            an = m // 12
            plus_value_brute = max(0, cap_pea_brut - cumul_versements)
            
            # Fiscalité française : PEA (>5 ans = 17.2% PS, <5 ans = 30% Flat Tax) vs CTO (30% Flat Tax toujours)
            taux_impot_pea = 0.172 if an >= 5 else 0.30
            net_pea = cumul_versements + plus_value_brute * (1 - taux_impot_pea)
            net_cto = cumul_versements + plus_value_brute * (1 - 0.30)
            
            # Ajustement à l'inflation (Pouvoir d'achat réel selon la formule académique)
            pouvoir_achat_reel = net_pea / ((1 + inflation_annuelle) ** an)
            
            proj_data.append({
                "Année": an,
                "Versements Cumulés": round(cumul_versements, 0),
                "PEA Net d'Impôts": round(net_pea, 0),
                "Compte-Titres (CTO) Net": round(net_cto, 0),
                "Pouvoir d'Achat Réel (Hors Inflation)": round(pouvoir_achat_reel, 0),
                "Banque Classique (Frais élevés)": round(cap_banque, 0)
            })
            
    df_proj = pd.DataFrame(proj_data).set_index("Année")
    
    # Alerte Plafond PEA 150 000 €
    pourcentage_plafond = min(100.0, (cumul_versements / 150000) * 100)
    st.progress(pourcentage_plafond / 100, text=f"Jauge du Plafond Légal PEA : {cumul_versements:,.0f} € / 150 000 € versés ({pourcentage_plafond:.1f}%)")
    if annee_plafond_atteint:
        st.info(f"🎯 **Plafond légal de 150 000 € atteint au bout de {annee_plafond_atteint} ans !** Passé cette date, le simulateur coupe automatiquement vos versements mensuels, mais vos intérêts composés continuent de travailler tout seuls.")

    p1, p2, p3, p4 = st.columns(4)
    derniere_ligne = df_proj.iloc[-1]
    economie_fiscale = derniere_ligne["PEA Net d'Impôts"] - derniere_ligne["Compte-Titres (CTO) Net"]
    economie_frais = cap_pea_brut - derniere_ligne["Banque Classique (Frais élevés)"]
    
    p1.metric("Capital PEA (Net d'impôts)", f"{derniere_ligne['PEA Net d\'Impôts']:,.0f} €".replace(",", " "))
    p2.metric("Pouvoir d'Achat Réel (Inflation déduite)", f"{derniere_ligne['Pouvoir d\'Achat Réel (Hors Inflation)']:,.0f} €".replace(",", " "))
    p3.metric("Cadeau Fiscal du PEA (vs CTO)", f"+{economie_fiscale:,.0f} €".replace(",", " "), help="Économie réalisée grâce à l'exonération d'impôt sur le revenu après 5 ans.")
    p4.metric("Frais Sauvés (ETF vs Banque)", f"+{economie_frais:,.0f} €".replace(",", " "))
    
    st.area_chart(df_proj[["Versements Cumulés", "Compte-Titres (CTO) Net", "PEA Net d'Impôts", "Pouvoir d'Achat Réel (Hors Inflation)"]])

# ==============================================================================
# ONGLET 4 : SIMULATEUR DE RETRAITE / RENTE (MONTE CARLO)
# ==============================================================================
with tab4:
    st.subheader("🏖️ Estimation du Succès de Retrait (Simulation de Monte Carlo)")
    st.markdown("Inspiré des modèles de *Withdrawal Success Estimation* : une fois votre capital accumulé au bout de vos **" + str(horizon_futur) + " ans**, combien pouvez-vous retirer chaque mois sans jamais tomber à zéro ?")
    
    capital_depart_retraite = df_proj.iloc[-1]["PEA Net d'Impôts"]
    c_ret1, c_ret2 = st.columns(2)
    rente_mensuelle = c_ret1.number_input("Rente mensuelle souhaitée à la retraite (€/mois)", min_value=200, value=1500, step=100)
    duree_retraite = c_ret2.slider("Durée de la retraite à financer (Années)", 10, 40, 25)
    
    # Lancement de 500 simulations aléatoires de Monte Carlo
    nb_simulations = 500
    succes_count = 0
    courbes_mc = {}
    
    np.random.seed(42)
    for sim in range(nb_simulations):
        w_k = capital_depart_retraite
        trajectoire = [w_k]
        echec = False
        for a in range(1, duree_retraite + 1):
            # Tirage aléatoire selon la loi normale des rendements réels du portefeuille
            r_sim = np.random.normal(cagr_portefeuille - frais_moyens_ponderees, vol_portefeuille)
            w_k = w_k * (1 + r_sim) - (rente_mensuelle * 12)
            if w_k <= 0:
                w_k = 0
                echec = True
            trajectoire.append(round(w_k, 0))
        if not echec:
            succes_count += 1
        if sim < 30: # On affiche 30 trajectoires sur le graphique pour la lisibilité
            courbes_mc[f"Scénario {sim+1}"] = trajectoire
            
    taux_succes = (succes_count / nb_simulations) * 100
    
    m1, m2 = st.columns(2)
    m1.metric("Capital disponible au départ de la retraite", f"{capital_depart_retraite:,.0f} €".replace(",", " "))
    m2.metric("Probabilité de Succès (Capital > 0 €)", f"{taux_succes:.1f} %", help="Pourcentage des 500 scénarios boursiers où votre argent ne s'épuise jamais.")
    
    if taux_succes >= 85:
        st.success(f"✅ Excellente sécurité : Dans **{taux_succes:.1f}%** des scénarios de marché, votre portefeuille survit pendant {duree_retraite} ans tout en vous versant {rente_mensuelle} €/mois.")
    elif taux_succes >= 50:
        st.warning(f"⚠️ Risque modéré ({taux_succes:.1f}% de réussite) : En cas de krach boursier prolongé au début de votre retraite, votre capital pourrait s'épuiser.")
    else:
        st.error(f"🚨 Danger ({taux_succes:.1f}% de réussite) : La rente demandée ({rente_mensuelle} €/mois) est trop élevée par rapport au capital accumulé.")
        
    st.line_chart(pd.DataFrame(courbes_mc))

# ==============================================================================
# ONGLET 5 : AUDIT IA CONNECTÉ AU CERVEAU RAG & AU PORTEFEUILLE
# ==============================================================================
with tab5:
    st.subheader("🤖 Audit Scientifique de votre Portefeuille par l'IA")
    st.markdown("L'IA analyse **automatiquement** les actions que vous avez sélectionnées à gauche, vos ratios de Sharpe/Sortino et utilise les études académiques de notre base de connaissances (RAG).")
    
    api_key = st.text_input("🔑 Entrez votre clé API Google Gemini :", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)
        
        question_perso = st.text_input(
            "Une question précise ? (Ou laissez vide pour un audit complet automatique de votre portefeuille) :",
            placeholder="Ex: Mon portefeuille est-il trop risqué ? Pourquoi le SmartDCA bat-il mon DCA ?"
        )
        
        if st.button("🔍 Lancer l'Audit IA de mon Portefeuille"):
            with st.spinner("Consultation du Cerveau RAG et analyse mathématique en cours..."):
                # Base de connaissances RAG directement embarquée dans le système de l'IA
                cerveau_rag = f"""
                Tu es l'Ingénieur Financier IA du Micro-SaaS 'PEA Quant Lab'. 
                Tu dois analyser les données EXACTES du portefeuille de l'utilisateur en citant obligatoirement nos sources académiques et réglementaires internes :
                
                1. DONNÉES EN DIRECT DU PORTEFEUILLE DE L'UTILISATEUR :
                - Actifs sélectionnés et poids : {stats_individuelles}
                - Capital initial : {capital_initial} € | Versement mensuel DCA : {versement_mensuel} €/mois sur {horizon_futur} ans.
                - Rendement annuel moyen réel (CAGR) : {cagr_portefeuille*100:.2f} % / an
                - Volatilité annuelle : {vol_portefeuille*100:.2f} % | Pire krach historique (Max Drawdown) : {max_drawdown*100:.1f} %
                - Ratio de Sharpe : {sharpe_ratio:.2f} | Ratio de Sortino : {sortino_ratio:.2f}
                - Résultat Backtest ({annees_backtest} ans) : DCA Classique = {val_dca:,.0f} €, SmartDCA = {val_smart:,.0f} €, Lump Sum = {val_lsi:,.0f} €.
                - Capital projeté PEA Net d'impôts : {derniere_ligne['PEA Net d\'Impôts']:,.0f} € (Économie fiscale vs CTO : +{economie_fiscale:,.0f} €).
                - Taux de succès de la rente ({rente_mensuelle} €/mois) : {taux_succes:.1f} %.

                2. CONNAISSANCES ACADÉMIQUES & RÉGLEMENTAIRES (CERVEAU RAG À CITER) :
                - Intérêts composés : Formule Cn = Co(1+i)^n et règle des 72.
                - Études sur le DCA vs Lump Sum : Selon Constantinides (1979) et Leggio & Lien (2001), le Lump Sum (LSI) bat souvent le DCA dans un marché haussier, mais selon Stephen Brincks (Kennesaw State University, 2025), Statman (1995) et Dubil (2004), le DCA réduit drastiquement le risque psychologique de regret et la probabilité de perte sur le long terme (moins de 2,5% à 40 ans selon Hayden Brown, 2021).
                - SmartDCA : Selon Calvet et al. (2023), moduler ses versements selon les baisses du marché (SmartDCA) surperforme le DCA traditionnel.
                - Mesure du risque : Selon Sortino & Price (1994) et la théorie des perspectives de Kahneman & Tversky (1979), le Ratio de Sortino est plus pertinent que le Ratio de Sharpe car les investisseurs ne craignent que la volatilité négative (downside deviation).
                - Réglementation PEA (AMF / Service-Public) : Plafond de versement de 150 000 €, exonération d'impôt sur le revenu après 5 ans (seuls les prélèvements sociaux de 17,2% s'appliquent contre 30% de Flat Tax sur CTO). Les ETF synthétiques (comme BNP Easy S&P 500 à 0,15% de frais ou Amundi) permettent d'investir hors d'Europe tout en respectant l'éligibilité PEA.
                """
                
                demande = question_perso if question_perso else "Réalise un audit structuré complet de mon portefeuille actuel : analyse ma répartition, mes ratios de Sharpe et Sortino, compare mes résultats DCA/SmartDCA/Lump Sum, et donne-moi 3 conseils concrets basés sur les études académiques."
                prompt_final = f"{cerveau_rag}\n\nDemande de l'utilisateur : {demande}"
                
                # Boucle intelligente anti-erreur 404 : teste automatiquement les modèles Gemini compatibles
                modeles_a_tester = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash-latest', 'gemini-1.5-pro-latest', 'gemini-pro']
                reponse_texte = None
                
                for nom_modele in modeles_a_tester:
                    try:
                        model = genai.GenerativeModel(nom_modele)
                        res = model.generate_content(prompt_final)
                        reponse_texte = res.text
                        st.caption(f"⚡ Généré avec succès via le modèle `{nom_modele}`")
                        break
                    except Exception:
                        continue
                        
                if reponse_texte:
                    st.markdown(reponse_texte)
                else:
                    st.error("Impossible de joindre un modèle Gemini. Vérifiez que votre clé API est valide sur Google AI Studio.")
    else:
        st.info("👆 Entrez votre clé API Gemini ci-dessus pour débloquer l'audit intelligent de votre portefeuille.")