import streamlit as st
import google.generativeai as genai
import json
import re
import unicodedata
from collections import Counter
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------
# 1. CONFIGURAÇÕES INICIAIS E DICIONÁRIOS
# ---------------------------------------------------------
st.set_page_config(page_title="Analisador de Músicas", page_icon="🎵", layout="wide")

DICTIONARIES = {
    'teocentrico': ['deus', 'jesus', 'cristo', 'espirito', 'santo', 'senhor', 'salvador', 'cruz', 'graca', 'sangue', 'cordeiro', 'redentor', 'evangelho', 'trindade', 'pai', 'eterno', 'majestade', 'gloria', 'ressurreicao', 'reino', 'justica', 'noivo', 'criou', 'palavra', 'escudo', 'ceu', 'jeova', 'messias', 'emanuel', 'soberano', 'altissimo'],
    'antropocentrico': ['eu', 'meu', 'minha', 'mim', 'me', 'nos', 'vitoria', 'conquista', 'campeao', 'trofeu', 'vencer', 'venceu', 'sucesso', 'exaltado', 'exaltar', 'promessa', 'escolhido', 'historia', 'palco', 'plateia', 'merecer'],
    'sentimental': ['coracao', 'sentir', 'sinto', 'choro', 'chorando', 'lagrima', 'emocao', 'saudade', 'abraco', 'sentimento', 'angustia', 'dor', 'amor', 'amado', 'amou', 'paixao', 'tristeza', 'alegria', 'sofrer', 'sofrendo', 'desespero', 'desespere', 'acalma', 'amargo', 'doce', 'gosto'],
    'vago': ['luz', 'forca', 'energia', 'universo', 'destino', 'misterio', 'vibracao', 'vento', 'paz', 'magia', 'aura', 'poder', 'mundo', 'cor', 'sol', 'floresta', 'flor', 'montanha', 'muralha', 'sombra', 'terremoto', 'mar', 'estrela'],
    'filosofico': ['mestre', 'senhor', 'sabedoria', 'caminho', 'verdade', 'divino', 'criador', 'alma', 'ser', 'razao', 'fe', 'crenca', 'moral', 'virtude', 'bondade', 'esperanca']
}

STOP_WORDS = {'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'e', 'ou', 'mas', 'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na', 'nos', 'nas', 'por', 'para', 'com', 'que', 'se', 'como', 'mais', 'me', 'te', 'lhe', 'seus', 'suas', 'seu', 'sua', 'meu', 'minha', 'eu', 'tu', 'ele', 'ela', 'nós', 'vós', 'eles', 'elas', 'isso', 'isto', 'aquilo', 'já', 'tão', 'só', 'pra', 'pro', 'ao', 'aos', 'nem', 'não', 'sim', 'foi', 'tem', 'quem'}

# ---------------------------------------------------------
# 2. FUNÇÕES UTILITÁRIAS (MOTOR LÉXICO)
# ---------------------------------------------------------
def normalize_text(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_top_words(text):
    words = normalize_text(text).split()
    words = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
    return Counter(words).most_common(6)

def calculate_lexical_scores(lyrics):
    clean_lyrics = normalize_text(lyrics)
    scores = {k: 0 for k in DICTIONARIES.keys()}
    total_matches = 0

    for category, keywords in DICTIONARIES.items():
        for keyword in keywords:
            matches = len(re.findall(rf'\b{keyword}\b', clean_lyrics))
            scores[category] += matches
            total_matches += matches

    percentages = {k: 0 for k in DICTIONARIES.keys()}
    if total_matches > 0:
        for cat in scores:
            percentages[cat] = (scores[cat] / total_matches) * 100

    return {'rawCounts': scores, 'totalMatches': total_matches, 'percentages': percentages}

# ---------------------------------------------------------
# 3. CONEXÃO COM A IA (GEMINI API)
# ---------------------------------------------------------
def analyze_with_ai(input_text, api_key):
    genai.configure(api_key=api_key)
    
    # Configuramos o modelo
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    system_prompt = """Você é um Assistente de NLP Analista Musical e Linguista.
    ENTRADA: Letra de música.
    DIRETRIZES:
    1. Polissemia e Ambiguidade.
    2. Papel da Metáfora e Metonímia.
    3. Contexto e Pragmática (Dêixis).
    4. Isotopia Semântica.
    5. Relações de Sentido.
    TAREFAS:
    - Retorne a letra oficial limpa.
    - Avalie se a referência a Deus é Direta ou Obscura. Se não mencionar explicitamente a Trindade Cristã, ative 'alerta_secular', classifique como filosófica/vaga e zere teocentrico.
    - Dê uma pontuação SEMÂNTICA (0 a 100 somando 5 categorias: teocentrico, antropocentrico, sentimental, vago, filosofico).
    - Gere um estudo linguístico rigoroso.
    RESPONDA ESTRITAMENTE EM JSON SEGUINDO O SCHEMA ABAIXO."""

    # Simulando o schema passando a estrutura no prompt para garantir o JSON
    prompt = f"{system_prompt}\n\nSCHEMA JSON EXIGIDO:\n" + """
    {
      "letra_oficial": "string",
      "estilo_musical": "string",
      "direto_ou_obscuro": "string",
      "alerta_secular": boolean,
      "explicacao_semantica": "string",
      "estudo_linguistico": {
        "polissemia": "string", "figuras_linguagem": "string", "pragmatica": "string", "isotopia": "string", "relacoes_sentido": "string"
      },
      "semantica_teocentrico": 0, "semantica_antropocentrico": 0, "semantica_sentimental": 0, "semantica_vago": 0, "semantica_filosofico": 0
    }
    """ + f"\n\nLetra para análise: {input_text}"

    # Força a resposta em JSON
    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    return json.loads(response.text)

# ---------------------------------------------------------
# 4. INTERFACE STREAMLIT (UI)
# ---------------------------------------------------------
st.title("🎵 Analisador de Letras e Músicas")
st.markdown("Contagem léxica e análise semântica de Inteligência Artificial para composições.")

# Pega a API Key do secrets do Streamlit, ou pede pro usuário digitar
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    API_KEY = st.sidebar.text_input("Insira sua Gemini API Key", type="password")
    st.sidebar.markdown("[Pegue sua chave grátis aqui](https://aistudio.google.com/app/apikey)")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Entrada de Dados")
    lyrics_input = st.text_area("Cole a letra da música aqui:", height=300)
    analyze_btn = st.button("Executar Análise Híbrida (IA + Léxico)", use_container_width=True, type="primary")

if analyze_btn:
    if not API_KEY:
        st.error("Por favor, configure sua Gemini API Key no painel lateral.")
    elif not lyrics_input.strip():
        st.warning("Cole uma letra antes de analisar.")
    else:
        with st.spinner("Processando Inteligência Linguística e Semântica..."):
            try:
                # 1. Busca Semântica
                ai_data = analyze_with_ai(lyrics_input, API_KEY)
                
                # Normaliza Pesos Semânticos
                total_sem = sum([ai_data.get('semantica_teocentrico', 0), ai_data.get('semantica_antropocentrico', 0), ai_data.get('semantica_sentimental', 0), ai_data.get('semantica_vago', 0), ai_data.get('semantica_filosofico', 0)]) or 1
                sem_scores = {
                    'teocentrico': (ai_data.get('semantica_teocentrico', 0) / total_sem) * 100,
                    'antropocentrico': (ai_data.get('semantica_antropocentrico', 0) / total_sem) * 100,
                    'sentimental': (ai_data.get('semantica_sentimental', 0) / total_sem) * 100,
                    'vago': (ai_data.get('semantica_vago', 0) / total_sem) * 100,
                    'filosofico': (ai_data.get('semantica_filosofico', 0) / total_sem) * 100,
                }

                # 2. Busca Léxica
                lex_data = calculate_lexical_scores(ai_data['letra_oficial'])
                
                # 3. Mesclagem (Final)
                lex_weight = 0.8 if lex_data['totalMatches'] > 0 else 0.0
                sem_weight = 0.2 if lex_data['totalMatches'] > 0 else 1.0

                final_scores = {}
                for cat in sem_scores.keys():
                    final_scores[cat] = round((lex_data['percentages'][cat] * lex_weight) + (sem_scores[cat] * sem_weight))
                
                # Ajuste para fechar 100
                diff = 100 - sum(final_scores.values())
                if diff != 0:
                    max_key = max(final_scores, key=final_scores.get)
                    final_scores[max_key] += diff

                top_cat = max(final_scores, key=final_scores.get)
                cat_names = {
                    'teocentrico': "Teocêntrica (Adoração Direta)", 'antropocentrico': "Antropocêntrica (Foco Humano)",
                    'sentimental': "Sentimentalista (Foco Emocional)", 'vago': "Vaga / Abstrata", 'filosofico': "Filosófica / Espiritualidade Genérica"
                }

                # --- RENDERIZAÇÃO DOS RESULTADOS ---
                with col2:
                    st.success(f"**Diagnóstico Final:** {cat_names[top_cat]}")
                    
                    if ai_data.get('alerta_secular') or lex_data['rawCounts']['teocentrico'] == 0:
                        st.warning("⚠️ **Alerta Secular:** A letra não faz menção explícita à Trindade. O motor a classificou como possivelmente popular/filosófica.")

                    st.info(f"**Parecer Geral da IA:**\n\n{ai_data.get('explicacao_semantica')}")

                    # Visualizador Matemático (Tabs)
                    st.subheader("Matemática da Análise")
                    tab1, tab2, tab3 = st.tabs(["Barras de Peso", "Pizza", "Radar"])
                    
                    df_scores = pd.DataFrame({
                        'Categoria': ['Teocêntrico', 'Antropocêntrico', 'Sentimental', 'Vago', 'Filosófico'],
                        'Léxico (80%)': [lex_data['percentages']['teocentrico'], lex_data['percentages']['antropocentrico'], lex_data['percentages']['sentimental'], lex_data['percentages']['vago'], lex_data['percentages']['filosofico']],
                        'Semântico (20%)': [sem_scores['teocentrico'], sem_scores['antropocentrico'], sem_scores['sentimental'], sem_scores['vago'], sem_scores['filosofico']],
                        'Final (100%)': [final_scores['teocentrico'], final_scores['antropocentrico'], final_scores['sentimental'], final_scores['vago'], final_scores['filosofico']]
                    })

                    with tab1:
                        st.dataframe(df_scores.style.format("{:.1f}%", subset=['Léxico (80%)', 'Semântico (20%)', 'Final (100%)']), use_container_width=True)
                    
                    with tab2:
                        fig_pie = px.pie(df_scores, values='Final (100%)', names='Categoria', hole=0.4, color='Categoria',
                                         color_discrete_map={'Teocêntrico':'#2563eb', 'Antropocêntrico':'#f59e0b', 'Sentimental':'#f43f5e', 'Vago':'#475569', 'Filosófico':'#0891b2'})
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with tab3:
                        fig_radar = px.line_polar(df_scores, r='Final (100%)', theta='Categoria', line_close=True, markers=True)
                        fig_radar.update_traces(fill='toself', line_color='#2563eb')
                        st.plotly_chart(fig_radar, use_container_width=True)

                # Estudo Linguístico Profundo
                st.divider()
                st.subheader("📖 Estudo Semântico Profundo")
                linguistica = ai_data.get('estudo_linguistico', {})
                colA, colB = st.columns(2)
                with colA:
                    with st.expander("Polissemia / Ambiguidade", expanded=True):
                        st.write(linguistica.get('polissemia', 'N/A'))
                    with st.expander("Metáfora / Metonímia", expanded=True):
                        st.write(linguistica.get('figuras_linguagem', 'N/A'))
                    with st.expander("Isotopia Semântica"):
                        st.write(linguistica.get('isotopia', 'N/A'))
                with colB:
                    with st.expander("Contexto e Pragmática (Dêixis)", expanded=True):
                        st.write(linguistica.get('pragmatica', 'N/A'))
                    with st.expander("Relações de Sentido"):
                        st.write(linguistica.get('relacoes_sentido', 'N/A'))

                # Palavras mais citadas
                top_w = get_top_words(ai_data['letra_oficial'])
                if top_w:
                    st.divider()
                    st.subheader("📊 Top Palavras Utilizadas")
                    df_words = pd.DataFrame(top_w, columns=['Palavra', 'Frequência'])
                    fig_bar = px.bar(df_words, x='Frequência', y='Palavra', orientation='h', color='Frequência', color_continuous_scale='Emrld')
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)

            except Exception as e:
                st.error(f"Erro na análise: {e}")
