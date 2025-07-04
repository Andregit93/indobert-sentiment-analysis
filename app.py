import streamlit as st
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googleapiclient.discovery import build
from wordcloud import WordCloud, STOPWORDS
import torch.nn.functional as F
import time

# Konfigurasi awal - HARUS DI AWAL
st.set_page_config(
    page_title="Sistem Analisis Sentimen IndoBERT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header dengan penjelasan model
st.title("🤖 Sistem Analisis Sentimen Korupsi")
st.markdown("""
**Sistem ini menggunakan model IndoBERT yang telah dilatih khusus untuk analisis sentimen korupsi dalam bahasa Indonesia.**  
Model ini mampu mengklasifikasikan komentar YouTube menjadi dua kategori: **Positif** 😊 atau **Negatif** 😠.
""")

with st.expander("📚 Tentang Model IndoBERT", expanded=False):
    st.markdown("""
    **Arsitektur Model:**  
    - Berbasis transformer BERT yang dioptimalkan untuk bahasa Indonesia  
    - Dilatih pada dataset komentar YouTube terkait kasus korupsi  
    - Menggunakan teknik fine-tuning dengan K-Fold Cross Validation  
    
    **Keunggulan:**  
    - Akurasi tinggi pada teks bahasa Indonesia informal  
    - Kemampuan memahami konteks khusus kasus korupsi  
    - Telah divalidasi pada ribuan komentar YouTube  
    
    **Cara Penggunaan:**  
    1. Pilih mode input (URL video atau Tema)  
    2. Masukkan URL/tema yang ingin dianalisis  
    3. Klik tombol "Mulai Analisis"  
    4. Hasil akan ditampilkan dalam berbagai visualisasi  
    """)

# Konfigurasi API YouTube
api_key = 'AIzaSyCXIWHrND6VqZae7iysaiqyVLHJURRnHpE'
youtube = build('youtube', 'v3', developerKey=api_key)

def load_model_from_huggingface():
    """
    Fungsi cache untuk memuat model dari Hugging Face.
    Mengembalikan tokenizer dan model.
    """
    repo_id = "andresevtian/indobert-sentiment-analysis"
    
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        repo_id,
        use_safetensors=True,
        trust_remote_code=False
    )
    model.eval()
    return tokenizer, model


def load_model():
    """
    Fungsi pemanggil dengan spinner dan error handler.
    """
    with st.spinner("🔄 Memuat model..."):
        try:
            tokenizer, model = load_model_from_huggingface()
            st.success("✅ Model berhasil dimuat.")
            return tokenizer, model
        except Exception as e:
            st.error(f"❌ Gagal memuat model: {e}")
            return None, None


# Fungsi ekstraksi ID video dari URL
def extract_video_id(url):
    patterns = [
        r"(?:https?:\/\/)?(?:www\.|m\.)?youtu\.be\/([\w\-]{11})",
        r"(?:https?:\/\/)?(?:www\.|m\.)?youtube\.com\/.*[?&]v=([\w\-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Fungsi mengambil komentar dari video ID (SEMUA KOMENTAR)
def get_all_comments_from_video(video_id):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet", 
            videoId=video_id, 
            maxResults=1000
        )
        response = request.execute()
    except Exception as e:
        return [], f"Error: {str(e)}"
    
    while response:
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            
        if "nextPageToken" in response:
            request = youtube.commentThreads().list(
                part="snippet", 
                videoId=video_id, 
                pageToken=response["nextPageToken"],
                maxResults=100
            )
            response = request.execute()
        else:
            break
    
    return comments, None

# Fungsi mencari video berdasarkan tema (SEMUA VIDEO)
def search_videos_by_theme(theme, max_videos=100):
    videos = []
    next_page_token = None
    
    while len(videos) < max_videos:
        request = youtube.search().list(
            part="id,snippet",
            q=theme,
            type="video",
            maxResults=min(50, max_videos - len(videos)),
            pageToken=next_page_token
        )
        response = request.execute()
        
        for item in response.get("items", []):
            if len(videos) >= max_videos:
                break
                
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            videos.append((video_id, title))
        
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    
    return videos

# Fungsi mengambil komentar berdasarkan tema (20 KOMENTAR PER VIDEO)
def get_comments_by_theme(theme, max_videos=10, max_comments_per_video=20):
    videos = search_videos_by_theme(theme, max_videos)
    all_comments = []
    video_info = []
    
    for video_id, title in videos:
        comments, error = get_comments_from_video_id(video_id, max_comments_per_video)
        if not error and comments:
            all_comments.extend(comments)
            video_info.append({
                "video_id": video_id,
                "title": title,
                "comment_count": len(comments)
            })
    
    return all_comments, video_info

# Fungsi mengambil komentar terbatas (untuk mode tema)
def get_comments_from_video_id(video_id, max_comments=20):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet", 
            videoId=video_id, 
            maxResults=min(100, max_comments))
        response = request.execute()
    except Exception as e:
        return [], f"Error: {str(e)}"
    
    comment_count = 0
    while response and comment_count < max_comments:
        for item in response.get("items", []):
            if comment_count >= max_comments:
                break
                
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            comment_count += 1
            
        if "nextPageToken" in response and comment_count < max_comments:
            request = youtube.commentThreads().list(
                part="snippet", 
                videoId=video_id, 
                pageToken=response["nextPageToken"],
                maxResults=min(100, max_comments - comment_count))
            response = request.execute()
        else:
            break
    
    return comments, None

# Preprocessing komentar
def preprocess_comments(comments):
    if not comments:
        return pd.DataFrame(columns=["text"])
    
    df = pd.DataFrame(comments, columns=["text"])
    df.dropna(inplace=True)
    df = df[df["text"].str.strip() != ""]
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].str.replace(r"http\S+", "", regex=True)
    df["text"] = df["text"].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)
    df["text"] = df["text"].str.strip()
    return df

# Analisis sentimen
def analyze_sentiment(df, tokenizer, model):
    if df.empty:
        return df, 0
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    texts = df["text"].tolist()
    sentiments = []
    confidences = []
    
    # Proses dalam batch
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = F.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=1)
        
        sentiments.extend(preds.cpu().numpy())
        confidences.extend(torch.max(probs, dim=1).values.cpu().numpy())
    
    df["sentiment"] = sentiments
    df["confidence"] = confidences
    avg_confidence = np.mean(confidences) if confidences else 0
    return df, avg_confidence

# Visualisasi Sentimen
def plot_sentiment_distribution(df):
    if df.empty:
        st.warning("Tidak ada data untuk visualisasi")
        return
        
    sentiment_labels = {0: "Negatif", 1: "Positif"}
    df["sentiment_label"] = df["sentiment"].map(sentiment_labels)
    sentiment_counts = df["sentiment_label"].value_counts()

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=["#FF4B4B", "#4CAF50"])
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points')
    
    plt.xlabel("Sentimen", fontsize=12)
    plt.ylabel("Jumlah Komentar", fontsize=12)
    plt.title("Distribusi Sentimen Komentar", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)

# WordCloud
def generate_wordcloud(df):
    if df.empty:
        st.warning("Tidak ada data untuk word cloud")
        return
        
    stopwords = set(STOPWORDS)
    indonesian_stopwords = {
        "yang", "yg", "dan", "di", "ke", "untuk", "dengan", "dari", "ini", "itu", 
        "nya", "saya", "kamu", "kita", "mereka", "dia", "aku", "kak", "nya",
        "lah", "pun", "sih", "ya", "ga", "gak", "tidak", "bukan", "jangan",
        "atau", "juga", "ada", "adalah", "akan", "saat", "pada", "kalau", "kalo"
    }
    stopwords.update(indonesian_stopwords)
    
    text = ' '.join(df["text"])
    
    if not text.strip():
        st.warning("Teks tidak cukup untuk word cloud")
        return
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        colormap='viridis',
        max_words=200
    ).generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud Komentar", fontsize=14)
    st.pyplot(plt)

# Ambil kata-kata paling umum
def get_top_words(df, sentiment="all", top_n=20):
    if df.empty:
        return []
    
    if sentiment != "all":
        df = df[df["sentiment"] == (1 if sentiment == "positive" else 0)]
    
    all_text = ' '.join(df["text"]).split()
    
    stopwords = {
        "yang", "yg", "dan", "di", "ke", "untuk", "dengan", "dari", "ini", "itu", 
        "nya", "saya", "kamu", "kita", "mereka", "dia", "aku", "kak", "nya",
        "lah", "pun", "sih", "ya", "ga", "gak", "tidak", "bukan", "jangan",
        "atau", "juga", "ada", "adalah", "akan", "saat", "pada", "kalau", "kalo"
    }
    filtered_words = [word for word in all_text if word not in stopwords]
    
    word_freq = Counter(filtered_words)
    return word_freq.most_common(top_n)

# Visualisasi kata umum
def plot_top_words(top_words, title):
    if not top_words:
        return
        
    words, freqs = zip(*top_words)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(freqs), y=list(words), palette="viridis")
    plt.title(title, fontsize=14)
    plt.xlabel("Frekuensi", fontsize=12)
    plt.ylabel("Kata", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    st.pyplot(plt)

# Main Application
def main():
    # Load model
    tokenizer, model = load_model()
    if tokenizer is None or model is None:
        return
    
    # Pilihan input
    st.subheader("🔍 Pilih Mode Input")
    input_type = st.radio("", 
                          ["URL Video YouTube", "Tema/Kata Kunci"],
                          index=0,
                          horizontal=True)
    
    # Input berdasarkan pilihan
    comments = []
    video_info = []
    
    if input_type == "URL Video YouTube":
        video_url = st.text_input("🔗 Masukkan URL Video YouTube:", placeholder="https://www.youtube.com/watch?v=...")
        
        if st.button("🚀 Mulai Analisis"):
            if not video_url:
                st.warning("Harap masukkan URL video YouTube")
                return
                
            video_id = extract_video_id(video_url)
            if not video_id:
                st.error("URL video tidak valid")
                return
                
            with st.spinner("🔄 Mengambil komentar dari YouTube..."):
                comments, error = get_all_comments_from_video(video_id)
                
                if error:
                    st.error(f"Error: {error}")
                    return
                    
                if not comments:
                    st.warning("Tidak ada komentar yang ditemukan")
                    return
                    
                # Dapatkan judul video
                try:
                    video_request = youtube.videos().list(part="snippet", id=video_id)
                    video_response = video_request.execute()
                    video_title = video_response['items'][0]['snippet']['title'] if video_response.get('items') else "Unknown Title"
                except:
                    video_title = "Unknown Title"
                
                video_info = [{
                    "video_id": video_id,
                    "title": video_title,
                    "comment_count": len(comments)
                }]
                
                st.success(f"✅ Berhasil mengambil {len(comments)} komentar dari video: {video_title}")
    
    else:  # Input berdasarkan tema
        theme = st.text_input("🔍 Masukkan tema/kata kunci:", placeholder="exp: korupsi pertamina.")
        max_videos = st.slider("Jumlah video yang akan dianalisis", 1, 100, 5)
        
        if st.button("🚀 Mulai Analisis"):
            if not theme:
                st.warning("Harap masukkan tema/kata kunci")
                return
                
            with st.spinner(f"🔍 Mencari video terkait '{theme}'..."):
                videos = search_videos_by_theme(theme, max_videos)
                
                if not videos:
                    st.warning("Tidak ditemukan video untuk tema tersebut")
                    return
                    
                st.info(f"📺 Ditemukan {len(videos)} video terkait tema '{theme}'")
                
                # Tampilkan daftar video
                st.subheader("Video yang Dianalisis:")
                for idx, (video_id, title) in enumerate(videos):
                    st.markdown(f"{idx+1}. [{title}](https://www.youtube.com/watch?v={video_id})")
            
            with st.spinner("🔄 Mengambil komentar dari video..."):
                comments, video_info = get_comments_by_theme(theme, max_videos, 20)
                
                if not comments:
                    st.warning("Tidak ada komentar yang ditemukan")
                    return
                    
                total_comments = len(comments)
                st.success(f"✅ Berhasil mengambil {total_comments} komentar dari {len(video_info)} video")
                
                # Tampilkan statistik per video
                st.subheader("Statistik Komentar per Video:")
                for info in video_info:
                    st.markdown(f"- **{info['title']}**: {info['comment_count']} komentar")
    
    # Proses analisis jika ada komentar
    if comments:
        # Preprocessing
        with st.spinner("🧹 Membersihkan dan memproses komentar..."):
            df = preprocess_comments(comments)
            st.info(f"Komentar setelah pembersihan: {len(df)}")
        
        # Analisis sentimen
        with st.spinner("🤖 Menganalisis sentimen komentar..."):
            analyzed_df, avg_confidence = analyze_sentiment(df, tokenizer, model)
            
            # Hitung distribusi sentimen
            sentiment_counts = analyzed_df["sentiment"].value_counts()
            positive_count = sentiment_counts.get(1, 0)
            negative_count = sentiment_counts.get(0, 0)
            total = len(analyzed_df)
            positive_perc = (positive_count / total) * 100 if total > 0 else 0
            negative_perc = (negative_count / total) * 100 if total > 0 else 0
            
            # Tampilkan metrik utama
            st.subheader("📊 Hasil Analisis Sentimen")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Komentar", total)
            col2.metric("Komentar Positif", f"{positive_count} ({positive_perc:.1f}%)")
            col3.metric("Komentar Negatif", f"{negative_count} ({negative_perc:.1f}%)")
            col4.metric("Rata-rata Confidence", f"{avg_confidence:.4f}")

        # Visualisasi
        tab1, tab2, tab3 = st.tabs(["Distribusi Sentimen", "Word Cloud", "Analisis Kata"])
        
        with tab1:
            st.subheader("Distribusi Sentimen")
            plot_sentiment_distribution(analyzed_df)
            
            # TIDAK MENAMPILKAN CONTOH KOMENTAR (sesuai permintaan)
        
        with tab2:
            st.subheader("Word Cloud Komentar")
            generate_wordcloud(analyzed_df)
        
        with tab3:
            st.subheader("Kata Paling Umum")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Semua Komentar**")
                top_words_all = get_top_words(analyzed_df, "all")
                plot_top_words(top_words_all, "20 Kata Paling Umum")
            
            with col2:
                st.markdown("**Komentar Positif**")
                top_words_positive = get_top_words(analyzed_df, "positive")
                plot_top_words(top_words_positive, "20 Kata Paling Umum (Positif)")
            
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("**Komentar Negatif**")
                top_words_negative = get_top_words(analyzed_df, "negative")
                plot_top_words(top_words_negative, "20 Kata Paling Umum (Negatif)")

if __name__ == "__main__":
    main()