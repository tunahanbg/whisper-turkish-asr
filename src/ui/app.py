"""
Streamlit web application for ASR system.
DRY: Modüler UI, tüm backend modüllerini kullanır.
"""

import streamlit as st
from pathlib import Path
import numpy as np
import time
from datetime import datetime
from loguru import logger

from config import config
from src.models import ModelManager
from src.preprocessing import AudioPreprocessor, VoiceActivityDetector
from src.audio import AudioRecorder, AudioFileHandler
from src.utils.audio_utils import format_timestamp, get_audio_duration


# Sayfa yapılandırması
st.set_page_config(
    page_title=config.get('ui.title', 'Konuşma Tanıma Sistemi'),
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Session state'i başlat."""
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = None
    
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = AudioPreprocessor()
    
    if 'vad' not in st.session_state:
        st.session_state.vad = VoiceActivityDetector()
    
    if 'file_handler' not in st.session_state:
        st.session_state.file_handler = AudioFileHandler()
    
    if 'transcription_history' not in st.session_state:
        st.session_state.transcription_history = []
    
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    
    if 'recorder' not in st.session_state:
        st.session_state.recorder = None


def load_model():
    """Model'i yükle (lazy loading)."""
    if st.session_state.model_manager is None:
        with st.spinner("Model yükleniyor... ⏳"):
            try:
                st.session_state.model_manager = ModelManager()
                st.session_state.model_manager.load_model()
                st.success("✅ Model başarıyla yüklendi!")
                logger.info("Model loaded via UI")
            except Exception as e:
                st.error(f"❌ Model yükleme hatası: {e}")
                logger.error(f"Model loading failed: {e}")
                raise


def sidebar_settings():
    """Sidebar ayarlar paneli."""
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        
        # Model ayarları
        st.subheader("🤖 Model")
        model_variant = st.selectbox(
            "Model Varyantı",
            options=["tiny", "base", "small", "medium", "large"],
            index=3,  # medium
            help="Büyük modeller daha doğru ama yavaştır"
        )
        
        if model_variant != config.get('model.variant'):
            if st.button("Model'i Güncelle"):
                config.set('model.variant', model_variant)
                if st.session_state.model_manager:
                    st.session_state.model_manager = None
                st.rerun()
        
        st.divider()
        
        # Dil ayarları
        st.subheader("🌍 Dil")
        language = st.radio(
            "Dil Seçimi",
            options=["Otomatik", "Türkçe", "İngilizce"],
            index=0,
            help="Otomatik: Model dili otomatik algılar"
        )
        
        language_map = {
            "Otomatik": None,
            "Türkçe": "tr",
            "İngilizce": "en"
        }
        selected_language = language_map[language]
        
        st.divider()
        
        # Ön işleme ayarları
        st.subheader("🔧 Ön İşleme")
        
        preprocessing_enabled = st.checkbox(
            "Ön işleme etkin",
            value=config.get('audio.preprocessing.enabled', True)
        )
        
        if preprocessing_enabled:
            normalize = st.checkbox(
                "Normalize",
                value=config.get('audio.preprocessing.normalize', True),
                help="Ses seviyesini normalize et"
            )
            
            trim_silence = st.checkbox(
                "Sessizlik kırpma",
                value=config.get('audio.preprocessing.trim_silence', True),
                help="Başta ve sonda sessizlik kırp"
            )
            
            denoise = st.checkbox(
                "Gürültü azaltma",
                value=config.get('audio.preprocessing.denoise', False),
                help="Arka plan gürültüsünü azalt (yavaşlatabilir)"
            )
        
        st.divider()
        
        # VAD ayarları
        st.subheader("🎙️ VAD (Sessizlik Algılama)")
        
        vad_enabled = st.checkbox(
            "VAD etkin",
            value=config.get('vad.enabled', True),
            help="Otomatik sessizlik algılama ve kayıt durdurma"
        )
        
        if vad_enabled:
            silence_duration = st.slider(
                "Sessizlik süresi (saniye)",
                min_value=2,
                max_value=30,
                value=config.get('vad.min_silence_duration_ms', 10000) // 1000,
                help="Bu kadar sessizlik sonrası kayıt otomatik durur"
            )
        
        st.divider()
        
        # İstatistikler
        st.subheader("📊 İstatistikler")
        st.metric("Transkripsiyon Sayısı", len(st.session_state.transcription_history))
        
        if st.session_state.model_manager and st.session_state.model_manager.is_loaded:
            st.success("🟢 Model Aktif")
        else:
            st.warning("🟡 Model Yüklenmedi")
        
        return selected_language


def transcribe_audio(audio: np.ndarray, language: str = None) -> dict:
    """
    Ses verisini transkribe et.
    
    Args:
        audio: Audio numpy array
        language: Dil kodu
    
    Returns:
        Transcription result
    """
    # Model'i yükle
    load_model()
    
    # Transkripsiyon - Preprocessing DEVRE DIŞI (Whisper kendi preprocessing'ini yapıyor)
    with st.spinner("Transkripsiyon yapılıyor... ✍️"):
        start_time = time.time()
        
        # Audio'yu Whisper'ın beklediği formata çevir
        # Sounddevice float32 veriyor ama [-1, 1] aralığında olmayabilir
        import numpy as np
        
        # DEBUG: Gelen audio'yu kontrol et
        logger.info(f"Transcription input - dtype: {audio.dtype}, shape: {audio.shape}, "
                   f"range: [{audio.min():.4f}, {audio.max():.4f}]")
        
        # Float32'ye çevir ve normalize et
        audio_normalized = audio.astype(np.float32)
        max_amplitude = np.abs(audio_normalized).max()
        
        if max_amplitude < 0.001:
            logger.error("Audio is nearly SILENT! Cannot transcribe.")
            st.error("❌ Ses çok sessiz veya bozuk! Mikrofon ayarlarınızı kontrol edin.")
            return None
        
        if max_amplitude > 1.0:
            logger.debug(f"Normalizing audio - max amplitude: {max_amplitude:.4f}")
            audio_normalized = audio_normalized / max_amplitude
        
        logger.debug(f"Audio after normalization - range: [{audio_normalized.min():.4f}, "
                    f"{audio_normalized.max():.4f}]")
        
        model = st.session_state.model_manager.get_model()
        result = model.transcribe(audio_normalized, language=language)
        
        elapsed_time = time.time() - start_time
        audio_duration = get_audio_duration(audio, config.get('audio.sample_rate'))
        rtf = elapsed_time / audio_duration if audio_duration > 0 else 0
    
    # Sonucu formatla
    formatted_result = model.format_output(result, include_timestamps=True, include_segments=True)
    formatted_result['processing_time'] = elapsed_time
    formatted_result['audio_duration'] = audio_duration
    formatted_result['rtf'] = rtf
    formatted_result['timestamp'] = datetime.now().isoformat()
    
    # Geçmişe ekle
    st.session_state.transcription_history.append(formatted_result)
    
    return formatted_result


def display_transcription_result(result: dict):
    """Transkripsiyon sonucunu göster."""
    st.success("✅ Transkripsiyon Tamamlandı!")
    
    # Ana metin - BÜYÜK VE OKUNAKLI
    st.markdown("### 📝 Transkripsiyon")
    st.markdown(f"""
    <div style='background-color: #1E1E1E; padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50;'>
        <p style='font-size: 18px; line-height: 1.6; color: #FFFFFF; margin: 0;'>
            {result['text']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("") # Spacer
    
    # Bilgiler
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌍 Dil", result['language'].upper())
    
    with col2:
        st.metric("⏱️ İşlem Süresi", f"{result['processing_time']:.2f}s")
    
    with col3:
        st.metric("🎵 Ses Süresi", f"{result['audio_duration']:.2f}s")
    
    with col4:
        st.metric("⚡ RTF", f"{result['rtf']:.2f}x")
    
    # Segmentler (varsa)
    if 'segments' in result and result['segments']:
        with st.expander("📊 Detaylı Segmentler"):
            for i, seg in enumerate(result['segments'], 1):
                start_time = format_timestamp(seg['start'])
                end_time = format_timestamp(seg['end'])
                st.markdown(f"**[{start_time} → {end_time}]** {seg['text']}")
    
    # İndirme butonu
    text_content = result['text']
    st.download_button(
        label="💾 Metni İndir",
        data=text_content,
        file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )


def file_upload_tab(language: str):
    """Dosya yükleme sekmesi."""
    st.header("📁 Ses Dosyası Yükle")
    
    # Desteklenen formatlar
    supported_formats = config.get('ui.supported_formats', ['wav', 'mp3', 'm4a', 'ogg'])
    max_size_mb = config.get('ui.max_upload_size_mb', 200)
    
    st.info(f"📌 Desteklenen formatlar: {', '.join(supported_formats)} | "
           f"Maksimum boyut: {max_size_mb}MB")
    
    uploaded_file = st.file_uploader(
        "Ses dosyası seçin",
        type=supported_formats,
        help=f"Maksimum {max_size_mb}MB"
    )
    
    if uploaded_file is not None:
        # Dosya bilgisi
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.write(f"📄 **Dosya:** {uploaded_file.name} ({file_size_mb:.2f}MB)")
        
        # Boyut kontrolü
        if file_size_mb > max_size_mb:
            st.error(f"❌ Dosya çok büyük! Maksimum {max_size_mb}MB olmalı.")
            return
        
        # Transkribe et butonu
        if st.button("🚀 Transkribe Et", type="primary", use_container_width=True):
            try:
                # Geçici dosyaya kaydet
                temp_path = Path(f"./data/cache/{uploaded_file.name}")
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Audio yükle
                audio, sr = st.session_state.file_handler.load(temp_path)
                
                # Audio player
                st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
                
                # Transkribe et
                result = transcribe_audio(audio, language)
                
                # Sonucu göster
                display_transcription_result(result)
                
                # Geçici dosyayı sil
                temp_path.unlink()
                
            except Exception as e:
                st.error(f"❌ Hata oluştu: {e}")
                logger.error(f"Transcription failed: {e}")


def microphone_tab(language: str):
    """Mikrofon kaydı sekmesi."""
    st.header("🎤 Mikrofon ile Kaydet")
    
    st.info("📌 Kaydı başlat butonuna tıklayın. VAD etkinse, sessizlik algılandığında kayıt otomatik durur.")
    
    # Mikrofon Gain Ayarı
    st.markdown("#### 🎚️ Mikrofon Seviyesi")
    input_gain = st.slider(
        "Mikrofon Gain (Ses çok düşükse artırın)",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Mikrofonunuzdan gelen ses çok düşükse bu değeri artırın. Önerilen: 3.0-5.0"
    )
    
    # Kayıt kontrolleri
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔴 Kaydı Başlat", disabled=st.session_state.is_recording, 
                    type="primary", use_container_width=True):
            try:
                st.session_state.recorder = AudioRecorder(
                    vad=st.session_state.vad,
                    input_gain=input_gain
                )
                st.session_state.recorder.start_recording()
                st.session_state.is_recording = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ Kayıt başlatılamadı: {e}")
    
    with col2:
        if st.button("⏹️ Kaydı Durdur", disabled=not st.session_state.is_recording,
                    use_container_width=True):
            if st.session_state.recorder:
                audio = st.session_state.recorder.stop_recording()
                st.session_state.is_recording = False
                
                if len(audio) > 0:
                    # Transkribe et
                    result = transcribe_audio(audio, language)
                    
                    # Result None olabilir (sessiz audio)
                    if result:
                        display_transcription_result(result)
                else:
                    st.warning("⚠️ Ses kaydedilmedi!")
                
                st.session_state.recorder = None
    
    # Kayıt durumu - GÖRÜNÜR
    if st.session_state.is_recording:
        # Büyük uyarı kutusu
        st.markdown("""
        <div style='background-color: #FF4444; padding: 15px; border-radius: 10px; text-align: center;'>
            <h2 style='color: white; margin: 0;'>🔴 KAYIT DEVAM EDİYOR</h2>
            <p style='color: white; margin: 5px 0 0 0;'>10 saniye sessizlik sonrası otomatik duracak</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Süreyi göster
        if st.session_state.recorder:
            duration = st.session_state.recorder.get_recording_duration()
            st.metric("⏱️ Kayıt Süresi", f"{duration:.1f}s")
            st.info("💡 Süreyi güncellemek için 'Kaydı Durdur' butonuna tıklayın.")
    
    # Not: Gerçek zamanlı mikrofon kaydı için streamlit-webrtc kullanılabilir
    st.info("💡 **Not:** Mikrofon erişimi için browser izinleri gerekebilir.")


def history_tab():
    """Transkripsiyon geçmişi sekmesi."""
    st.header("📜 Transkripsiyon Geçmişi")
    
    if not st.session_state.transcription_history:
        st.info("Henüz transkripsiyon yapılmadı.")
        return
    
    # Temizle butonu
    if st.button("🗑️ Geçmişi Temizle"):
        st.session_state.transcription_history = []
        st.rerun()
    
    st.divider()
    
    # Geçmişi ters sırada göster (en yeni en üstte)
    for i, result in enumerate(reversed(st.session_state.transcription_history), 1):
        with st.expander(f"#{i} - {result['language'].upper()} - "
                        f"{datetime.fromisoformat(result['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}"):
            
            st.markdown(f"**Metin:** {result['text']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dil", result['language'].upper())
            with col2:
                st.metric("İşlem Süresi", f"{result['processing_time']:.2f}s")
            with col3:
                st.metric("Ses Süresi", f"{result['audio_duration']:.2f}s")


def main():
    """Ana uygulama fonksiyonu."""
    # Session state'i başlat
    initialize_session_state()
    
    # Başlık
    st.title(config.get('ui.title', '🎤 Konuşma Tanıma Sistemi'))
    st.markdown("**Türkçe ve İngilizce destekli yerel konuşma tanıma sistemi**")
    st.markdown("---")
    
    # Sidebar ayarlar
    selected_language = sidebar_settings()
    
    # Ana sekmeler
    tab1, tab2, tab3 = st.tabs(["📁 Dosya Yükle", "🎤 Mikrofon", "📜 Geçmiş"])
    
    with tab1:
        file_upload_tab(selected_language)
    
    with tab2:
        microphone_tab(selected_language)
    
    with tab3:
        history_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Gazi Üniversitesi - Bilgisayar Mühendisliği</p>
            <p>Tunahan Başaran Güneysu</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

