from pathlib import Path
from typing import Any, Dict, List, Tuple
import streamlit as st
from image_processing import (
    ALLOWED_EXTENSIONS,
    NON_BANANA_REJECTION_MESSAGE,
    RIPENESS_CATEGORIES,
    analyze_pil_image,
    is_model_available,
    load_image_from_source,
)

st.set_page_config(
    page_title="CAVENDISH VISION — Deteksi Kematangan Pisang Cavendish",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_SESSION_STATE: Dict[str, Any] = {
    "current_page": "Beranda",
    "input_mode": "Galeri",
    "detection_triggered": False,
    "last_file_hash": None,
    "uploader_nonce": 0,
    "hero_banana_stage": "matang",
}

HERO_BANANA_PALETTES: Dict[str, Dict[str, Any]] = {
    "mentah": {
        "status_label": "Mentah • Kulit Hijau Segar",
        "badge_bg": "#E9F2EC",
        "badge_fg": "#2F6B45",
        "circle_fill": "#E8F3EC",
        "g1_stops": ("#387A43", "#589E49", "#468942"),
        "g2_stops": ("#295F36", "#468A41", "#63A64E"),
        "stem_fill": "#295C33",
        "spots_svg": "",
    },
    "setengah_matang": {
        "status_label": "Setengah Matang • Hijau Kekuningan",
        "badge_bg": "#F2F7E4",
        "badge_fg": "#5B7A1C",
        "circle_fill": "#F3F7E8",
        "g1_stops": ("#6CA644", "#D8D04E", "#EBC34B"),
        "g2_stops": ("#3E7A3E", "#9DC242", "#F0CC55"),
        "stem_fill": "#4A6E36",
        "spots_svg": "",
    },
    "matang": {
        "status_label": "Matang • Kuning Cerah Merata",
        "badge_bg": "#FDF6E2",
        "badge_fg": "#946610",
        "circle_fill": "#FDF7E6",
        "g1_stops": ("#FCE166", "#F4C95D", "#E2A234"),
        "g2_stops": ("#F5D253", "#EAB944", "#D49026"),
        "stem_fill": "#5E6B38",
        "spots_svg": "",
    },
    "terlalu_matang": {
        "status_label": "Terlalu Matang • Kuning Bercak Cokelat",
        "badge_bg": "#F6ECE4",
        "badge_fg": "#7A4B21",
        "circle_fill": "#F7EFE8",
        "g1_stops": ("#E3A93E", "#C4822C", "#834F22"),
        "g2_stops": ("#C98C32", "#9E6324", "#6B3F1B"),
        "stem_fill": "#4A3520",
        "spots_svg": (
            '<circle cx="132" cy="154" r="4.2" fill="#5A3619" opacity="0.82"/>'
            '<circle cx="155" cy="148" r="3.2" fill="#4E2E14" opacity="0.78"/>'
            '<circle cx="178" cy="144" r="5.0" fill="#5A3619" opacity="0.85"/>'
            '<circle cx="202" cy="130" r="3.8" fill="#4E2E14" opacity="0.80"/>'
            '<circle cx="222" cy="112" r="4.5" fill="#5A3619" opacity="0.82"/>'
            '<circle cx="240" cy="88" r="3.4" fill="#4E2E14" opacity="0.78"/>'
            '<circle cx="145" cy="128" r="3.6" fill="#5A3619" opacity="0.75"/>'
            '<circle cx="185" cy="116" r="4.0" fill="#4E2E14" opacity="0.78"/>'
            '<circle cx="212" cy="92" r="3.3" fill="#5A3619" opacity="0.75"/>'
            '<circle cx="110" cy="158" r="3.5" fill="#4E2E14" opacity="0.80"/>'
        ),
    },
}


def init_session_state() -> None:
    for key, default_val in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_val


def navigate_to(page_name: str) -> None:
    st.session_state.current_page = page_name
    st.session_state.detection_triggered = False


def reset_detection_session() -> None:
    st.session_state.detection_triggered = False
    st.session_state.last_file_hash = None
    st.session_state.uploader_nonce += 1


def inject_custom_css() -> None:
    css_path = Path(__file__).resolve().parent / "style.css"
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def render_header_and_navigation() -> None:
    current = st.session_state.current_page
    head_left, head_right = st.columns([1.25, 1.35], gap="medium", vertical_alignment="center")

    with head_left:
        st.markdown(
            """
            <div class="cv-brand-bar">
                <div class="cv-logo-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#F4C95D" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M4 19c4.5-1 8.5-4 11-8.5C17.5 6 18 3 18 3s2 1.5 2 5.5c0 5-4.5 10.5-11.5 12C6 21 4 19 4 19z"/>
                        <path d="M14.5 6.5C12 9 8.5 11.5 4 12.5"/>
                    </svg>
                </div>
                <p class="cv-brand-text">CAVENDISH VISION</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    nav_items: List[Tuple[str, str]] = [
        ("Beranda", "nav_beranda"),
        ("Deteksi Pisang", "nav_deteksi"),
        ("Tentang Aplikasi", "nav_tentang"),
    ]
    with head_right:
        nav_cols = st.columns(len(nav_items), gap="small")
        for col, (page_label, btn_key) in zip(nav_cols, nav_items):
            with col:
                if st.button(
                    page_label,
                    key=btn_key,
                    type="primary" if current == page_label else "secondary",
                    use_container_width=True,
                ):
                    navigate_to(page_label)
                    st.rerun()

    st.markdown('<div class="cv-nav-divider"></div>', unsafe_allow_html=True)


def get_hero_illustration_html(stage_key: str) -> str:
    pal = HERO_BANANA_PALETTES.get(stage_key, HERO_BANANA_PALETTES["matang"])
    g1_a, g1_b, g1_c = pal["g1_stops"]
    g2_a, g2_b, g2_c = pal["g2_stops"]

    return (
        f'<div class="cv-hero-right-card">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem; flex-wrap: wrap; gap: 0.35rem;">'
        f'<span style="font-size: 0.72rem; font-weight: 700; color: {pal["badge_fg"]}; background: {pal["badge_bg"]}; padding: 0.24rem 0.68rem; border-radius: 99px;">PISANG CAVENDISH</span>'
        f'<span style="font-size: 0.74rem; font-weight: 700; color: {pal["badge_fg"]};">{pal["status_label"]}</span>'
        f"</div>"
        f'<svg viewBox="0 0 340 175" width="100%" style="max-width: 285px; height: auto; display: block; margin: 0 auto;">'
        f"<defs>"
        f'<linearGradient id="bananaGrad1_{stage_key}" x1="0%" y1="0%" x2="100%" y2="100%">'
        f'<stop offset="0%" stop-color="{g1_a}" /><stop offset="50%" stop-color="{g1_b}" /><stop offset="100%" stop-color="{g1_c}" />'
        f"</linearGradient>"
        f'<linearGradient id="bananaGrad2_{stage_key}" x1="0%" y1="10%" x2="95%" y2="90%">'
        f'<stop offset="0%" stop-color="{g2_a}" /><stop offset="50%" stop-color="{g2_b}" /><stop offset="100%" stop-color="{g2_c}" />'
        f"</linearGradient>"
        f'<filter id="softShadow" x="-10%" y="-10%" width="120%" height="125%">'
        f'<feDropShadow dx="0" dy="6" stdDeviation="5" flood-color="#152C1E" flood-opacity="0.12"/>'
        f"</filter>"
        f"</defs>"
        f'<path d="M35 32 L35 16 L55 16" fill="none" stroke="#2F6B45" stroke-width="2.2" stroke-linecap="round" opacity="0.5"/>'
        f'<path d="M305 32 L305 16 L285 16" fill="none" stroke="#2F6B45" stroke-width="2.2" stroke-linecap="round" opacity="0.5"/>'
        f'<path d="M35 144 L35 160 L55 160" fill="none" stroke="#2F6B45" stroke-width="2.2" stroke-linecap="round" opacity="0.5"/>'
        f'<path d="M305 144 L305 160 L285 160" fill="none" stroke="#2F6B45" stroke-width="2.2" stroke-linecap="round" opacity="0.5"/>'
        f'<circle cx="175" cy="90" r="64" fill="{pal["circle_fill"]}"/>'
        f'<path d="M98 124 C132 134, 202 110, 232 44 C237 32, 246 34, 244 48 C234 114, 168 158, 95 144 C85 142, 88 121, 98 124 Z" fill="url(#bananaGrad2_{stage_key})" filter="url(#softShadow)"/>'
        f'<path d="M82 136 C130 152, 215 130, 252 54 C257 42, 268 46, 265 60 C252 134, 172 178, 82 158 C70 155, 72 132, 82 136 Z" fill="url(#bananaGrad1_{stage_key})" filter="url(#softShadow)"/>'
        f'<path d="M100 145 C144 154, 210 132, 248 66" fill="none" stroke="#FFFDF6" stroke-width="2.1" stroke-linecap="round" opacity="0.65"/>'
        f'{pal["spots_svg"]}'
        f'<path d="M236 36 C244 25, 259 29, 263 42 C257 50, 244 48, 236 36 Z" fill="{pal["stem_fill"]}"/>'
        f'<circle cx="79" cy="147" r="4" fill="#4A3520"/>'
        f'<circle cx="93" cy="133" r="3.5" fill="#4A3520"/>'
        f"</svg>"
        f'<div style="font-size: 0.72rem; color: #6C7D72; margin-top: 0.25rem;">'
        f"Klik tingkat kematangan di bawah untuk melihat perubahan warna pisang:"
        f"</div>"
        f"</div>"
    )


def render_beranda_page() -> None:
    hero_left, hero_right = st.columns([1.22, 1], gap="medium", vertical_alignment="center")

    with hero_left:
        st.markdown(
            """
            <div class="cv-hero-left">
                <div class="cv-hero-badge">
                    🍌 CAVENDISH RIPENESS SCANNER
                </div>
                <h1 class="cv-hero-heading">
                    Deteksi Kematangan <span>Pisang Cavendish</span> dengan Mudah
                </h1>
                <p class="cv-hero-description">
                    Pindai foto buah Pisang Cavendish melalui kamera atau galeri untuk mengetahui
                    tingkat kematangan buah secara otomatis, cepat, dan objektif berdasarkan warna kulitnya.
                </p>
                <div class="cv-hero-highlights">
                    <span class="cv-highlight-tag">📷 Kamera &amp; Galeri</span>
                    <span class="cv-highlight-tag">🍌 Khusus Varietas Cavendish</span>
                    <span class="cv-highlight-tag">⚡ Hasil Instan</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cta_col1, cta_col2 = st.columns([1.15, 1], gap="small")
        with cta_col1:
            if st.button("Mulai Deteksi →", key="hero_cta_deteksi", type="primary", use_container_width=True):
                navigate_to("Deteksi Pisang")
                st.rerun()
        with cta_col2:
            if st.button("Pelajari Panduan", key="hero_cta_tentang", use_container_width=True):
                navigate_to("Tentang Aplikasi")
                st.rerun()

    with hero_right:
        current_stage = st.session_state.hero_banana_stage
        st.markdown(get_hero_illustration_html(current_stage), unsafe_allow_html=True)

        stage_options: List[Tuple[str, str]] = [
            ("mentah", "Mentah"),
            ("setengah_matang", "Setengah"),
            ("matang", "Matang"),
            ("terlalu_matang", "Lewat"),
        ]
        s_cols = st.columns(4, gap="small")
        for col, (s_key, s_label) in zip(s_cols, stage_options):
            with col:
                if st.button(
                    s_label,
                    key=f"btn_hero_stage_{s_key}",
                    type="primary" if current_stage == s_key else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.hero_banana_stage = s_key
                    st.rerun()

    st.markdown(
        """
        <div class="cv-sec-bar">
            <h2 class="cv-sec-title">Empat Kategori Kematangan</h2>
            <span class="cv-sec-pill">Klasifikasi Warna Kulit</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cat_items = list(RIPENESS_CATEGORIES.values())
    row1_col1, row1_col2 = st.columns(2, gap="medium")
    row2_col1, row2_col2 = st.columns(2, gap="medium")
    cat_cols = [row1_col1, row1_col2, row2_col1, row2_col2]

    for idx, cat in enumerate(cat_items):
        with cat_cols[idx]:
            st.markdown(
                f"""
                <div class="cv-card" style="border-top: 3px solid {cat['color_hex']};">
                    <div class="cv-cat-top">
                        <span class="cv-cat-badge" style="background: {cat['bg_hex']}; color: {cat['color_hex']};">
                            {cat['stage']}
                        </span>
                        <span class="cv-cat-swatch" style="background: {cat['color_hex']};"></span>
                    </div>
                    <h3 class="cv-cat-title">{cat['label']}</h3>
                    <p class="cv-cat-desc">{cat['description']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _build_progress_bar_html(label: str, percentage: float, color_hex: str) -> str:
    clamped = min(100.0, max(0.0, percentage))
    return (
        '<div class="cv-progress-item">'
        '<div class="cv-progress-label-row">'
        f'<span class="cv-progress-name"><span class="cv-progress-dot" style="background:{color_hex};"></span>{label}</span>'
        f'<span class="cv-progress-pct">{percentage:.1f}%</span>'
        "</div>"
        '<div class="cv-progress-track">'
        f'<div class="cv-progress-fill" style="width:{clamped:.1f}%; background:{color_hex};"></div>'
        "</div>"
        "</div>"
    )


def _build_rejection_card_html(status_label: str, title: str, message: str) -> str:
    return (
        '<div class="cv-reject-card">'
        '<div class="cv-result-top-bar">'
        f'<span class="cv-result-status-label" style="color:#B45309;">{status_label}</span>'
        '<span class="cv-sec-pill" style="background:#FEF3C7; color:#B45309;">Ditolak</span>'
        "</div>"
        f'<h3 class="cv-reject-title">{title}</h3>'
        f'<p class="cv-reject-msg">{message}</p>'
        "</div>"
    )


def render_deteksi_page() -> None:
    st.markdown(
        """
        <div class="cv-sec-bar" style="margin-top: 0;">
            <h2 class="cv-sec-title">Deteksi Kematangan Pisang</h2>
            <span class="cv-sec-pill">Kamera &amp; Galeri</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1, 1.08], gap="medium")
    selected_file = None
    nonce = st.session_state.uploader_nonce

    with left_col:
        st.markdown(
            """
            <div class="cv-panel-head">
                <div class="cv-step-badge">01</div>
                <div>
                    <p class="cv-panel-title">Pilih Sumber Foto</p>
                    <p class="cv-panel-sub">Unggah foto dari Galeri atau ambil langsung dengan Kamera.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        mode_col1, mode_col2 = st.columns(2, gap="small")
        with mode_col1:
            if st.button(
                "📁 Galeri",
                key="btn_mode_galeri",
                type="primary" if st.session_state.input_mode == "Galeri" else "secondary",
                use_container_width=True,
            ):
                st.session_state.input_mode = "Galeri"
                st.session_state.detection_triggered = False
                st.rerun()

        with mode_col2:
            if st.button(
                "📷 Kamera",
                key="btn_mode_kamera",
                type="primary" if st.session_state.input_mode == "Kamera" else "secondary",
                use_container_width=True,
            ):
                st.session_state.input_mode = "Kamera"
                st.session_state.detection_triggered = False
                st.rerun()

        if st.session_state.input_mode == "Galeri":
            selected_file = st.file_uploader(
                "Pilih file gambar (JPG, JPEG, PNG)",
                type=list(ALLOWED_EXTENSIONS),
                key=f"uploader_galeri_{nonce}",
            )
        else:
            selected_file = st.camera_input(
                "Arahkan kamera ke buah Pisang Cavendish",
                key=f"input_kamera_{nonce}",
            )

    pil_image, image_meta = load_image_from_source(selected_file)
    current_hash = image_meta.get("file_hash") if (image_meta and "file_hash" in image_meta) else None
    if current_hash != st.session_state.last_file_hash:
        st.session_state.last_file_hash = current_hash
        st.session_state.detection_triggered = False

    with right_col:
        st.markdown(
            """
            <div class="cv-panel-head">
                <div class="cv-step-badge">02</div>
                <div>
                    <p class="cv-panel-title">Pratinjau &amp; Hasil Deteksi</p>
                    <p class="cv-panel-sub">Periksa foto yang dipilih lalu tekan tombol Mulai Deteksi.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if image_meta is not None and "error" in image_meta:
            st.markdown(
                f"""
                <div class="cv-notice-box" style="border-color: #EF4444; background: #FEF2F2;">
                    <div style="font-weight: 800; color: #991B1B; font-size: 0.98rem; margin-bottom: 0.2rem;">
                        File Tidak Didukung
                    </div>
                    <div style="font-size: 0.84rem; color: #7F1D1D;">
                        {image_meta['error']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif pil_image is not None and image_meta is not None:
            st.image(
                pil_image,
                caption=f"{image_meta['filename']} ({image_meta['resolution']})",
                use_container_width=True,
            )
        else:
            st.markdown(
                """
                <div class="cv-empty-preview">
                    <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">🍌</div>
                    <div style="font-weight: 700; color: #152C1E; font-size: 0.92rem; margin-bottom: 0.15rem;">
                        Belum Ada Foto Dipilih
                    </div>
                    <div style="font-size: 0.82rem; color: #6C7D72;">
                        Unggah foto dari Galeri atau ambil foto melalui Kamera di sebelah kiri.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("Mulai Deteksi", key="btn_run_detection", type="primary", use_container_width=True):
            st.session_state.detection_triggered = True

        if st.session_state.detection_triggered:
            if image_meta is not None and "error" in image_meta:
                return
            if pil_image is None:
                title_msg = "Model belum dilatih" if not is_model_available() else "Pilih Foto Terlebih Dahulu"
                body_msg = (
                    "Silakan pilih foto Pisang Cavendish terlebih dahulu untuk melihat pratinjau pemindaian."
                    if not is_model_available()
                    else "Unggah foto atau ambil foto dari kamera sebelum memulai deteksi."
                )
                st.markdown(
                    f"""
                    <div class="cv-notice-box">
                        <div style="font-weight: 800; font-size: 0.98rem; color: #152C1E; margin-bottom: 0.2rem;">
                            {title_msg}
                        </div>
                        <div style="font-size: 0.84rem; color: #47584D;">
                            {body_msg}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                return

            with st.spinner("Memindai tingkat kematangan pisang..."):
                analysis = analyze_pil_image(pil_image)

            if not analysis["segmentation_success"]:
                st.markdown(
                    _build_rejection_card_html(
                        "Identifikasi Objek",
                        "Objek Tidak Dapat Dikenali",
                        NON_BANANA_REJECTION_MESSAGE,
                    ),
                    unsafe_allow_html=True,
                )
            elif not analysis["model_trained"]:
                st.markdown(
                    (
                        '<div class="cv-reject-card">'
                        '<h3 class="cv-reject-title">Model Belum Dilatih</h3>'
                        '<p class="cv-reject-msg">Silakan jalankan pelatihan model terlebih dahulu agar sistem dapat melakukan klasifikasi.</p>'
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
            elif not analysis["is_banana"]:
                st.markdown(
                    _build_rejection_card_html(
                        "Hasil Identifikasi",
                        "Bukan Pisang",
                        NON_BANANA_REJECTION_MESSAGE,
                    ),
                    unsafe_allow_html=True,
                )
            else:
                pred_info = analysis["predicted_info"]
                peel_pct = analysis["peel_color_percentages"]
                g_pct = float(peel_pct["green"])
                y_pct = float(peel_pct["yellow"])
                b_pct = float(peel_pct["brown"])
                o_pct = float(peel_pct["other"])
                explanation = analysis.get("explanation_text") or pred_info.get("explanation", "")

                bars_html = (
                    _build_progress_bar_html("Hijau", g_pct, "#2F6B45")
                    + _build_progress_bar_html("Kuning", y_pct, "#EAB308")
                    + _build_progress_bar_html("Cokelat", b_pct, "#8C5A2B")
                )
                if o_pct > 0.0:
                    bars_html += _build_progress_bar_html("Warna Lainnya", o_pct, "#94A3B8")

                result_html = (
                    f'<div class="cv-result-card" style="border-top: 4px solid {pred_info["color_hex"]};">'
                    '<div class="cv-result-top-bar">'
                    '<span class="cv-result-status-label">Hasil Klasifikasi Kematangan</span>'
                    f'<span class="cv-sec-pill" style="background:{pred_info["bg_hex"]}; color:{pred_info["color_hex"]};">Teridentifikasi Pisang</span>'
                    "</div>"
                    f'<h3 class="cv-result-big-title" style="color:{pred_info["color_hex"]};">{pred_info["label"]}</h3>'
                    f'<div class="cv-result-explanation" style="border-left-color:{pred_info["color_hex"]};">{explanation}</div>'
                    '<div class="cv-progress-group">'
                    '<div class="cv-progress-heading">'
                    "<span>Komposisi Warna Kulit Pisang</span>"
                    '<span style="font-size:0.75rem; font-weight:600; color:#6C7D72;">Area Objek Tersegmentasi</span>'
                    "</div>"
                    f"{bars_html}"
                    "</div>"
                    "</div>"
                )
                st.markdown(result_html, unsafe_allow_html=True)

            if st.button("Deteksi Foto Lain", key="btn_reset_detection", use_container_width=True):
                reset_detection_session()
                st.rerun()


def render_tentang_page() -> None:
    st.markdown(
        """
        <div class="cv-sec-bar" style="margin-top: 0;">
            <h2 class="cv-sec-title">Tentang Cavendish Vision</h2>
            <span class="cv-sec-pill">Informasi &amp; Panduan</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2, gap="medium")

    with col_a:
        st.markdown(
            """
            <div class="cv-card">
                <div style="display:flex; align-items:center; gap:0.65rem; margin-bottom:0.5rem;">
                    <div class="cv-step-badge">🎯</div>
                    <h3 class="cv-cat-title" style="margin:0;">Tujuan Aplikasi</h3>
                </div>
                <p class="cv-cat-desc">
                    Membantu mengenali tingkat kematangan buah <strong>Pisang Cavendish</strong> secara cepat,
                    mudah, dan objektif hanya melalui foto buah dari kamera maupun galeri perangkat.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="cv-card">
                <div style="display:flex; align-items:center; gap:0.65rem; margin-bottom:0.5rem;">
                    <div class="cv-step-badge">🍌</div>
                    <h3 class="cv-cat-title" style="margin:0;">Karakteristik Pisang Cavendish</h3>
                </div>
                <p class="cv-cat-desc">
                    Memiliki perubahan warna kulit yang konsisten saat matang: dari <strong>hijau segar</strong> (Mentah),
                    <strong>hijau kekuningan</strong> (Setengah Matang), <strong>kuning cerah</strong> (Matang),
                    hingga <strong>bercak cokelat</strong> (Terlalu Matang).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="cv-sec-bar">
            <h2 class="cv-sec-title">Petunjuk Penggunaan</h2>
            <span class="cv-sec-pill">4 Langkah</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    guides: List[Tuple[str, str, str]] = [
        ("01", "Buka Menu Deteksi", "Pilih menu Deteksi Pisang di bagian kanan atas."),
        ("02", "Masukkan Foto", "Gunakan Galeri untuk unggah file atau Kamera untuk memotret."),
        ("03", "Mulai Deteksi", "Tekan tombol Mulai Deteksi untuk memindai warna kulit pisang."),
        ("04", "Lihat Hasil", "Periksa kategori kematangan buah atau pindai foto pisang lainnya."),
    ]

    g_cols = st.columns(4, gap="small")
    for col, (g_num, g_title, g_desc) in zip(g_cols, guides):
        with col:
            st.markdown(
                f"""
                <div class="cv-card">
                    <div class="cv-step-badge" style="margin-bottom: 0.6rem;">{g_num}</div>
                    <h4 class="cv-cat-title" style="font-size: 0.94rem;">{g_title}</h4>
                    <p class="cv-cat-desc" style="font-size: 0.81rem;">{g_desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_footer() -> None:
    st.markdown(
        """
        <div class="cv-footer">
            <div><strong>CAVENDISH VISION</strong></div>
            <div>Deteksi Tingkat Kematangan Buah Pisang Cavendish</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_session_state()
    inject_custom_css()
    render_header_and_navigation()

    page_renderers = {
        "Beranda": render_beranda_page,
        "Deteksi Pisang": render_deteksi_page,
        "Tentang Aplikasi": render_tentang_page,
    }
    renderer = page_renderers.get(st.session_state.current_page, render_beranda_page)
    renderer()
    render_footer()


if __name__ == "__main__":
    main()
