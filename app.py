import streamlit as st
import pandas as pd
import io
import os
import tempfile
from datetime import datetime

# Import our detection classes
from pii_detector import PIIDetector
from text_pii_detector import TextPIIDetector

# Page configuration
st.set_page_config(
    page_title="Privacy Sandbox",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Remove smooth transitions with custom CSS for faster feel
st.markdown("""
<style>
* {
    transition: none !important;
    animation: none !important;
}
.stApp > div {
    animation: none !important;
    transition: none !important;
}
.stButton > button {
    transition: none !important;
    animation-duration: 0s !important;
}
.css-1d391kg {
    transition: none !important;
    padding-top: 1rem;
}
.main .block-container {
    transition: none !important;
    animation: none !important;
    padding-top: 2rem;
    padding-bottom: 1rem;
}
.stDataFrame {
    transition: none !important;
}
.stSpinner {
    animation-duration: 0.5s !important;
}
</style>
""", unsafe_allow_html=True)


class CompletePrivacySandboxApp:
    """Complete Streamlit application with CSV, Text, and Image file support"""

    def __init__(self):
        # Lazy-initialized detectors (avoids Streamlit caching errors with 'self')
        self.csv_detector = None
        self.text_detector = None

    def get_csv_detector(self):
        """Get or create CSV detector"""
        if self.csv_detector is None:
            self.csv_detector = PIIDetector()
        return self.csv_detector

    def get_text_detector(self):
        """Get or create text detector"""
        if self.text_detector is None:
            self.text_detector = TextPIIDetector()
        return self.text_detector

    def apply_confidence_setting(self):
        """Apply sidebar confidence threshold to all detectors"""
        confidence = st.session_state.get('confidence_threshold', 0.8)
        self.get_csv_detector().confidence_threshold = confidence
        self.get_text_detector().confidence_threshold = confidence

    def main(self):
        """Main application interface"""
        st.title("🔒 Privacy Sandbox for AI/ML Datasets")
        st.markdown("**Automatically detect and redact sensitive personal information (PII) from your datasets**")

        self.setup_sidebar()

        current_page = st.session_state.get('page', 'home')

        if current_page == 'home':
            self.show_home_page()
        elif current_page == 'upload':
            self.show_upload_page()
        elif current_page == 'about':
            self.show_about_page()

    def setup_sidebar(self):
        """Setup sidebar navigation and settings"""
        st.sidebar.title("🧭 Navigation")

        # Only initialize if not already set
        if 'page' not in st.session_state:
            st.session_state.page = 'home'
        
        # Get current page AFTER initialization check
        current_page = st.session_state.page

        col1, col2, col3 = st.sidebar.columns(3)

        with col1:
            if st.button("🏠", use_container_width=True,
                         type="primary" if current_page == 'home' else "secondary"):
                st.session_state.page = 'home'
                st.rerun()

        with col2:
            if st.button("📤", use_container_width=True,
                         type="primary" if current_page == 'upload' else "secondary"):
                st.session_state.page = 'upload'
                st.rerun()

        with col3:
            if st.button("ℹ️", use_container_width=True,
                         type="primary" if current_page == 'about' else "secondary"):
                st.session_state.page = 'about'
                st.rerun()

        st.sidebar.caption("Home | Upload | About")
        st.sidebar.markdown("---")

        if current_page == 'upload':
            # The confidence slider only affects CSV & text scanning.
            # Hide it when the user is on Images to avoid confusion.
            file_type = st.session_state.get('file_type_choice')

            if file_type != "🖼️ Images":
                st.sidebar.title("⚙️ Settings")

                confidence_threshold = st.sidebar.slider(
                    "Detection Confidence",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.8,
                    step=0.1,
                    help="Lower = more sensitive (applies to CSV & text scanning)"
                )
                st.session_state.confidence_threshold = confidence_threshold
            elif 'confidence_threshold' not in st.session_state:
                st.session_state.confidence_threshold = 0.8
        else:
            if 'confidence_threshold' not in st.session_state:
                st.session_state.confidence_threshold = 0.8

    def show_home_page(self):
        """Home page with overview and quick start"""
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ## 🎯 Privacy Sandbox - AI/ML Data Protection
            
            **Automatically detect and redact sensitive PII:**
            📧 Emails • 📱 Phones • 👤 Names • 💳 Credit Cards • 🆔 SSNs • 🏠 Addresses
            
            **Benefits:**
            ✅ **Legal Compliance** - GDPR, FERPA, HIPAA ready
            ✅ **Safe Collaboration** - Share data without privacy risks  
            ✅ **ML Ready** - Clean datasets for training
            ✅ **Time Saving** - Automated vs manual review
            
            **File Support:**
            📊 **CSV** (.csv) | 📝 **Text** (.txt, .json, .md) | 🖼️ **Images** (.jpg, .png, .bmp)
            """)

        with col2:
            st.markdown("### 🚀 Get Started")

            st.info("""
            **Quick Start:**
            
            1️⃣ Click 📤 Upload
            2️⃣ Choose file type  
            3️⃣ Upload & scan
            4️⃣ Download clean files
            """)

            if st.button("🚀 Start Now", use_container_width=True, type="primary"):
                st.session_state.page = 'upload'
                st.rerun()

        st.markdown("---")
        st.markdown("### 📝 Sample Data Available")
        st.info("💡 **Tip:** Upload `customers.csv` or `employees.csv` from your `data/input/` folder to test!")

    def show_upload_page(self):
        """File upload and processing page"""
        st.header("📤 Upload & Clean Your Files")

        file_type = st.radio(
            "Choose file type to process:",
            ["📊 CSV/Tabular Data", "📝 Text Documents", "🖼️ Images"],
            horizontal=True,
            key="file_type_choice"
        )

        if file_type == "📊 CSV/Tabular Data":
            self.show_csv_upload()
        elif file_type == "📝 Text Documents":
            self.show_text_upload()
        else:
            self.show_image_upload()

    def show_csv_upload(self):
        """CSV file upload and processing"""
        st.subheader("📊 CSV File Processing")

        uploaded_file = st.file_uploader(
            "Choose a CSV file to clean",
            type=['csv'],
            help="Upload a CSV file and we'll automatically detect and redact PII",
            key="csv_uploader"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("File Size", f"{uploaded_file.size} bytes")

                st.markdown("#### 📋 Original Data Preview")
                st.dataframe(df.head(), use_container_width=True)

                redaction_method = st.selectbox(
                    "Redaction Method:",
                    ["Smart Redaction", "Complete Masking", "Partial Redaction"],
                    help="Smart Redaction preserves IDs and applies targeted fixes"
                )

                if st.button("🔍 Scan CSV for PII", type="primary", use_container_width=True):
                    with st.spinner("Scanning CSV for sensitive information..."):
                        self.apply_confidence_setting()
                        if redaction_method == "Smart Redaction":
                            df_cleaned, pii_results = self.get_csv_detector().smart_detect_and_redact(df)
                        else:
                            detector = self.get_csv_detector()
                            pii_results = detector.detect_pii_in_dataframe(df)
                            method = 'mask' if redaction_method == "Complete Masking" else 'partial'
                            df_cleaned = detector.redact_dataframe(df, pii_results, method=method)

                        st.session_state.csv_original = df
                        st.session_state.csv_cleaned = df_cleaned
                        st.session_state.csv_pii_results = pii_results
                        st.session_state.csv_filename = uploaded_file.name

                if 'csv_cleaned' in st.session_state:
                    self.show_csv_results()

            except Exception as e:
                st.error(f"Error loading CSV file: {str(e)}")
                st.info("Please make sure your file is a valid CSV format.")

    def show_text_upload(self):
        """Text file upload and processing"""
        st.subheader("📝 Text File Processing")

        uploaded_files = st.file_uploader(
            "Choose text files to clean",
            type=['txt', 'json', 'md'],
            accept_multiple_files=True,
            help="Upload text files and we'll detect PII using advanced NLP",
            key="text_uploader"
        )

        if uploaded_files:
            st.success(f"✅ Uploaded {len(uploaded_files)} text file(s)")

            col1, col2 = st.columns(2)

            with col1:
                redaction_strategies = st.multiselect(
                    "Redaction Strategies:",
                    ["mask", "pseudonymize", "partial_mask", "label"],
                    default=["mask", "pseudonymize"],
                    help="Choose multiple strategies to compare results"
                )

            with col2:
                st.metric(
                    "Detection Confidence",
                    st.session_state.get('confidence_threshold', 0.8),
                    help="Adjust using the sidebar slider"
                )

            if st.button("🔍 Scan Text Files for PII", type="primary", use_container_width=True):
                if not redaction_strategies:
                    st.error("Please select at least one redaction strategy!")
                else:
                    with st.spinner("Processing text files with advanced NLP..."):
                        self.apply_confidence_setting()
                        results_all = []
                        text_detector = self.get_text_detector()

                        for uploaded_file in uploaded_files:
                            temp_path = f"/tmp/{uploaded_file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            analysis = text_detector.process_text_file(
                                temp_path,
                                redaction_strategies=redaction_strategies
                            )

                            if analysis:
                                results_all.append((uploaded_file.name, analysis))

                            os.remove(temp_path)

                        st.session_state.text_results = results_all

            if 'text_results' in st.session_state:
                self.show_text_results()

    def show_image_upload(self):
        """Image file upload and processing"""
        st.subheader("🖼️ Image PII Detection")

        try:
            from image_detector import ImagePIIDetector
        except ImportError:
            st.error("❌ Image processing not available. Make sure you have installed: opencv-python, pytesseract")
            st.info("Run: `pip install opencv-python pytesseract` and `brew install tesseract`")
            return

        uploaded_files = st.file_uploader(
            "Choose image files to scan for PII",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Upload images and we'll detect faces and text PII",
            key="image_uploader"
        )

        if uploaded_files:
            st.success(f"✅ Uploaded {len(uploaded_files)} image(s)")

            st.markdown("#### 🔧 Processing Options")
            col1, col2 = st.columns(2)

            with col1:
                detect_faces = st.checkbox("👤 Detect Faces", value=True)
                detect_text = st.checkbox("📝 Detect Text PII", value=True)

                if not detect_faces and not detect_text:
                    st.warning("⚠️ Select at least one detection method!")

            with col2:
                face_method = st.selectbox(
                    "Face Redaction:",
                    ["Blur", "Pixelate", "Black Box"],
                    help="How to hide detected faces"
                )

                text_method = st.selectbox(
                    "Text Redaction:",
                    ["Blur", "Pixelate", "Black Box"],
                    help="How to hide detected text PII"
                )

            if st.button("🔍 Scan Images for PII", type="primary", use_container_width=True):
                if not detect_faces and not detect_text:
                    st.error("Please select at least one detection method!")
                else:
                    with st.spinner("Processing images for PII detection..."):
                        detector = ImagePIIDetector()
                        results_all = []

                        for uploaded_file in uploaded_files:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(uploaded_file.getbuffer())
                                temp_path = tmp_file.name

                            methods = []
                            if detect_faces:
                                if face_method == "Blur":
                                    methods.append('blur_faces')
                                elif face_method == "Pixelate":
                                    methods.append('pixelate_faces')
                                else:
                                    methods.append('black_box_faces')

                            if detect_text:
                                if text_method == "Blur":
                                    methods.append('blur_text')
                                elif text_method == "Pixelate":
                                    methods.append('pixelate_text')
                                else:
                                    methods.append('redact_text')

                            try:
                                results = detector.process_image(temp_path, redaction_methods=methods)
                                if results:
                                    results_all.append((uploaded_file.name, results, temp_path))
                            except Exception as e:
                                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

                        if results_all:
                            st.session_state.image_results = results_all
                            st.success("✅ Image processing complete!")

            if 'image_results' in st.session_state:
                self.show_image_results()

    def show_csv_results(self):
        """Display CSV processing results"""
        st.markdown("---")
        st.subheader("🔍 CSV PII Detection Results")

        df_original = st.session_state.csv_original
        df_cleaned = st.session_state.csv_cleaned
        pii_results = st.session_state.csv_pii_results

        total_pii_items = sum(len(items) for items in pii_results.values())
        affected_columns = len(pii_results)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("PII Items Found", total_pii_items)
        with col2:
            st.metric("Affected Columns", affected_columns)
        with col3:
            st.metric("Clean Columns", len(df_original.columns) - affected_columns)

        if pii_results:
            st.warning(f"⚠️ Found PII in {affected_columns} columns")

            with st.expander("📊 Detailed Detection Results", expanded=False):
                for column, items in pii_results.items():
                    st.write(f"**Column: {column}** ({len(items)} items)")
                    for item in items[:3]:
                        if hasattr(item, 'get') and 'pii_detected' in item:
                            pii_types = [result.entity_type for result in item['pii_detected']]
                            st.write(f"- Row {item['row']}: {pii_types}")
        else:
            st.success("✅ No PII detected - your CSV data appears to be clean!")

        st.markdown("#### 🔄 Before & After Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Original Data**")
            st.dataframe(df_original.head(), use_container_width=True)

        with col2:
            st.write("**Cleaned Data**")
            st.dataframe(df_cleaned.head(), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 💾 Download Cleaned Data")

        col1, col2 = st.columns(2)

        with col1:
            csv_buffer = io.StringIO()
            df_cleaned.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Clean CSV",
                data=csv_buffer.getvalue(),
                file_name=f"clean_{st.session_state.csv_filename}",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            report = self.generate_csv_report(df_original, df_cleaned, pii_results)
            st.download_button(
                label="📋 Download Privacy Report",
                data=report,
                file_name=f"privacy_report_{st.session_state.csv_filename.replace('.csv', '.txt')}",
                mime="text/plain",
                use_container_width=True
            )

    def show_text_results(self):
        """Display text processing results"""
        st.markdown("---")
        st.subheader("📝 Text File PII Detection Results")

        results_all = st.session_state.text_results

        for filename, analysis in results_all:
            st.markdown(f"### 📄 {filename}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("File Size", f"{analysis['file_size']} chars")
            with col2:
                st.metric("Total PII Found", analysis['total_pii_found'])
            with col3:
                st.metric("PII Types", len(analysis['pii_by_type']))
            with col4:
                st.metric("Redacted Versions", len(analysis['redacted_versions']))

            if analysis['pii_by_type']:
                with st.expander(f"📊 PII Breakdown for {filename}", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**By Type:**")
                        for pii_type, count in analysis['pii_by_type'].items():
                            st.write(f"- {pii_type}: {count}")

                    with col2:
                        st.write("**By Detection Source:**")
                        for source, count in analysis['pii_by_source'].items():
                            st.write(f"- {source}: {count}")

            if analysis['redacted_versions']:
                st.markdown("**🔒 Redacted Versions:**")

                strategy_tabs = st.tabs(list(analysis['redacted_versions'].keys()))

                for i, (strategy, data) in enumerate(analysis['redacted_versions'].items()):
                    with strategy_tabs[i]:
                        st.markdown(f"**{strategy.replace('_', ' ').title()} Strategy:**")

                        text_preview = data['content'][:1000]
                        if len(data['content']) > 1000:
                            text_preview += "\n\n... (truncated for display)"

                        st.text_area(
                            "Redacted text preview:",
                            text_preview,
                            height=200,
                            key=f"text_preview_{filename}_{strategy}"
                        )

                        redactions = data['redaction_info']['redactions']
                        if redactions:
                            st.write(f"**Redactions Applied:** {len(redactions)}")
                            with st.expander("View redaction details"):
                                for j, redaction in enumerate(redactions[:10]):
                                    st.write(f"{j+1}. {redaction['entity_type']}: '{redaction['original']}' → '{redaction['replacement']}'")
                                if len(redactions) > 10:
                                    st.write(f"... and {len(redactions) - 10} more")

                        st.download_button(
                            f"📥 Download {strategy} version",
                            data=data['content'],
                            file_name=f"{os.path.splitext(filename)[0]}_{strategy}.txt",
                            mime="text/plain",
                            key=f"download_{filename}_{strategy}"
                        )

            st.markdown("---")

    def show_image_results(self):
        """Display image processing results"""
        st.markdown("---")
        st.subheader("🖼️ Image PII Detection Results")

        results_all = st.session_state.image_results

        for filename, results, temp_path in results_all:
            st.markdown(f"### 📸 {filename}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Faces Detected", results['faces_detected'])
            with col2:
                st.metric("Text Regions", len(results['text_regions']))
            with col3:
                st.metric("PII Text Regions", len(results['pii_text_regions']))

            try:
                import cv2
                from PIL import Image

                if os.path.exists(temp_path):
                    original_image = cv2.imread(temp_path)
                    if original_image is not None:
                        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**📷 Original Image:**")
                            st.image(original_rgb, caption="Original", use_container_width=True)

                        with col2:
                            if results['redacted_versions']:
                                first_method = list(results['redacted_versions'].keys())[0]
                                redacted_image = results['redacted_versions'][first_method]
                                redacted_rgb = cv2.cvtColor(redacted_image, cv2.COLOR_BGR2RGB)
                                st.markdown(f"**🔒 {first_method.replace('_', ' ').title()}:**")
                                st.image(redacted_rgb, caption=f"Redacted ({first_method})", use_container_width=True)
                else:
                    st.info("Original image preview unavailable (temp file cleared). Redacted versions still shown below.")

            except Exception as e:
                st.error(f"Error displaying images: {str(e)}")

            if results['pii_text_regions']:
                with st.expander(f"📋 Text PII Details for {filename}", expanded=False):
                    for i, region in enumerate(results['pii_text_regions'], 1):
                        st.write(f"**Region {i}:** '{region['text']}'")
                        st.write(f"  - PII Types: {', '.join(region['pii_types'])}")
                        st.write(f"  - Confidence: {region['confidence']}")

            if results['redacted_versions']:
                st.markdown("**🔒 All Redacted Versions:**")

                versions = list(results['redacted_versions'].items())

                if len(versions) > 0:
                    import cv2
                    from PIL import Image
                    cols = st.columns(min(len(versions), 3))

                    for i, (version_name, processed_image) in enumerate(versions):
                        col_idx = i % 3

                        with cols[col_idx]:
                            st.write(f"**{version_name.replace('_', ' ').title()}**")

                            try:
                                rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                                st.image(rgb_image, caption=version_name, use_container_width=True)

                                pil_image = Image.fromarray(rgb_image)
                                img_buffer = io.BytesIO()
                                pil_image.save(img_buffer, format='PNG')

                                st.download_button(
                                    "📥 Download",
                                    data=img_buffer.getvalue(),
                                    file_name=f"{os.path.splitext(filename)[0]}_{version_name}.png",
                                    mime="image/png",
                                    use_container_width=True,
                                    key=f"download_{filename}_{version_name}"
                                )
                            except Exception as e:
                                st.error(f"Error processing {version_name}: {str(e)}")

            if st.button(f"📋 Generate Report for {filename}", key=f"report_{filename}"):
                try:
                    from image_detector import ImagePIIDetector
                    detector = ImagePIIDetector()
                    report = detector.generate_detection_report(results)

                    st.text_area(
                        f"Detection Report - {filename}",
                        report,
                        height=300
                    )

                    st.download_button(
                        "📥 Download Report",
                        data=report,
                        file_name=f"{os.path.splitext(filename)[0]}_report.txt",
                        mime="text/plain",
                        key=f"report_download_{filename}"
                    )
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")

            st.markdown("---")

    def generate_csv_report(self, df_original, df_cleaned, pii_results):
        """Generate CSV privacy report"""
        report = f"""
PRIVACY SANDBOX - CSV DATA CLEANING REPORT
=========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
File: {st.session_state.get('csv_filename', 'Unknown')}

DATASET INFORMATION:
- Original rows: {len(df_original)}
- Original columns: {len(df_original.columns)}
- Columns: {', '.join(df_original.columns)}

PII DETECTION RESULTS:
- Total PII items found: {sum(len(items) for items in pii_results.values())}
- Affected columns: {len(pii_results)}
- Clean columns: {len(df_original.columns) - len(pii_results)}

REDACTION SUMMARY:
"""

        if pii_results:
            for column, items in pii_results.items():
                report += f"- Column '{column}': {len(items)} PII items redacted\n"
        else:
            report += "- No PII detected - file was already clean\n"

        report += f"""

COMPLIANCE STATUS: {"✓ READY" if pii_results else "✓ CLEAN"}

This report confirms that the CSV dataset has been processed according to 
privacy best practices and is suitable for sharing and analysis.
"""

        return report

    def show_about_page(self):
        """About page with project information"""
        st.header("ℹ️ About Privacy Sandbox")

        st.markdown("""
        ### 🎯 Project Mission
        
        Privacy Sandbox helps researchers, students, and developers work safely with real-world datasets 
        that contain sensitive personal information. Our tool automatically detects and redacts PII, 
        enabling safer collaboration and compliance with privacy regulations.
        
        ### 🛠️ Technology Stack
        
        - **Detection Engine**: Microsoft Presidio + spaCy + Custom Regex
        - **Web Interface**: Streamlit
        - **Data Processing**: Pandas, NumPy
        - **Image Processing**: OpenCV, Tesseract OCR
        - **File Support**: CSV, TXT, JSON, MD, JPG, PNG, BMP
        
        ### 📜 Privacy & Compliance
        
        This tool helps meet requirements for:
        - **GDPR** (General Data Protection Regulation)
        - **FERPA** (Family Educational Rights and Privacy Act)
        - **HIPAA** (Health Insurance Portability and Accountability Act)
        - **CCPA** (California Consumer Privacy Act)
        
        ### 🔒 Data Security
        
        - Files are processed locally - no data sent to external servers
        - Temporary processing only - files are not permanently stored
        - Multiple redaction strategies available
        - Detailed audit trails and reports
        
        ### ✅ Current Features
        
        **CSV Processing:**
        - Smart PII detection with ID preservation
        - Multiple redaction strategies
        - Before/after comparison
        - Downloadable clean files and reports
        
        **Text Processing:**
        - Advanced NLP with multiple detection engines
        - Presidio + spaCy + Custom Regex
        - Multiple redaction strategies: masking, pseudonymization, partial redaction
        - Support for .txt, .json, .md files
        - Comprehensive analysis reports
        
        **Image Processing:**
        - Multi-pass face detection using OpenCV
        - OCR text extraction and PII detection
        - Multiple redaction methods: blur, pixelate, black boxes
        - Support for .jpg, .png, .bmp files
        - Visual before/after comparisons
        
        ### 🔮 Coming Soon
        
        - Excel file support (.xlsx, .xls)
        - Audio transcript cleaning
        - API for automated pipelines
        - Batch processing interface
        """)


def main():
    """Main entry point"""
    app = CompletePrivacySandboxApp()
    app.main()


if __name__ == "__main__":
    main()
