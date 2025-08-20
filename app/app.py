import streamlit as st
from PIL import Image
from modules.ocr.extract import extract_and_check
from modules.visual.ela import ela_image
from modules.signature.verify_signature import verify_signature
from utils.io import save_upload
import os

st.set_page_config(page_title="ForensicAgent (Phase 1 - Week 1)", layout="wide")
st.title("ForensicAgent — Multi-Modal Document Forgery (Phase 1 – Week 1)")
st.caption("OCR + ELA + Signature baseline in a simple Streamlit UI")


with st.sidebar:
    st.header("Expected Entities (OCR Check)")
    name = st.text_input("Name contains", value="RAHUL")  # edit for your tests
    dob = st.text_input("DOB contains", value="17/11/2002")
    docid = st.text_input("ID contains", value="")
    expected = {"NAME": name, "DOB": dob, "ID": docid}
    st.markdown("---")
    st.info("Tip: Adjust expected values to match your test document.")


# Helper for file validation
def validate_file(file, max_size_mb=5):
    if file is None:
        return False, "No file uploaded."
    if file.size > max_size_mb * 1024 * 1024:
        return False, f"File too large (> {max_size_mb} MB)."
    if not file.type.startswith("image/"):
        return False, "File is not an image."
    return True, ""


col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Document (PNG/JPG)")
    doc_file = st.file_uploader(
        "Document Image", type=["png", "jpg", "jpeg"], key="doc"
    )
    sig_ref = st.file_uploader(
        "Reference Signature", type=["png", "jpg", "jpeg"], key="sigref"
    )
    sig_test = st.file_uploader(
        "Test Signature", type=["png", "jpg", "jpeg"], key="sigtest"
    )

    # Show thumbnails for uploaded files
    if doc_file:
        st.image(doc_file, caption="Document Preview", width=200)
    if sig_ref:
        st.image(sig_ref, caption="Reference Signature Preview", width=120)
    if sig_test:
        st.image(sig_test, caption="Test Signature Preview", width=120)

    run = st.button("Run Analysis", type="primary")

with col2:
    st.subheader("Results")
    if run:
        valid, msg = validate_file(doc_file)
        if not valid:
            st.error(f"Document: {msg}")
        elif sig_ref and not validate_file(sig_ref)[0]:
            st.error("Reference Signature: Not a valid image or too large.")
        elif sig_test and not validate_file(sig_test)[0]:
            st.error("Test Signature: Not a valid image or too large.")
        else:
            with st.spinner("Processing document..."):
                # Save uploads
                path_doc = save_upload(doc_file, "document")
                path_sig_ref = save_upload(sig_ref, "sig_ref") if sig_ref else None
                path_sig_test = save_upload(sig_test, "sig_test") if sig_test else None

                # 1) OCR
                st.markdown("### OCR & Consistency Check")
                try:
                    text, score_ocr, flags = extract_and_check(path_doc, expected)
                    st.text_area("Extracted Text", text, height=200)
                    st.write("**OCR Consistency Score:**", round(score_ocr, 2))
                    st.write("Flags:", flags)
                    st.success("OCR extraction complete.")
                    # Download extracted text
                    st.download_button(
                        "Download OCR Text", text, file_name="ocr_text.txt"
                    )
                except Exception as e:
                    st.error(f"OCR failed: {e}")

                # 2) ELA
                st.markdown("### Error Level Analysis (ELA)")
                try:
                    ela = ela_image(path_doc, quality=90, enhance=10.0)
                    st.image(ela, caption="ELA Visualization", use_column_width=True)
                    st.caption(
                        "Bright regions are likely to have compression artifacts → potential tampering."
                    )
                    # Download ELA image
                    from io import BytesIO

                    buf = BytesIO()
                    ela.save(buf, format="JPEG")
                    st.download_button(
                        "Download ELA Image",
                        buf.getvalue(),
                        file_name="ela.jpg",
                        mime="image/jpeg",
                    )
                    st.success("ELA analysis complete.")
                except Exception as e:
                    st.error(f"ELA failed: {e}")

                # 3) Signature baseline
                st.markdown("### Signature Verification (Baseline)")
                if path_sig_ref and path_sig_test:
                    try:
                        score_sig = verify_signature(path_sig_ref, path_sig_test)
                        st.write("**Signature Similarity:**", round(score_sig, 3))
                        st.caption(
                            "Baseline template matching. Higher is more similar (0..1)."
                        )
                        st.success("Signature verification complete.")
                    except Exception as e:
                        st.error(f"Signature check failed: {e}")
                else:
                    st.info(
                        "Upload both reference and test signatures to compute similarity."
                    )
    else:
        st.info("Upload at least a document image and click 'Run Analysis'.")
