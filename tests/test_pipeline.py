from pathlib import Path

import fitz

from bd_ct_nlp.pipeline import (
    _base_clinical_section,
    clinical_section,
    cut_physician_footer,
    descriptive_ngrams,
    make_tfidf_for_cv,
    negation_audit,
    normalize_clinical,
    process_dataset,
    process_text,
    remove_hff_footer,
    remove_signature_block,
    tokenize_for_ngrams,
    PipelineOptions,
    write_private_outputs,
)


def test_extract_cerner_report_ignores_metadata():
    pdf_text = """TAC (Relatório Externo) Results Report
PT NAME: SYNTHETIC PERSON
MRN: 123456
TAC (Relatório Externo)
TAC CEREBRAL
RELATÓRIO:
Sem sinais de edema cerebral. Desvio da linha média de 10 mm.
Comments
Printed By: SYNTHETIC PERSON"""
    result = clinical_section(pdf_text)
    assert result["mode"] == "CERNER_REPORT_SECTION"
    assert "10 mm" in result["clinical_raw"]
    assert "MRN" not in result["clinical_raw"]
    assert "Printed By" not in result["clinical_raw"]


def test_standard_report_and_headerless_fallback():
    standard = clinical_section("Name: EXAMPLE\nRELATÓRIO:\nHemorragia subaracnoideia.\nValidado por: EXAMPLE")
    assert standard["mode"] == "STANDARD_REPORT_SECTION"
    assert "Name:" not in standard["clinical_raw"]
    assert "Validado" not in standard["clinical_raw"]
    fallback = clinical_section("9999999\nExame sem alterações.")
    assert fallback["mode"] == "FALLBACK_CLINICAL_NO_HEADER"
    assert fallback["clinical_raw"] == "Exame sem alterações."
    complex_fallback = clinical_section("9999999\nTC-CE que revelou contusões hemorrágicas.")
    assert complex_fallback["mode"] == "FALLBACK_CLINICAL_NO_HEADER"
    assert "contusões" in complex_fallback["clinical_raw"]


def test_unrecognised_administrative_document_fails_closed():
    sec = clinical_section("MRN: 123456\nPt Name: EXAMPLE\nwithout radiology section")
    assert sec["mode"].startswith("REVIEW")
    assert sec["clinical_raw"] == ""


def test_physician_footer_cut_and_early_review():
    text = "Hemorragia intraventricular extensa associada a dilatação ventricular importante e desvio da linha média de aproximadamente 10 mm em exame inicial.\nDr. Example Physician\nRua: example."
    # Generic synthetic name, never real physician in public tests.
    result, cut, status = cut_physician_footer(text, ["Example Physician"])
    assert cut and status == "CUT_AT_PHYSICIAN"
    assert "Rua" not in result and "10 mm" in result
    unchanged, cut, status = cut_physician_footer("Example Physician\nEdema cerebral", ["Example Physician"])
    assert not cut and status == "REVIEW_EARLY_NAME"
    assert unchanged.startswith("Example Physician")


def test_hff_admin_lines_removed_but_radiology_preserved():
    text = (
        "Desvio da linha média de 10 mm.\n"
        "Chefia Administrativa: EXAMPLE\n"
        "IC 19 - Venteira\n"
        "Herniação presente."
    )
    clean, removed = remove_hff_footer(text)
    assert "10 mm" in clean and "Herniação presente" in clean
    assert sum(removed.values()) == 2
    assert "Chefia" not in clean


def test_signature_block_removed():
    txt = "Achados compatíveis com edema cerebral.\nMédico:\nAssinatura\nValidado por: EXAMPLE\nTranscrito por: EXAMPLE"
    clean, removed, code = remove_signature_block(txt)
    assert removed and code == "CUT_SIGNATURE_BLOCK"
    assert "Validado" not in clean
    assert "edema cerebral" in clean


def test_negation_preserved_for_main_features_and_audit_only():
    source = "Sem sinais de edema cerebral, com herniação subfalcina. Não se exclui desvio da linha média."
    result = process_text(source)
    # No label is supplied to process_text. Audit tags MUST not enter nlp_text.
    assert "neg_" not in result["nlp_text"]
    assert "incerto_" not in result["nlp_text"]
    assert "sem sinais" in result["nlp_text"]
    assert "nao se exclui" in result["nlp_text"]
    aware, stats = negation_audit(normalize_clinical(source))
    assert "neg_edema_cerebral" in aware
    assert "incerto_desvio_linha_media" in aware
    assert "neg_herniacao_subfalcina" not in aware
    assert stats["cues"] == 2


def test_numbers_stopwords_and_ngrams():
    text = tokenize_for_ngrams("Desvio de 10 mm; hematoma de 6,6 cm; 2025; do encéfalo.")
    assert "10 mm" in text and "6_6 cm" in text
    assert "2025" not in text
    assert " de " not in f" {text} " and " do " not in f" {text} "
    assert "sem" in tokenize_for_ngrams("sem sinais de edema")
    assert "nao" in tokenize_for_ngrams("não se observa")
    ngrams = descriptive_ngrams(["efeito massa", "efeito massa", "massa efeito"], 2, top=5)
    assert int(ngrams.loc[ngrams.ngram == "efeito massa", "n_relatorios"].iloc[0]) == 2
    assert make_tfidf_for_cv().ngram_range == (1, 3)
    assert not hasattr(make_tfidf_for_cv(), "vocabulary_")


def test_local_dataset_with_synthetic_pdfs(tmp_path):
    base = tmp_path / "data"
    for group in ("BD", "NBD"):
        (base / group).mkdir(parents=True)
    texts = {
        "BD": "RELATÓRIO:\nHemorragia subaracnoideia e edema cerebral.\nValidado por: FAKE",
        "NBD": "RELATÓRIO:\nExame sem alterações.\nValidado por: FAKE",
    }
    for group, txt in texts.items():
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((60, 70), txt)
        doc.save(base / group / f"synthetic_{group}.pdf")
        doc.close()
    out_dir = tmp_path / "outputs"
    df = process_dataset(PipelineOptions(base, out_dir))
    summary = write_private_outputs(df, out_dir)
    assert summary["N_total"] == 2
    assert summary["empty_clinical_texts"] == 0
    assert summary["tfidf_fitted"] is False
    assert (out_dir / "PRIVATE_audit_by_pdf.csv").exists()
    assert (out_dir / "LOCAL_audit_summary_no_patient_text.json").exists()
