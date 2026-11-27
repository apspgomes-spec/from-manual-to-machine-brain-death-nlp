"""End-to-end demo with completely synthetic report PDFs (no clinical data)."""
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz

from bd_ct_nlp.pipeline import PipelineOptions, process_dataset, write_private_outputs


def make_pdf(path: Path, content: str) -> None:
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((50, 80), content, fontsize=10)
    pdf.save(str(path))
    pdf.close()


def main() -> None:
    with TemporaryDirectory(prefix="bdct_synthetic_") as directory:
        root = Path(directory)
        for group in ("BD", "NBD"):
            (root / group).mkdir()
        make_pdf(
            root / "BD" / "SYNTHETIC_BD.pdf",
            "RELATORIO:\nHemorragia subaracnoideia. Desvio da linha media de 10 mm.",
        )
        make_pdf(
            root / "NBD" / "SYNTHETIC_NBD.pdf",
            "RELATORIO:\nNao se observam sinais de hidrocefalia.",
        )
        data = process_dataset(PipelineOptions(root, root / "outputs"))
        report = write_private_outputs(data, root / "outputs")
        print("Synthetic run — no real records:")
        print("PDFs:", report["N_total"])
        print("Empty texts:", report["empty_clinical_texts"])
        print("TF-IDF fitted:", report["tfidf_fitted"])
        print("ML trained:", report["ml_models_trained"])


if __name__ == "__main__":
    main()
