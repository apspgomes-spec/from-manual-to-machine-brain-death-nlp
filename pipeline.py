"""Reproducible, label-blind preprocessing of Portuguese brain CT reports.

This package implements the V5 *methodological approach* from a local notebook
workflow. It is not a claim of exact byte-for-byte replication of the V5 outputs:
full replication requires the private physician-footer configuration, original
PDFs, clinical adjudication of short reports, and comparison with local audits.

Do not put patient files, identifiers, per-report outputs or n-gram exports on GitHub.
"""
from __future__ import annotations

import json
import hashlib
import platform
from importlib.metadata import version
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

MIN_CLINICAL_WORDS = 15
FUNCTION_WORDS = frozenset({
    "o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da",
    "dos", "das", "em", "no", "na", "nos", "nas", "ao", "aos", "pelo",
    "pela", "pelos", "pelas", "por", "para", "que",
})
# Do not remove e, ou, com, sem, nao, se, mas, sinais, ausencia.

PHRASE_MAP: tuple[tuple[str, str], ...] = (
    (r"hemorragia\s+subaracnoideia", "hemorragia_subaracnoideia"),
    (r"hemorragia\s+sub\s*aracnoideia", "hemorragia_subaracnoideia"),
    (r"hemorragia\s+intraventricular", "hemorragia_intraventricular"),
    (r"edema\s+cerebral", "edema_cerebral"),
    (r"tronco\s+cerebral", "tronco_cerebral"),
    (r"cisternas\s+da\s+base", "cisternas_base"),
    (r"apagamento\s+das\s+cisternas", "apagamento_cisternas"),
    (r"apagamento\s+cisternal", "apagamento_cisternas"),
    (r"desvio\s+da\s+linha\s+media", "desvio_linha_media"),
    (r"herniacao\s+subfalcina", "herniacao_subfalcina"),
    (r"herniacao\s+subfalcial", "herniacao_subfalcina"),
    (r"herniacao\s+transtentorial", "herniacao_transtentorial"),
    (r"herniacao\s+amigdalina", "herniacao_amigdalina"),
    (r"enfarte\s+cerebral", "enfarte_cerebral"),
    (r"hidrocefalia\s+aguda", "hidrocefalia_aguda"),
)

# All matching is performed on accent-folded, lowercase strings.
EXTERNAL_TITLE = re.compile(r"^tac\s*\(\s*relatorio\s+externo\s*\)")
REPORT_MARKER = re.compile(r"^relatorio\s*:\s*")
OTHER_REPORT_MARKER = re.compile(
    r"^(?:relatorio|leitura\s+e\s+interpretacao|resultados)\s*:?\s*$"
)
INLINE_CT = re.compile(r"^tac?\s+(?:ce|cranio)\s*:\s*\S")
FOOTER_START = tuple(re.compile(p) for p in (
    r"^comments\s*$", r"^result\s+comments\s*:?", r"^requisition\s+comments\s*:?",
    r"^relatorio\s+validado\s+por\b", r"^validado\s+por\s*:",
    r"^relatorio\s+assinado\s+por\b", r"^assinado\s+por\b",
    r"^printed\s+(?:on|by)\s*:", r"^order\s+date\s*/\s*time\s*:",
    r"^copyright\b", r"^pagina\s+\d+\s+de\s+\d+\s*$",
    r"^page\s+\d+\s+of\s+\d+\s*$",
))
SKIP_LINES = tuple(re.compile(p) for p in (
    r"^(?:result status|final result|observation dtime|result name|order name|result result name)\s*:?\s*$",
    r"^(?:tc cranio|tac cerebral|tc cranioencefalica)\s*$",
    r"^tac\s*\(\s*relatorio\s+externo\s*\)\s*$",
    r"^relatorio\s*:\s*$", r"^finding\s*:\s*$",
))

# Clinical plausibility check is NOT a predictive feature whitelist.
CLINICAL_CUES = re.compile(
    r"\b(?:hemorrag\w*|hematom\w*|herniac\w*|enfarte\w*|isquem\w*|"
    r"edema\w*|cranio\w*|cerebral\w*|parenquim\w*|hipodens\w*|"
    r"hiperdens\w*|cistern\w*|ventricul\w*|hidrocefalia\w*|"
    r"fratur\w*|contus\w*|lesao\w*|densidade\w*|sulcos\w*|"
    r"vascul\w*|massa\w*|tromb\w*|arteria\w*|tronco\w*|desvio\w*|"
    r"encefal\w*|cerebel\w*|exame\s+sem\s+alterac\w*|"
    r"tc\s*[-–]?\s*ce\s+que\s+revelou)\b"
)

# High-specificity metadata cues; are they exhaustive? No. Residual QA remains mandatory.
RESIDUAL_ADMIN = {
    "cerner": r"\bcerner\b", "crystal": r"\bcrystal\b",
    "mrn": r"\bmrn\b", "pt_name": r"\bpt\s+name\b",
    "no_known_allergies": r"\bno\s+known\b", "printed": r"\bprinted\b",
    "comments": r"\b(?:result\s+comments|requisition\s+comments)\b",
    "copyright": r"\bcopyright\b", "acct": r"\bacct\b",
    "page_marker": r"\bpagina\s+\d+\s+de\s+\d+\b",
    "hff_domain": r"\bhff[.\s-]+min[.\s-]+saude[.\s-]+pt\b",
    "admin_office": r"\b(?:chefia\s+administrativa|tecnica\s+coordenadora)\b",
    "validated_by": r"\b(?:validado\s+por|transcrito\s+por)\b",
    "contact": r"\b(?:email\s*:\s*\S+|www\.[^\s]+)\b",
    "professional_id": r"\bced(?:ula)?\s+(?:prof(?:issional)?|medic[oa])\b",
}
UNSAFE_ADMIN = re.compile(
    r"\b(?:cerner|crystal\s+reports|business\s+objects|copyright|mrn|"
    r"pt\s+(?:name|id)|printed\s+(?:by|on)|all\s+rights\s+reserved|"
    r"allergies|acct\s+no|nurs\s+sta|rm\s*&\s*bed|results\s+report|"
    r"health\s+services\s+inc)\b"
)

# Observed site-specific footer lines, anchored to line start. Exact names are
# read from a PRIVATE file supplied locally; they are never embedded in code.
HFF_FOOTER_RULES = {
    "director": re.compile(r"^diretor\s+do\s+servico\s+de\s+(?:imagiologia|neurorradiologia)\b"),
    "coordination": re.compile(r"^(?:tecnica\s+coordenadora|chefia\s+administrativa)\b"),
    "address": re.compile(r"^ic\s*19\s*[-:]\s*venteira\b"),
    "contact": re.compile(
        r"^(?:\+?\s*351\b.*|email\s*:.*|www\..*)"
        r"hff[.\s-]+min[.\s-]+saude[.\s-]+pt\b"
    ),
}
SIGNATURE_LINE = re.compile(r"^medico\s*:\s*$")
SIGNATURE_CONFIRM = re.compile(r"^(?:validado|transcrito)\s+por\b")

# Negation tokens are only exploratory diagnostics, not predictive features.
UNCERTAIN = re.compile(
    r"(?<!\w)(?:nao\s+se\s+exclui|sem\s+excluir|nao\s+e\s+possivel\s+excluir|"
    r"nao\s+se\s+pode\s+excluir)\b"
)
NEGATIVE = re.compile(
    r"(?<!\w)(?:sem\s+sinais\s+de|sem\s+evidencia\s+de|"
    r"nao\s+se\s+observa(?:m)?|nao\s+se\s+identifica(?:m)?|"
    r"nao\s+se\s+detecta(?:m)?|nao\s+ha\s+(?:evidencia\s+de|sinais\s+de)?|"
    r"ausencia\s+de|sem\s+|nao\s+existe(?:m)?)\b"
)
NEGATION_CUE = re.compile(f"(?:{UNCERTAIN.pattern})|(?:{NEGATIVE.pattern})")
SCOPE_STOP = re.compile(r"[.;:\n]")
CONTRAST = re.compile(
    r",\s*(?:com|mas|porem|contudo|observa-se|identifica-se)\b|"
    r"\b(?:mas|porem|contudo|todavia|entretanto)\b"
)
CONTENT_TOKEN = re.compile(r"\b[a-z][a-z0-9_]*\b")
NEG_IGNORE = frozenset({
    "o", "a", "os", "as", "de", "do", "da", "dos", "das", "em", "no", "na",
    "nos", "nas", "e", "ou", "com", "por", "para", "que", "se", "um", "uma",
    "sinal", "sinais", "evidencia", "evidencias", "ha", "existe", "existem",
    "observa", "identifica", "sem", "nao", "ausencia", "outras", "outros",
    "qualquer", "quaisquer",
})


def fold(text: object) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower().replace("\x00", " "))
    return "".join(c for c in text if not unicodedata.combining(c))


def tidy_line(line: str) -> str:
    return re.sub(r"[ \t]+", " ", fold(line)).strip()


def extract_pdf(path: Path) -> str:
    with fitz.open(str(path)) as pdf:
        return "\n".join(page.get_text("text") for page in pdf)


def _is_footer(line: str) -> bool:
    s = tidy_line(line)
    return any(pattern.search(s) for pattern in FOOTER_START)


def _base_clinical_section(raw: str) -> dict:
    """V1 anchored extractor, preserved to document the original provenance."""
    lines = [line.strip() for line in raw.splitlines()]
    low = [tidy_line(line) for line in lines]
    external = [
        i for i, line in enumerate(low)
        if EXTERNAL_TITLE.search(line) and "results report" not in line
    ]
    is_cerner = bool(
        external and any("cerner health" in line or "results report" in line for line in low)
    )
    if is_cerner:
        anchor = external[0]
        footer = next(
            (i for i in range(anchor + 1, len(lines)) if _is_footer(lines[i])),
            len(lines),
        )
        report_markers = [i for i in range(anchor + 1, footer) if REPORT_MARKER.search(low[i])]
        if report_markers:
            start, mode = report_markers[0], "CERNER_REPORT_SECTION"
        else:
            start, mode = anchor + 1, "CERNER_AFTER_EXTERNAL_MARKER"
    else:
        report_markers = [i for i, line in enumerate(low) if REPORT_MARKER.search(line)]
        if not report_markers:
            report_markers = [i for i, line in enumerate(low) if OTHER_REPORT_MARKER.fullmatch(line)]
        if not report_markers:
            report_markers = [i for i, line in enumerate(low) if INLINE_CT.search(line)]
        if report_markers:
            start, mode = report_markers[-1], "STANDARD_REPORT_SECTION"
        else:
            return {
                "clinical_raw": "", "mode": "REVIEW_NO_REPORT_ANCHOR",
                "raw_lines": len(lines), "clinical_lines": 0,
                "removed_before_lines": 0, "removed_after_lines": 0,
            }
    stop = next((i for i in range(start, len(lines)) if _is_footer(lines[i])), len(lines))
    selected = []
    for i in range(start, stop):
        line = lines[i]
        if i == start:
            line = re.sub(
                r"^\s*(?:relat[oó]rio|leitura\s+e\s+interpreta[cç][aã]o|"
                r"resultados|tac?\s+(?:ce|cranio))\s*:\s*",
                "", line, flags=re.I,
            )
            if tidy_line(line) in ("relatorio", "leitura e interpretacao", "resultados"):
                line = ""
        if line and not any(p.fullmatch(tidy_line(line)) for p in SKIP_LINES):
            selected.append(line)
    return {
        "clinical_raw": "\n".join(selected), "mode": mode,
        "raw_lines": len(lines), "clinical_lines": len(selected),
        "removed_before_lines": start, "removed_after_lines": len(lines) - stop,
    }


def clinical_section(raw: str) -> dict:
    """Anchored extraction; conservative headerless fallback; never use labels."""
    original = _base_clinical_section(raw)
    if original["clinical_raw"].strip():
        return original
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    candidate = "\n".join(
        line for i, line in enumerate(lines)
        if not (i <= 2 and re.fullmatch(r"\d{4,}", line))
    )
    normalized = fold(candidate)
    if candidate and not UNSAFE_ADMIN.search(normalized) and CLINICAL_CUES.search(normalized):
        return {
            "clinical_raw": candidate,
            "mode": "FALLBACK_CLINICAL_NO_HEADER",
            "raw_lines": len(lines), "clinical_lines": len(candidate.splitlines()),
            "removed_before_lines": 0, "removed_after_lines": 0,
        }
    return original


def remove_hff_footer(text: str) -> tuple[str, dict[str, int]]:
    """Remove only site-specific, line-anchored administrative footer lines."""
    kept: list[str] = []
    removed: Counter = Counter()
    for line in str(text).splitlines():
        rule = next(
            (name for name, rx in HFF_FOOTER_RULES.items() if rx.search(tidy_line(line))),
            None,
        )
        if rule is None:
            kept.append(line)
        else:
            removed[rule] += 1
    return "\n".join(kept).strip(), dict(removed)


def load_private_signatures(path: Path | None) -> list[str]:
    """Private UTF-8 text file: one professional signature marker per line."""
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"Private signature file not found: {path}")
    lines = [fold(line.strip()) for line in path.read_text(encoding="utf-8-sig").splitlines()]
    names = [line for line in lines if line and not line.startswith("#")]
    if not names:
        raise ValueError("Private signature file contains no markers")
    return sorted(set(names), key=len, reverse=True)


def cut_physician_footer(text: str, markers: Iterable[str]) -> tuple[str, bool, str]:
    """If a configured signature starts a footer, truncate at its whole line.

    Do not embed actual clinician names in this public repository.
    """
    names = [re.escape(fold(n)) for n in markers if n.strip()]
    if not names:
        return text, False, "NO_PRIVATE_MARKERS"
    regex = re.compile(r"\b(?:" + "|".join(names) + r")\b")
    lines = str(text).splitlines()
    for i, line in enumerate(lines):
        match = regex.search(fold(line))
        if match:
            prefix = "\n".join(lines[:i]).strip()
            if len(prefix.split()) < 15:
                return text, False, "REVIEW_EARLY_NAME"
            if len(line[:match.start()].strip()) > 100:
                return text, False, "REVIEW_NAME_IN_LONG_LINE"
            return prefix, True, "CUT_AT_PHYSICIAN"
    return text, False, "NO_PHYSICIAN_FOOTER"


def remove_signature_block(text: str) -> tuple[str, bool, str]:
    """Remove line `Médico:` plus subsequent `Validado/Transcrito por` block.

    This is an observed institutional layout, not a universal heuristic.
    """
    lines = str(text).splitlines()
    for j, line in enumerate(lines):
        if not SIGNATURE_LINE.fullmatch(tidy_line(line)):
            continue
        if any(SIGNATURE_CONFIRM.match(tidy_line(s)) for s in lines[j + 1:j + 5]):
            prefix = "\n".join(lines[:j]).strip()
            if prefix:
                return prefix, True, "CUT_SIGNATURE_BLOCK"
            return text, False, "REVIEW_SIGNATURE_AT_START"
    return text, False, "NO_SIGNATURE_BLOCK"


def normalize_clinical(text: str) -> str:
    s = fold(text)
    for pattern, canonical in PHRASE_MAP:
        s = re.sub(r"\b(?:" + pattern + r")\b", canonical, s)
    return re.sub(r"[ \t]+", " ", s).strip()


def negation_audit(text: str) -> tuple[str, dict[str, int]]:
    """Exploratory lexical annotation ONLY. Never feed this output to TF-IDF."""
    parts: list[str] = []
    pos = 0
    counts: Counter = Counter()
    for cue in NEGATION_CUE.finditer(text):
        if cue.start() < pos:
            continue
        parts.append(text[pos:cue.end()])
        tail = text[cue.end():]
        boundaries = [len(tail), 120]
        for pattern in (SCOPE_STOP, CONTRAST, NEGATION_CUE):
            match = pattern.search(tail)
            if match:
                boundaries.append(match.start())
        end = min(boundaries)
        context = tail[:end]
        prefix = "incerto_" if UNCERTAIN.fullmatch(cue.group()) else "neg_"
        tagged = 0

        def transform(match: re.Match) -> str:
            nonlocal tagged
            word = match.group()
            if word in NEG_IGNORE or word.startswith(("neg_", "incerto_")):
                return word
            tagged += 1
            return prefix + word

        parts.append(CONTENT_TOKEN.sub(transform, context))
        counts["cues"] += 1
        counts["matched" if tagged else "unmatched"] += 1
        counts["tagged"] += tagged
        pos = cue.end() + end
    parts.append(text[pos:])
    return "".join(parts), {
        "cues": int(counts["cues"]),
        "matched": int(counts["matched"]),
        "unmatched": int(counts["unmatched"]),
        "tagged": int(counts["tagged"]),
    }


def tokenize_for_ngrams(text: str) -> str:
    s = fold(text)
    s = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", " ", s)
    s = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", s)
    s = re.sub(r"\b(?:19|20)\d{2}\b", " ", s)
    s = re.sub(r"\b\d{5,}\b", " ", s)
    s = re.sub(r"(?<=\d)[,.](?=\d)", "_", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(w for w in s.split() if w not in FUNCTION_WORDS)


def descriptive_ngrams(texts: Iterable[str], n: int, top: int = 30) -> pd.DataFrame:
    values = [str(text) for text in texts if str(text).strip()]
    if not values:
        return pd.DataFrame(columns=("ngram", "frequencia", "n_relatorios"))
    vectorizer = CountVectorizer(
        ngram_range=(n, n), min_df=1, lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )
    x = vectorizer.fit_transform(values)
    return (
        pd.DataFrame({
            "ngram": vectorizer.get_feature_names_out(),
            "frequencia": np.asarray(x.sum(axis=0)).ravel(),
            "n_relatorios": np.asarray((x > 0).sum(axis=0)).ravel(),
        })
        .sort_values(["frequencia", "ngram"], ascending=[False, True])
        .head(top).reset_index(drop=True)
    )


def make_tfidf_for_cv() -> TfidfVectorizer:
    """Return UNFITTED TF-IDF; fit only inside each outer training fold."""
    return TfidfVectorizer(
        ngram_range=(1, 3), min_df=2, max_features=5000,
        token_pattern=r"(?u)\b\w+\b", lowercase=False,
        sublinear_tf=True, norm="l2",
    )


def _residual_checks(text: str, markers: Iterable[str]) -> dict[str, bool]:
    normalized = fold(text)
    result = {key: bool(re.search(regex, normalized)) for key, regex in RESIDUAL_ADMIN.items()}
    result["physician_marker"] = any(
        bool(re.search(r"\b" + re.escape(fold(n)) + r"\b", normalized))
        for n in markers
    )
    return result


@dataclass(frozen=True)
class PipelineOptions:
    base_dir: Path
    output_dir: Path
    private_signatures: Path | None = None


def process_text(raw: str, markers: Iterable[str] = ()) -> dict:
    """One report, label-blind. All original text remains in PRIVATE exports."""
    section = clinical_section(raw)
    original_clinical = section["clinical_raw"]
    no_footer, footer_count = remove_hff_footer(original_clinical)
    no_physician, physician_cut, physician_status = cut_physician_footer(no_footer, markers)
    no_signature, signature_cut, signature_status = remove_signature_block(no_physician)
    # PRIVATE audit evidence: exact removed tails for clinical review.
    physician_tail = (
        no_footer[len(no_physician):].strip()
        if physician_cut and no_footer.startswith(no_physician) else ""
    )
    signature_tail = (
        no_physician[len(no_signature):].strip()
        if signature_cut and no_physician.startswith(no_signature) else ""
    )
    norm = normalize_clinical(no_signature)
    aware, neg = negation_audit(norm)
    # The critical V5 decision: natural-language negation retained in predictors.
    final = tokenize_for_ngrams(norm)
    residual = _residual_checks(no_signature, markers)
    flags = []
    if section["mode"].startswith("REVIEW"):
        flags.append("NO_ANCHOR")
    # QA-only conservative flag for narratives with 15 or fewer tokens.
    if len(final.split()) <= MIN_CLINICAL_WORDS:
        flags.append("SHORT_CLINICAL_TEXT")
    if any(residual.values()):
        flags.append("ADMIN_RESIDUE")
    if physician_status.startswith("REVIEW") or signature_status.startswith("REVIEW"):
        flags.append("FOOTER_REVIEW")
    if neg["unmatched"]:
        flags.append("NEGATION_AUDIT_ONLY")
    return {
        "raw_text": raw,
        "clinical_text": no_signature,
        "normalized_text": norm,
        "negation_aware_text": aware,
        "nlp_text": final,
        "template": section["mode"],
        "raw_words": len(raw.split()),
        "clinical_words": len(final.split()),
        "removed_admin_lines": sum(footer_count.values()),
        "physician_footer_cut": physician_cut,
        "signature_block_cut": signature_cut,
        "PRIVATE_physician_removed_tail": physician_tail,
        "PRIVATE_signature_removed_tail": signature_tail,
        "physician_footer_status": physician_status,
        "signature_block_status": signature_status,
        "negation_cues": neg["cues"],
        "negation_unmatched_cues": neg["unmatched"],
        "negation_tagged_tokens": neg["tagged"],
        "review_flags": "|".join(flags),
        **{f"resid_{key}": int(value) for key, value in residual.items()},
    }


def process_dataset(options: PipelineOptions) -> pd.DataFrame:
    """Read PDF folders, without touching originals. Exports are PRIVATE."""
    markers = load_private_signatures(options.private_signatures)
    base_dir = Path(options.base_dir)
    if not (base_dir / "BD").is_dir() or not (base_dir / "NBD").is_dir():
        raise FileNotFoundError("Both BD/ and NBD/ directories are required")
    records: list[dict] = []
    seen_filenames: set[str] = set()
    for group, label in (("BD", 1), ("NBD", 0)):
        for pdf_path in sorted((base_dir / group).glob("*.pdf")):
            if pdf_path.name.lower() in seen_filenames:
                raise ValueError("Repeated PDF filename across folders")
            seen_filenames.add(pdf_path.name.lower())
            raw = extract_pdf(pdf_path)
            result = process_text(raw, markers)
            # DO NOT export the original PDF filename in public files.
            result.update({"group": group, "label": label, "filename": pdf_path.name})
            records.append(result)
    if not records:
        raise ValueError("No PDF files found in BD/ or NBD/")
    return pd.DataFrame(records)


def summarize(data: pd.DataFrame) -> dict:
    """No filenames, free text, or row-level information in summary."""
    summary = {
        "software_versions": {
            "Python": platform.python_version(),
            "PyMuPDF": version("PyMuPDF"),
            "pandas": version("pandas"),
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
        },
        "pipeline_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "N_total": int(len(data)),
        "N_BD": int((data["group"] == "BD").sum()),
        "N_NBD": int((data["group"] == "NBD").sum()),
        "templates": {
            str(index): {str(col): int(value) for col, value in row.items()}
            for index, row in pd.crosstab(data["template"], data["group"]).to_dict("index").items()
        },
        "median_raw_words": {str(k): float(v) for k, v in data.groupby("group")["raw_words"].median().items()},
        "median_nlp_words": {str(k): float(v) for k, v in data.groupby("group")["clinical_words"].median().items()},
        "empty_clinical_texts": int((data["clinical_text"].str.strip() == "").sum()),
        "short_reports": int(data["review_flags"].str.contains("SHORT_CLINICAL_TEXT", regex=False).sum()),
        "no_anchor": int(data["review_flags"].str.contains("NO_ANCHOR", regex=False).sum()),
        "admin_residue_detected": int(data["review_flags"].str.contains("ADMIN_RESIDUE", regex=False).sum()),
        "physician_footer_cuts": int(data["physician_footer_cut"].sum()),
        "signature_block_cuts": int(data["signature_block_cut"].sum()),
        "negation_cues_audit_only": int(data["negation_cues"].sum()),
        "negation_unmatched_audit_only": int(data["negation_unmatched_cues"].sum()),
        "tfidf_fitted": False,
        "ml_models_trained": False,
        "note": "Zero residues means zero matches to the prespecified QC regex; not certified anonymisation.",
    }
    return summary


def write_private_outputs(data: pd.DataFrame, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_dir / "PRIVATE_text_processing_full.csv", index=False, encoding="utf-8-sig")
    audit_cols = [
        "filename", "group", "template", "raw_words", "clinical_words",
        "review_flags", "removed_admin_lines", "physician_footer_cut",
        "signature_block_cut", "physician_footer_status", "signature_block_status",
        "negation_cues", "negation_unmatched_cues", "negation_tagged_tokens",
    ] + [name for name in data.columns if name.startswith("resid_")]
    data[audit_cols].to_csv(out_dir / "PRIVATE_audit_by_pdf.csv", index=False, encoding="utf-8-sig")
    removed = data.loc[
        data["physician_footer_cut"] | data["signature_block_cut"],
        ["filename", "group", "PRIVATE_physician_removed_tail", "PRIVATE_signature_removed_tail"],
    ]
    removed.to_csv(out_dir / "PRIVATE_removed_tails_for_review.csv", index=False, encoding="utf-8-sig")
    for n in (1, 2, 3):
        # N-grams may contain identifying names; save only in PRIVATE outputs.
        descriptive_ngrams(data["nlp_text"], n, top=30).to_csv(
            out_dir / f"PRIVATE_ngram_{n}_descriptive.csv", index=False, encoding="utf-8-sig"
        )
    summary = summarize(data)
    (out_dir / "LOCAL_audit_summary_no_patient_text.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
