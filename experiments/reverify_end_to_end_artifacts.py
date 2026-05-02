#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    relative_path: str
    extractor: Optional[Callable[[Mapping[str, Any]], Dict[str, Any]]] = None
    qualitative_checks: tuple[str, ...] = ()


def _sha256_local(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: List[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=timeout)


def _ssh_base(host: str, user: str, password: str) -> List[str]:
    return [
        "sshpass",
        "-p",
        password,
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=10",
        f"{user}@{host}",
    ]


def _remote_file_info(host: str, user: str, password: str, remote_path: str) -> Dict[str, Any]:
    py = (
        "import hashlib, json\n"
        "from pathlib import Path\n"
        f"p = Path({remote_path!r}).expanduser()\n"
        "if not p.exists() or not p.is_file():\n"
        "    print(json.dumps({'exists': False}))\n"
        "else:\n"
        "    h = hashlib.sha256()\n"
        "    with p.open('rb') as f:\n"
        "        for chunk in iter(lambda: f.read(1024*1024), b''):\n"
        "            h.update(chunk)\n"
        "    st = p.stat()\n"
        "    print(json.dumps({'exists': True, 'sha256': h.hexdigest(), 'size': st.st_size, 'mtime': st.st_mtime, 'path': str(p)}))\n"
    )
    proc = _run(_ssh_base(host, user, password) + [f"python3 - <<'PY'\n{py}PY"], timeout=240)
    if proc.returncode != 0:
        return {"exists": False, "error": proc.stderr.strip() or proc.stdout.strip(), "path": remote_path}
    try:
        return json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return {"exists": False, "error": f"Bad JSON from remote: {proc.stdout!r}", "path": remote_path}


# ----- metric extractors -----

def _pick_summary_row(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for row in payload.get("summary_rows", []):
        if str(row.get("intervention_name", row.get("intervention", ""))) == str(name):
            return row
    raise KeyError(name)


def extract_four_lang(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = payload.get("row_aggregate", [])
    keep = [r for r in rows if int(r.get("n_icl", 0)) == 64]
    out = {}
    for r in keep:
        key = f"{r['model']}::{r['pair']}"
        out[key] = {
            "helpful_exact": float(r["helpful_exact"]["mean"]),
            "zs_exact": float(r["zs_exact"]["mean"]),
            "helpful_cer": float(r["helpful_cer"]["mean"]),
            "zs_cer": float(r["zs_cer"]["mean"]),
            "helpful_control_exact_margin": float(r["helpful_control_exact_margin"]["mean"]),
        }
    return out


def extract_hindi_practical(payload: Mapping[str, Any]) -> Dict[str, Any]:
    base = _pick_summary_row(payload, "baseline_no_patch")
    chosen = _pick_summary_row(payload, "chosen_mean_shift")
    flip = _pick_summary_row(payload, "chosen_sign_flip")
    rand = payload.get("random_aggregate", {})
    return {
        "selected_alpha": float(payload.get("selected_alpha", float("nan"))),
        "baseline_em": float(base["exact_match"]["mean"]),
        "chosen_em": float(chosen["exact_match"]["mean"]),
        "baseline_cer": float(base["akshara_cer"]["mean"]),
        "chosen_cer": float(chosen["akshara_cer"]["mean"]),
        "chosen_delta_cer_improvement": float(chosen["delta_cer_improvement"]["mean"]),
        "chosen_delta_first_entry": float(chosen["delta_first_entry_correct"]["mean"]),
        "chosen_delta_gap": float(chosen["delta_first_token_gap_latin"]["mean"]),
        "flip_delta_cer_improvement": float(flip["delta_cer_improvement"]["mean"]),
        "random_delta_cer_improvement_mean": float(rand.get("delta_cer_improvement_mean", float("nan"))),
    }


def extract_hindi_intervention(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = {str(r["intervention"]): r for r in payload.get("summary_rows", [])}
    return {
        "mean_shift_cer": float(rows["calibrated_mean_shift"]["akshara_cer"]["mean"]),
        "sign_flip_cer": float(rows["calibrated_sign_flip"]["akshara_cer"]["mean"]),
        "zero_both_cer": float(rows["zero_both_channels"]["akshara_cer"]["mean"]),
        "mean_shift_delta_cer": float(rows["calibrated_mean_shift"]["delta_cer_improvement"]["mean"]),
    }


def extract_telugu_practical(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = {str(r["intervention"]): r for r in payload.get("summary_rows", [])}
    chosen = rows["chosen_mean_shift"]
    base = rows["baseline_no_patch"]

    def scalar(row: Mapping[str, Any], key: str, alt: Optional[str] = None) -> float:
        val = row.get(key)
        if isinstance(val, Mapping):
            return float(val.get("mean", float("nan")))
        if val is not None:
            return float(val)
        if alt is not None:
            alt_val = row.get(alt)
            if isinstance(alt_val, Mapping):
                return float(alt_val.get("mean", float("nan")))
            if alt_val is not None:
                return float(alt_val)
        return float("nan")

    return {
        "baseline_full_em": scalar(base, "exact_match", "full_exact_match"),
        "chosen_full_em": scalar(chosen, "exact_match", "full_exact_match"),
        "baseline_full_cer": scalar(base, "akshara_cer", "full_akshara_cer"),
        "chosen_full_cer": scalar(chosen, "akshara_cer", "full_akshara_cer"),
        "chosen_delta_cer": scalar(chosen, "delta_akshara_cer", "delta_full_cer_improvement"),
        "chosen_bank": scalar(chosen, "exact_bank_copy_rate", "bank_copy_rate"),
        "base_bank": scalar(base, "exact_bank_copy_rate", "bank_copy_rate"),
        "baseline_cont_cer": scalar(base, "continuation_akshara_cer"),
        "chosen_cont_cer": scalar(chosen, "continuation_akshara_cer"),
        "chosen_delta_cont_cer": scalar(chosen, "delta_continuation_cer_improvement"),
    }


def extract_telugu_mediation(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = {}
    for r in payload.get("summary_rows", []):
        site = r.get("writer_site") or {}
        rows[f"{site.get('name','')}::{r.get('intervention_name','')}"] = r
    # use first writer site from current artifact
    writer_site = None
    for key in rows:
        if key.endswith("::writer_only"):
            writer_site = key.split("::", 1)[0]
            break
    if writer_site is None:
        return {}
    return {
        "writer_site": writer_site,
        "writer_only_delta_gap": float(rows[f"{writer_site}::writer_only"]["delta_mean_gap"]),
        "mediator_overwrite_delta_gap": float(rows[f"{writer_site}::writer_plus_mediator_recipient_overwrite"]["delta_mean_gap"]),
        "offband_overwrite_delta_gap": float(rows[f"{writer_site}::writer_plus_offband_recipient_overwrite"]["delta_mean_gap"]),
    }


def extract_telugu_sufficiency(payload: Mapping[str, Any]) -> Dict[str, Any]:
    rows = {str(r["intervention_name"]): r for r in payload.get("summary_rows", [])}
    return {
        "site_donor_delta_gap": float(rows["site_donor"]["delta_mean_gap"]),
        "offband_donor_delta_gap": float(rows["offband_donor"]["delta_mean_gap"]),
        "random_delta_gap": float(rows["site_random_delta"]["delta_mean_gap"]),
    }


def extract_hindi_patch_safety(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summaries = payload.get("summaries", {})
    return {
        "overall_baseline_exact": float(summaries["all"]["baseline_no_patch"]["exact_match_rate"]),
        "overall_patched_exact": float(summaries["all"]["chosen_mean_shift"]["exact_match_rate"]),
        "english_baseline_exact": float(summaries["english"]["baseline_no_patch"]["exact_match_rate"]),
        "english_patched_exact": float(summaries["english"]["chosen_mean_shift"]["exact_match_rate"]),
    }


def extract_kshot(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary_by_k", {})
    keep = {}
    for k in [8, 32, 64, 128]:
        if str(k) in summary:
            helpful = summary[str(k)]["icl_helpful"]
            keep[str(k)] = {
                "em": float(helpful["mean_exact_match"]),
                "cer": float(helpful["mean_akshara_cer"]),
                "first_entry": float(helpful["mean_first_entry_correct"]),
                "bank": float(helpful.get("exact_bank_copy_rate", float("nan"))),
            }
    return keep


def extract_temp(payload: Mapping[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary_by_temp", {})
    keep = {}
    for t in ["0.0", "0.2", "0.7"]:
        if t in summary:
            helpful = summary[t]["icl_helpful"]
            keep[t] = {
                "em": float(helpful["mean_exact_match"]),
                "cer": float(helpful["mean_akshara_cer"]),
                "exact_bank": float(helpful["exact_bank_copy_rate"]),
                "fuzzy_bank": float(helpful["fuzzy_bank_copy_rate"]),
            }
    return keep


def extract_writer_probe(payload: Mapping[str, Any]) -> Dict[str, Any]:
    heads = payload.get("ranked_heads", [])[:5]
    return {
        "top_heads": [
            {
                "label": str(h.get("head_label", h.get("name", ""))),
                "score": float(h.get("delta_mean_gap", h.get("score", float("nan")))),
            }
            for h in heads
        ],
        "top_group_probe": payload.get("top_group_probe", {}),
        "random_group_probe": payload.get("random_group_probe", {}),
    }


def extract_cross_model(payload: Mapping[str, Any]) -> Dict[str, Any]:
    s = payload["summary"]
    h, c, z = s["icl_helpful"], s["icl_corrupt"], s["zs"]
    return {
        "helpful_em": float(h["mean_exact_match"]),
        "corrupt_em": float(c["mean_exact_match"]),
        "zs_em": float(z["mean_exact_match"]),
        "helpful_cer": float(h["mean_akshara_cer"]),
        "corrupt_cer": float(c["mean_akshara_cer"]),
        "zs_cer": float(z["mean_akshara_cer"]),
        "helpful_first_entry": float(h["mean_first_entry_correct"]),
        "corrupt_first_entry": float(c["mean_first_entry_correct"]),
        "helpful_exact_bank": float(h["exact_bank_copy_rate"]),
        "corrupt_exact_bank": float(c["exact_bank_copy_rate"]),
        "delta_em_helpful_minus_corrupt": float(h["mean_exact_match"] - c["mean_exact_match"]),
        "delta_cer_helpful_minus_corrupt": float(c["mean_akshara_cer"] - h["mean_akshara_cer"]),
    }


def extract_prompt_comp(payload: Mapping[str, Any]) -> Dict[str, Any]:
    s = payload["summary_by_condition"]
    return {
        cond: {
            "em": float(vals["mean_exact_match"]),
            "cer": float(vals["mean_akshara_cer"]),
            "first_entry": float(vals["mean_first_entry_correct"]),
            "exact_bank": float(vals["exact_bank_copy_rate"]),
            "exact_nearest_bank": float(vals["exact_nearest_bank_copy_rate"]),
            "fuzzy_bank": float(vals["fuzzy_bank_copy_rate"]),
            "removed_copy": float(vals.get("exact_removed_target_copy_rate", float("nan"))),
        }
        for cond, vals in s.items()
    }


ARTIFACTS: List[ArtifactSpec] = [
    ArtifactSpec("four_lang_seed_aggregate", "research/results/autoresearch/four_lang_thesis_panel/seed_aggregate.json", extract_four_lang),
    ArtifactSpec("hindi_practical_patch_v1", "research/results/autoresearch/hindi_practical_patch_eval_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json", extract_hindi_practical),
    ArtifactSpec("hindi_practical_patch_review200", "research/results/autoresearch/hindi_practical_patch_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_practical_patch_eval.json", extract_hindi_practical),
    ArtifactSpec("hindi_intervention_v1", "research/results/autoresearch/hindi_intervention_eval_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_intervention_eval.json", extract_hindi_intervention),
    ArtifactSpec("hindi_intervention_review200", "research/results/autoresearch/hindi_intervention_eval_review200_v1/1b/aksharantar_hin_latin/seed42/nicl64/hindi_1b_intervention_eval.json", extract_hindi_intervention),
    ArtifactSpec("telugu_mediation_1b", "research/results/autoresearch/telugu_continuation_mediation_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_mediation_panel.json", extract_telugu_mediation),
    ArtifactSpec("telugu_mediation_4b", "research/results/autoresearch/telugu_continuation_mediation_v1/4b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_mediation_panel.json", extract_telugu_mediation),
    ArtifactSpec("telugu_sufficiency_1b", "research/results/autoresearch/telugu_continuation_final_site_sufficiency_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_final_site_sufficiency.json", extract_telugu_sufficiency),
    ArtifactSpec("telugu_sufficiency_4b", "research/results/autoresearch/telugu_continuation_final_site_sufficiency_v1/4b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_final_site_sufficiency.json", extract_telugu_sufficiency),
    ArtifactSpec("telugu_practical_patch_1b", "research/results/autoresearch/telugu_continuation_practical_patch_eval_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json", extract_telugu_practical),
    ArtifactSpec("telugu_practical_patch_4b", "research/results/autoresearch/telugu_continuation_practical_patch_eval_v1/4b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json", extract_telugu_practical),
    ArtifactSpec("telugu_practical_patch_review200", "research/results/autoresearch/telugu_continuation_practical_patch_eval_review200_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_continuation_practical_patch_eval.json", extract_telugu_practical),
    ArtifactSpec("hindi_patch_safety", "research/results/autoresearch/hindi_patch_safety_audit_v1/1b/seed42/hindi_1b_patch_safety_audit.json", extract_hindi_patch_safety),
    ArtifactSpec("kshot_1b_hindi", "research/results/autoresearch/kshot_regime_sweep_v1/1b/aksharantar_hin_latin/seed42/kshot_regime_sweep.json", extract_kshot),
    ArtifactSpec("kshot_1b_telugu", "research/results/autoresearch/kshot_regime_sweep_v1/1b/aksharantar_tel_latin/seed42/kshot_regime_sweep.json", extract_kshot),
    ArtifactSpec("temperature_1b_telugu", "research/results/autoresearch/telugu_temperature_sweep_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_temperature_sweep.json", extract_temp),
    ArtifactSpec("temperature_4b_telugu", "research/results/autoresearch/telugu_temperature_sweep_v1/4b/aksharantar_tel_latin/seed42/nicl64/telugu_temperature_sweep.json", extract_temp),
    ArtifactSpec("writer_probe_1b_telugu", "research/results/autoresearch/telugu_writer_head_probe_v1/1b/aksharantar_tel_latin/seed42/nicl64/telugu_writer_head_probe.json", extract_writer_probe),
    ArtifactSpec("writer_probe_4b_telugu", "research/results/autoresearch/telugu_writer_head_probe_v1/4b/aksharantar_tel_latin/seed42/nicl64/telugu_writer_head_probe.json", extract_writer_probe),
    ArtifactSpec("cross_model_qwen15_hin_64", "research/results/autoresearch/cross_model_behavioral_v1/qwen2.5-1.5b/aksharantar_hin_latin/seed42/nicl64/cross_model_behavioral.json", extract_cross_model),
    ArtifactSpec("cross_model_qwen15_tel_64", "research/results/autoresearch/cross_model_behavioral_v1/qwen2.5-1.5b/aksharantar_tel_latin/seed42/nicl64/cross_model_behavioral.json", extract_cross_model),
    ArtifactSpec("cross_model_qwen3_hin_64", "research/results/autoresearch/cross_model_behavioral_v1/qwen2.5-3b/aksharantar_hin_latin/seed42/nicl64/cross_model_behavioral.json", extract_cross_model),
    ArtifactSpec("cross_model_qwen3_tel_64", "research/results/autoresearch/cross_model_behavioral_v1/qwen2.5-3b/aksharantar_tel_latin/seed42/nicl64/cross_model_behavioral.json", extract_cross_model),
    ArtifactSpec("cross_model_llama1_hin_64", "research/results/autoresearch/cross_model_behavioral_v1/llama3.2-1b/aksharantar_hin_latin/seed42/nicl64/cross_model_behavioral.json", extract_cross_model),
    ArtifactSpec("cross_model_llama1_tel_64", "research/results/autoresearch/cross_model_behavioral_v1/llama3.2-1b/aksharantar_tel_latin/seed42/nicl64/cross_model_behavioral.json", extract_cross_model),
    ArtifactSpec("cross_model_llama3_hin_64", "research/results/autoresearch/cross_model_behavioral_v1/llama3.2-3b/aksharantar_hin_latin/seed42/nicl64/cross_model_behavioral.json", extract_cross_model),
    ArtifactSpec("cross_model_llama3_tel_64", "research/results/autoresearch/cross_model_behavioral_v1/llama3.2-3b/aksharantar_tel_latin/seed42/nicl64/cross_model_behavioral.json", extract_cross_model),
    ArtifactSpec("prompt_comp_1b_telugu", "research/results/autoresearch/prompt_composition_ablation_v1/1b/aksharantar_tel_latin/seed42/nicl64/prompt_composition_ablation.json", extract_prompt_comp),
    ArtifactSpec("prompt_comp_4b_telugu", "research/results/autoresearch/prompt_composition_ablation_v1/4b/aksharantar_tel_latin/seed42/nicl64/prompt_composition_ablation.json", extract_prompt_comp),
]


def _qualitative_assertions(extracted: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, evidence: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    hp = extracted.get("hindi_practical_patch_v1", {})
    if hp:
        add(
            "Hindi practical patch improves CER and sign flip harms",
            hp["chosen_cer"] < hp["baseline_cer"] and hp["flip_delta_cer_improvement"] < 0,
            f"baseline CER={hp['baseline_cer']:.3f}, chosen CER={hp['chosen_cer']:.3f}, sign-flip ΔCERimp={hp['flip_delta_cer_improvement']:.3f}",
        )

    hp200 = extracted.get("hindi_practical_patch_review200", {})
    if hp200:
        add(
            "Hindi review200 practical patch preserves CER gain and sign-flip harm",
            hp200["chosen_cer"] < hp200["baseline_cer"] and hp200["flip_delta_cer_improvement"] < 0,
            f"baseline CER={hp200['baseline_cer']:.3f}, chosen CER={hp200['chosen_cer']:.3f}, sign-flip ΔCERimp={hp200['flip_delta_cer_improvement']:.3f}",
        )

    hi = extracted.get("hindi_intervention_v1", {})
    if hi:
        add(
            "Hindi signed steering stronger than simple lesioning",
            hi["mean_shift_cer"] < hi["zero_both_cer"] and hi["mean_shift_cer"] < hi["sign_flip_cer"],
            f"mean-shift CER={hi['mean_shift_cer']:.3f}, zero-both CER={hi['zero_both_cer']:.3f}, sign-flip CER={hi['sign_flip_cer']:.3f}",
        )

    hi200 = extracted.get("hindi_intervention_review200", {})
    if hi200:
        add(
            "Hindi review200 signed steering remains stronger than simple lesioning",
            hi200["mean_shift_cer"] < hi200["zero_both_cer"] and hi200["mean_shift_cer"] < hi200["sign_flip_cer"],
            f"mean-shift CER={hi200['mean_shift_cer']:.3f}, zero-both CER={hi200['zero_both_cer']:.3f}, sign-flip CER={hi200['sign_flip_cer']:.3f}",
        )

    for key in ["telugu_sufficiency_1b", "telugu_sufficiency_4b"]:
        ts = extracted.get(key, {})
        if ts:
            add(
                f"{key} true-site donor beats off-band and random controls",
                ts["site_donor_delta_gap"] > ts["offband_donor_delta_gap"] and ts["site_donor_delta_gap"] > ts["random_delta_gap"],
                f"site={ts['site_donor_delta_gap']:.3f}, offband={ts['offband_donor_delta_gap']:.3f}, random={ts['random_delta_gap']:.3f}",
            )

    t1 = extracted.get("temperature_1b_telugu", {})
    if t1 and all(k in t1 for k in ["0.0", "0.2", "0.7"]):
        add(
            "1B Telugu bank-copy rises under sampling",
            t1["0.7"]["exact_bank"] >= t1["0.0"]["exact_bank"] and t1["0.7"]["fuzzy_bank"] >= t1["0.0"]["fuzzy_bank"],
            f"exact bank 0.0->{t1['0.0']['exact_bank']:.3f}, 0.7->{t1['0.7']['exact_bank']:.3f}; fuzzy 0.0->{t1['0.0']['fuzzy_bank']:.3f}, 0.7->{t1['0.7']['fuzzy_bank']:.3f}",
        )

    ll = extracted.get("cross_model_llama1_hin_64", {})
    lt = extracted.get("cross_model_llama1_tel_64", {})
    if ll and lt:
        add(
            "Llama 3.2 1B behaves like a capability-floor counterexample",
            ll["helpful_em"] <= 0.01 and lt["helpful_em"] <= 0.01,
            f"Hindi helpful EM={ll['helpful_em']:.3f}, Telugu helpful EM={lt['helpful_em']:.3f}",
        )

    for prefix in ["cross_model_qwen15", "cross_model_qwen3", "cross_model_llama3"]:
        hin = extracted.get(f"{prefix}_hin_64", {})
        tel = extracted.get(f"{prefix}_tel_64", {})
        if hin and tel:
            add(
                f"{prefix} supports cross-family behavioral regime map",
                hin["delta_em_helpful_minus_corrupt"] > 0 and tel["delta_em_helpful_minus_corrupt"] > 0 and tel["helpful_first_entry"] >= 0.9 and tel["helpful_em"] < hin["helpful_em"],
                f"Hindi ΔEM(h-c)={hin['delta_em_helpful_minus_corrupt']:.3f}; Telugu ΔEM(h-c)={tel['delta_em_helpful_minus_corrupt']:.3f}; Telugu helpful first-entry={tel['helpful_first_entry']:.3f}; helpful EM Hindi/Telugu={hin['helpful_em']:.3f}/{tel['helpful_em']:.3f}",
            )

    tp200 = extracted.get("telugu_practical_patch_review200", {})
    if tp200:
        add(
            "Telugu review200 practical patch remains effectively null",
            tp200["baseline_full_em"] == 0.0 and tp200["chosen_full_em"] == 0.0 and abs(tp200["chosen_delta_cont_cer"]) <= 0.02,
            f"baseline/chosen full EM={tp200['baseline_full_em']:.3f}/{tp200['chosen_full_em']:.3f}; continuation ΔCERimp={tp200['chosen_delta_cont_cer']:.3f}",
        )

    pc1 = extracted.get("prompt_comp_1b_telugu", {})
    if pc1:
        h = pc1["icl_helpful"]
        desc = pc1["icl_helpful_similarity_desc"]
        asc = pc1["icl_helpful_similarity_asc"]
        add(
            "1B Telugu similarity-forward improves entry/CER while increasing nearest-bank copy",
            desc["cer"] < h["cer"] and desc["first_entry"] > h["first_entry"] and desc["exact_nearest_bank"] > h["exact_nearest_bank"],
            f"helpful CER/entry/nearest={h['cer']:.3f}/{h['first_entry']:.3f}/{h['exact_nearest_bank']:.3f}; desc={desc['cer']:.3f}/{desc['first_entry']:.3f}/{desc['exact_nearest_bank']:.3f}",
        )
        add(
            "1B Telugu similarity-back suppresses bank-copy but harms entry",
            asc["exact_bank"] < h["exact_bank"] and asc["first_entry"] < h["first_entry"] and asc["cer"] > h["cer"],
            f"helpful bank/entry/CER={h['exact_bank']:.3f}/{h['first_entry']:.3f}/{h['cer']:.3f}; asc={asc['exact_bank']:.3f}/{asc['first_entry']:.3f}/{asc['cer']:.3f}",
        )

    pc4 = extracted.get("prompt_comp_4b_telugu", {})
    if pc4:
        h = pc4["icl_helpful"]
        add(
            "4B Telugu remains in a stable low-bank regime under prompt composition variants",
            h["exact_bank"] <= 0.05 and h["first_entry"] >= 0.95,
            f"helpful exact bank={h['exact_bank']:.3f}, helpful first-entry={h['first_entry']:.3f}",
        )

    safety = extracted.get("hindi_patch_safety", {})
    if safety:
        add(
            "Hindi patch has nontrivial off-task side effects",
            safety["english_patched_exact"] < safety["english_baseline_exact"] and safety["overall_patched_exact"] < safety["overall_baseline_exact"],
            f"English exact {safety['english_baseline_exact']:.3f}->{safety['english_patched_exact']:.3f}; overall exact {safety['overall_baseline_exact']:.3f}->{safety['overall_patched_exact']:.3f}",
        )

    return checks


def _markdown(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# End-to-end reverification report (2026-04-01)")
    lines.append("")
    lines.append(f"Generated: {report['generated_at_utc']}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This report re-checks a bounded oracle set of paper-critical artifacts locally and, when remote VM information is available, compares the local copies against the VM originals by hash. It also re-extracts the headline numbers used in the current paper narrative instead of trusting prose-only summaries.")
    lines.append("")
    lines.append("## Artifact integrity")
    lines.append("")
    lines.append("| Artifact | Local exists | Remote exists | Hash match |")
    lines.append("| --- | --- | --- | --- |")
    for row in report["artifacts"]:
        lines.append(f"| `{row['name']}` | {row['local']['exists']} | {row['remote'].get('exists', 'n/a')} | {row.get('hash_match', 'n/a')} |")
    lines.append("")
    lines.append("## Qualitative checks")
    lines.append("")
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- **{status}** — {check['name']}: {check['evidence']}")
    lines.append("")
    lines.append("## Extracted headline metrics")
    lines.append("")
    for name, metrics in report["extracted_metrics"].items():
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(metrics, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Re-verify paper-critical transliteration artifacts locally and against the VM.")
    ap.add_argument("--remote-host", type=str, default="")
    ap.add_argument("--remote-user", type=str, default="")
    ap.add_argument("--remote-pass", type=str, default=os.environ.get("VM_PASS", ""))
    ap.add_argument("--remote-root", type=str, default="~/Research/Honors_hindi_patch")
    ap.add_argument("--write-prefix", type=str, default="outputs/end_to_end_reverification_2026-04-01")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    remote_enabled = bool(args.remote_host and args.remote_user and args.remote_pass)
    extracted_metrics: Dict[str, Dict[str, Any]] = {}
    artifacts_out: List[Dict[str, Any]] = []

    for spec in ARTIFACTS:
        local_path = (REPO_ROOT / spec.relative_path).resolve()
        local_info = {
            "path": str(local_path),
            "exists": bool(local_path.exists()),
            "sha256": _sha256_local(local_path),
            "size": local_path.stat().st_size if local_path.exists() else None,
            "mtime": local_path.stat().st_mtime if local_path.exists() else None,
        }
        remote_info: Dict[str, Any] = {"exists": "n/a"}
        if remote_enabled:
            remote_rel = spec.relative_path
            remote_root = str(args.remote_root).rstrip("/")
            remote_path = f"{remote_root}/{remote_rel}"
            remote_info = _remote_file_info(args.remote_host, args.remote_user, args.remote_pass, remote_path)
        hash_match = (
            bool(local_info["exists"])
            and bool(remote_info.get("exists") is True)
            and str(local_info.get("sha256")) == str(remote_info.get("sha256"))
        ) if remote_enabled else "n/a"

        if local_info["exists"] and spec.extractor is not None:
            payload = json.loads(local_path.read_text(encoding="utf-8"))
            extracted_metrics[spec.name] = spec.extractor(payload)

        artifacts_out.append({
            "name": spec.name,
            "relative_path": spec.relative_path,
            "local": local_info,
            "remote": remote_info,
            "hash_match": hash_match,
        })

    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "remote_enabled": remote_enabled,
        "artifacts": artifacts_out,
        "extracted_metrics": extracted_metrics,
        "checks": _qualitative_assertions(extracted_metrics),
    }

    prefix = (REPO_ROOT / str(args.write_prefix)).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path), "checks": report["checks"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
