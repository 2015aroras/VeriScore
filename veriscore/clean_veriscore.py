import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from .utils import get_veriscore


def _mark_gemini_errors(veriscore_data: list[dict]):
    gemini_error_messages = [
        "FINISH_REASON_UNSPECIFIED",
        "STOP",
        "MAX_TOKENS",
        "SAFETY",
        "RECITATION",
        "OTHER",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
    ]

    for entry in veriscore_data:
        if (
            entry["model"] == "Gemini 1.5 Pro"
            and entry["original_answer"] in gemini_error_messages
        ):
            entry["error"] = True
        else:
            entry["error"] = False


def _add_original_answers(veriscore_data: list[dict], calmqa_data: list[dict]):
    answers_by_eng_q_model_and_language: dict[tuple[str, str, str], str] = {}
    for entry in calmqa_data:
        eng_question = entry["question"]["translations"]["English"]["text"].strip()
        for answer in entry["answers"]:
            language = answer["language"]
            model = answer["prompting_state"]["model_name"]
            answer_text = answer["translations"][language]["text"]

            answers_by_eng_q_model_and_language[(eng_question, model, language)] = (
                answer_text
            )

    for entry in veriscore_data:
        eng_question = entry["question"].strip()
        model = entry["model"]
        language = entry["prompt_source"].split("_")[0]

        entry["original_answer"] = answers_by_eng_q_model_and_language[
            (eng_question, model, language)
        ]


def _mark_errors(veriscore_data: list[dict]):
    if "original_answer" not in veriscore_data[0]:
        raise ValueError()

    _mark_gemini_errors(veriscore_data)


def _get_f1(triplet: tuple[int, int, int], K: int):
    precision = triplet[0] / (triplet[1] or 1)
    recall = min(triplet[0] / (K or 1), 1.)

    if precision == 0 or recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def _add_veriscores(veriscore_data: list[dict]):
    model_domain_triplet_dict = defaultdict(
        lambda: defaultdict(list)
    )  # triplet = [[supported, total, # of sentences], ...]

    for entry in veriscore_data:
        triplet = [0, 0, 0]
        triplet[1] = len(entry["all_claims"])
        triplet[2] = len(entry["claim_list"])
        if not entry["claim_search_results"]:
            triplet[0] = 0
        else:
            for claim_veri_res in entry["claim_verification_result"]:
                if claim_veri_res["verification_result"] == "supported":
                    triplet[0] += 1

        if entry["error"]:
            # Handle bad entries
            triplet = [0, 0, 0]

        entry["triplet"] = triplet

        model_name = entry["model"]

        domain = entry["prompt_source"].replace("_non_cultural", "_noncultural")
        model_domain_triplet_dict["Overall"][model_name].append(triplet)
        model_domain_triplet_dict[domain][model_name].append(triplet)
        model_domain_triplet_dict[domain.split("_")[0]][model_name].append(triplet)
        model_domain_triplet_dict[domain.split("_")[1]][model_name].append(triplet)

    df, domain_K = get_veriscore(model_domain_triplet_dict)

    for entry in veriscore_data:
        model_name = entry["model"]
        domain = entry["prompt_source"].replace("non_cultural", "noncultural")

        entry["veriscore"] = {
            dom: _get_f1(tuple(entry["triplet"]), domain_K[dom]["K_median"])
            for dom in ("Overall", domain, *domain.split("_"))
        }
        entry["veriscore_domain_k"] = {
            dom: domain_K[dom]["K_median"]
            for dom in ("Overall", domain, *domain.split("_"))
        }


def clean_veriscore(
    veriscore_data_paths: list[str],
    dataset_load_paths: list[str],
    output_path: str,
    *,
    compress: bool = False
):
    veriscore_data = [
        json.loads(line)
        for veriscore_path in veriscore_data_paths
        for line in Path(veriscore_path).open(encoding="utf-8").readlines()
    ]

    # print([
    #     line
    #     for veriscore_path in veriscore_data_paths
    #     for line in Path(veriscore_path).open(encoding="utf-8").readlines()
    #     if "What is the historical importance of Balochistan" in line
    # ])

    calmqa_data = [
        entry
        for dataset_load_path in dataset_load_paths
        for entry in json.load(Path(dataset_load_path).open(encoding="utf-8"))[
            "entries"
        ]
    ]

    _add_original_answers(veriscore_data, calmqa_data)
    _mark_errors(veriscore_data)
    _add_veriscores(veriscore_data)

    for entry in veriscore_data:
        del entry["claim_list"]
        del entry["all_claims"]
        del entry["claim_search_results"]
        del entry["abstained"]

    entries_by_source = defaultdict(list)
    for entry in veriscore_data:
        source = entry["prompt_source"]
        entries_by_source[source].append(entry)

    for source, source_entries in entries_by_source.items():
        file_path = Path(output_path) / f"{source}.json"

        if compress:
            encoded = json.dumps(source_entries, indent=2).encode('utf-8')
            compressed = gzip.compress(encoded)
            file_path.write_bytes(compressed)
        else:
            json.dump(
                source_entries,
                file_path.open("w", encoding="utf-8"),
                ensure_ascii=False,
                indent=2,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "veriscore_data_paths",
        nargs="+",
        help="Path to veriscore data files",
    )
    parser.add_argument(
        "--dataset_load_paths",
        nargs="+",
        dest="dataset_load_paths",
        help="Path of json files containing the dataset",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Path of file to save the results to",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="If set, compress resulting json",
    )

    args = parser.parse_args()

    clean_veriscore(
        veriscore_data_paths=args.veriscore_data_paths,
        dataset_load_paths=args.dataset_load_paths,
        output_path=args.output_path,
        compress=args.compress,
    )


if __name__ == "__main__":
    main()
