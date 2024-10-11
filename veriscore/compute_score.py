import json
import argparse

from collections import defaultdict

from veriscore import utils


def main(input_paths: list[str], *, output_path: str | None = None):
    input_data = []
    for input_path in input_paths:
        with open(input_path, "r") as f:
            input_data += [json.loads(line) for line in f if line.strip()]

    model_domain_triplet_dict = defaultdict(
        lambda: defaultdict(list)
    )  # triplet = [[supported, total, # of sentences], ...]

    for dict_item in input_data:
        triplet = [0, 0, 0]
        triplet[1] = len(dict_item["all_claims"])
        triplet[2] = len(dict_item["claim_list"])
        if not dict_item["claim_search_results"]:
            triplet[0] = 0
        else:
            for claim_veri_res in dict_item["claim_verification_result"]:
                if claim_veri_res["verification_result"] == "supported":
                    triplet[0] += 1

        model_name = dict_item["model"]
        domain = dict_item["prompt_source"]
        model_domain_triplet_dict["Overall"][model_name].append(triplet)
        model_domain_triplet_dict[domain][model_name].append(triplet)

    scores_df, _ = utils.get_veriscore(model_domain_triplet_dict)
    scores_df = scores_df.sort_index()
    print(scores_df)
    if output_path is not None:
        scores_df.to_csv(output_path)

        # with Path(output_path).open("w") as f:
        #     for domain in model_domain_triplet_dict:
        #         for model_name in model_domain_triplet_dict[domain]:
        #             f.write(
        #                 f"{domain.split('_')[0]},{model_name},{scores_df[domain][model_name]}\n"
        #             )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_paths", type=str, nargs="+")
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    main(args.input_paths, output_path=args.output_path)
