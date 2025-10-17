import os
import json

def combine_jsons(base_dirs, output_file):
    combined = {
        "question": [],
        "ground_truth": [],
        "context": [],
        "topic_id": [],
        "num_hops": [],
        "question_metadata": [],
        "internal_context": [],
        "verification_scores": [],
    }

    for base_dir in base_dirs:
        dataset_name = os.path.basename(base_dir)  # e.g., "mh_neuclir"
        if "neuclir" in dataset_name:
            prefix = "neuclir"
        elif "trecrag" in dataset_name:
            prefix = "trecrag2025"
        else:
            prefix = dataset_name

        # iterate over numeric subfolders
        for subfolder in sorted(
            [d for d in os.listdir(base_dir) if d.isdigit()],
            key=lambda x: int(x)
        ):
            subfolder_path = os.path.join(base_dir, subfolder)

            # paths
            unanswerable_path = os.path.join(subfolder_path, "unanswerable.json")
            nuggetbank_path = os.path.join(
                subfolder_path,
                "_data_multihop",
                f"{subfolder}__multihop_nugget_bank__mode_documents_unanswerable.json"
            )

            if not os.path.isfile(nuggetbank_path):
                continue  # skip if nuggetbank missing

            with open(nuggetbank_path, "r") as f:
                nuggetbank = json.load(f)

            # if unanswerable.json exists, load from there
            # if os.path.isfile(unanswerable_path):
            #     with open(unanswerable_path, "r") as f:
            #         data = json.load(f)
            #     questions = data["question"]
            #     ground_truths = data["ground_truth"]
            #     contexts = data["contexts"]
            # else:
            # fallback: reconstruct from nuggetbank
            questions = list(nuggetbank["nugget_bank"].keys())
            ground_truths = [
                nuggetbank["nugget_bank"][q]["answers"]["answer"]["answer"]
                for q in questions
            ]
            contexts = [
                nuggetbank["nugget_bank"][q]["metadata"]["full_context"]
                for q in questions
            ]

            # for each question, extract additional metadata
            for q, g, c in zip(questions, ground_truths, contexts):
                try:
                    q_meta = nuggetbank["nugget_bank"][q]["answers"]["answer"]["metadata"]
                except: 
                    print(q)
                num_hops = q_meta["cot_annotation"]["num_hops"]
                if num_hops=="":
                    num_hops=-1
                internal_context = nuggetbank["nugget_bank"][q]["metadata"]["full_internal_context"]
                verification_scores = {key:q_meta["scores"][key][-1] for key in q_meta["scores"]}

                scores_list = [q_meta["scores"][key][-1] for key in q_meta["scores"]]

                if scores_list[0]>=1 and scores_list[1]>=1 and scores_list[2]>=1 and scores_list[3]>=1 and scores_list[4]<=1 and scores_list[5]<=1:
                    combined["question"].append(q)
                    combined["ground_truth"].append(g)
                    combined["context"].append(c)
                    combined["topic_id"].append(f"{prefix}_{subfolder}")
                    combined["num_hops"].append(num_hops)
                    combined["question_metadata"].append(q_meta)
                    combined["internal_context"].append(internal_context)
                    combined["verification_scores"].append(verification_scores)

    # save combined JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    num_qs = len(combined["question"])
    print(f"Saved {num_qs} questions!!!")


if __name__ == "__main__":
    base_dirs = [
        "./generated_data/mh_neuclir",
        "./generated_data/mh_trecrag"
    ]
    output_file = "./generated_data/filtered_crumqs.json"
    combine_jsons(base_dirs, output_file)

