# LING 380 - Final Project

This repo contains the report, presentation, code, dataset, and results of the **LING 380** final project of Sivan Almogy, Avi Kabra, and William Palmer. This project studies whether GPT-2 and Llama have the prerequisite knowledge to understand figurative language. To run files please remove directory structure so that all files are in the same directory or update paths accordingly.

---

## File Descriptions

### `report.pdf`
- The final report for this project.

### `presentation.pdf`
- The dataset used for the experiment in the paper. The output of `data_gen.ipynb`.

### `data/`

- `dev.csv`: the original dev set provided by Liu et al. (2022) for testing purposes. We adapt this dataset using `filtered_parse.py` to get `rearranged_dev_filtered.csv` and `subjectID_dev.csv` used in our experiments. 
- `dev-categories.csv` the original dev set provided by Liu et al. (2022) with metaphor category information, but without labels. To be merged with dev.csv.
- `rearranged_dev_filtered.csv`: our adapted and filtered data set for the object identificaion task.
- `subjectID_dev.csv`: our adapted and filtered data set for the subject identificaion task.

### `code/`

- `parse.py`: python script to adapt the dataset. Superceded by `filtered_parse.py`.
- `filtered_parse.py`: same as parse.py, but also filters out noun-verb disagreements .
- `fig_gpt_test.py`: given the adapted test datasets, collects results for GPT-2.
- `fig_llama_test.py`: given the adapted test datasets, collects results for llama. Requires HuggingFace access tokens and llama permissions.
- `fig_analysis-script.R`: performs analysis based on the result files.

### `results/`

- `fig-gpt-comb.csv`: final results for GPT-2 on the object identification task.
- `fig-llama-comb.csv`: final results for Llama on the object identification task.
- `fig-gpt-subjectid-comb.csv`: final results for GPT-2 on the subject identification task.
- `fig-llama-subjectid-comb.csv`: final results for Llama on the subject identification task.

---
