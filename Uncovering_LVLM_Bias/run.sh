


# Physical Gender 

python test_vllm.py --model_name deepseek-ai/deepseek-vl2-tiny --out_dir outputs_vllm/physical_gender --im_path SocialCounterfactuals/images/ --metadata_path metadata/metadata_physical_gender.csv --batch_size 10 --n_images 10 --prompts keywords_characteristics,keywords_personality
python test_vllm.py --model_name Qwen/Qwen2.5-VL-3B-Instruct --out_dir outputs_vllm/physical_gender --im_path SocialCounterfactuals/images/ --metadata_path metadata/metadata_physical_gender.csv --batch_size 10 --n_images 10 --prompts keywords_characteristics,keywords_personality
python test_vllm.py --model_name OpenGVLab/InternVL3-2B --out_dir outputs_vllm/physical_gender --im_path SocialCounterfactuals/images/ --metadata_path metadata/metadata_physical_gender.csv --batch_size 10 --n_images 10 --prompts keywords_characteristics,keywords_personality
python test_vllm.py --model_name HuggingFaceTB/SmolVLM2-2.2B-Instruct --out_dir outputs_vllm/physical_gender --im_path SocialCounterfactuals/images/ --metadata_path metadata/metadata_physical_gender.csv --batch_size 10 --n_images 10 --prompts keywords_characteristics,keywords_personality


# Race Gender 
python test_vllm.py --model_name deepseek-ai/deepseek-vl2-tiny --out_dir outputs_vllm/race_gender/ --im_path SocialCounterfactuals/images/ --metadata_path metadata/metadata_race_gender.csv --batch_size 12 --n_images 12 --prompts keywords_characteristics,keywords_personality
python test_vllm.py --model_name Qwen/Qwen2.5-VL-3B-Instruct --out_dir outputs_vllm/race_gender/ --im_path SocialCounterfactuals/images/ --metadata_path metadata/metadata_race_gender.csv --batch_size 12 --n_images 12 --prompts keywords_characteristics,keywords_personality
python test_vllm.py --model_name OpenGVLab/InternVL3-2B --out_dir outputs_vllm/race_gender/ --im_path SocialCounterfactuals/images/ --metadata_path metadata/metadata_race_gender.csv --batch_size 12 --n_images 12 --prompts keywords_characteristics,keywords_personality
python test_vllm.py --model_name HuggingFaceTB/SmolVLM2-2.2B-Instruct --out_dir outputs_vllm/race_gender/ --im_path SocialCounterfactuals/images/ --metadata_path metadata/metadata_race_gender.csv --batch_size 12 --n_images 12 --prompts keywords_characteristics,keywords_personality


# Physical Race
python test_vllm.py --model_name deepseek-ai/deepseek-vl2-tiny --out_dir outputs_vllm/physical_race/ --im_path SocialCounterfactuals/images/ --metadata_path metadata/metadata_physical_race.csv --batch_size 12 --n_images 54 --prompts keywords_characteristics,keywords_personality
python test_vllm.py --model_name Qwen/Qwen2.5-VL-3B-Instruct --out_dir outputs_vllm/physical_race/ --im_path SocialCounterfactuals/images/ --metadata_path metadata/metadata_physical_race.csv --batch_size 12 --n_images 54 --prompts keywords_characteristics,keywords_personality
python test_vllm.py --model_name OpenGVLab/InternVL3-2B --out_dir outputs_vllm/physical_race/ --im_path SocialCounterfactuals/images/ --metadata_path metadata/metadata_physical_race.csv --batch_size 12 --n_images 54 --prompts keywords_characteristics,keywords_personality
python test_vllm.py --model_name HuggingFaceTB/SmolVLM2-2.2B-Instruct --out_dir outputs_vllm/physical_race/ --im_path SocialCounterfactuals/images/ --metadata_path metadata/metadata_physical_race.csv --batch_size 12 --n_images 54 --prompts keywords_characteristics,keywords_personality
