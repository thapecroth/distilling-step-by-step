CUDA_VISIBLE_DEVICES=1,2,3 python run.py --from_pretrained google/t5-v1_1-base --dataset cqa --model_type standard --label_type gt --batch_size 16 --bf16
CUDA_VISIBLE_DEVICES=1,2,3 python run.py --from_pretrained google/t5-v1_1-base --dataset cqa --model_type task_prefix --label_type gt --llm palm --alpha 0.5 --batch_size 16 --bf16

CUDA_VISIBLE_DEVICES=1,2,3 python run.py --from_pretrained google/t5-v1_1-base --dataset esnli --model_type standard --label_type gt --batch_size 16 --bf16
CUDA_VISIBLE_DEVICES=1,2,3 python run.py --from_pretrained google/t5-v1_1-base --dataset esnli --model_type task_prefix --label_type gt --llm palm --alpha 0.5 --batch_size 16 --bf16

CUDA_VISIBLE_DEVICES=1,2,3 python run.py --from_pretrained google/t5-v1_1-base --dataset anli1 --model_type standard --label_type gt --batch_size 16 --bf16
CUDA_VISIBLE_DEVICES=1,2,3 python run.py --from_pretrained google/t5-v1_1-base --dataset anli1 --model_type task_prefix --label_type gt --llm palm --alpha 0.5 --batch_size 16 --bf16

CUDA_VISIBLE_DEVICES=1,2,3 python run.py --from_pretrained google/t5-v1_1-base --dataset svamp --model_type standard --label_type gt --batch_size 16 --bf16
CUDA_VISIBLE_DEVICES=1,2,3 python run.py --from_pretrained google/t5-v1_1-base --dataset svamp --model_type task_prefix --label_type gt --llm palm --alpha 0.5 --batch_size 16 --bf16