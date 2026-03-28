torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") \
    univideo_inference_mgpu.py \
    --demo_task in_context_video_gen \
    --config configs/univideo_qwen2p5vl7b_hidden_hunyuanvideo.yaml
