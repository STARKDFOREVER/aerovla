(aerovla-server) PS D:\aerovla-server> python aerovla_server.py
[AeroVLA Server] http://0.0.0.0:6006
[AeroVLA Server] API docs: http://localhost:6006/api/docs
[AeroVLA Server] Frontend: http://localhost:6006/
INFO:     Started server process [13820]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:6006 (Press CTRL+C to quit)
[SERVER] Loading VLA engine from: D:\aerovla-server\openvla-7b
[SERVER] LoRA from: D:\aerovla-server\openvla-7b\weight-lora\aerial_vla
[SERVER] 4080 全精度 bf16 — 无需 bitsandbytes
You are offline and the cache for model files in Transformers v4.22.0 has been updated while your local cache seems to be the one of a previous version. It is very likely that all your calls to any `from_pretrained()` method will fail. Remove the offline mode and enable internet connection to have your cache be updated automatically, then you can go back to offline mode.
0it [00:00, ?it/s]
[AeroVLA] Loading base model from: D:\aerovla-server\openvla-7b
[AeroVLA] Loading LoRA adapter from: D:\aerovla-server\openvla-7b\weight-lora\aerial_vla
[AeroVLA] 4-bit: False, 8-bit: False
[SERVER] VLA engine load failed: local variable '_orig_requires_grad_' referenced before assignment
