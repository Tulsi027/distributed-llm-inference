"""
=====================================================================
WORKER 2 - COMPLETE CODE WITH NGROK FIX
=====================================================================
Replace ALL code in Colab 3 with this
=====================================================================
"""

# ============= STEP 1: INSTALL PACKAGES =============
print("📦 Installing packages...")
!pip install -q flask flask-cors pyngrok transformers torch accelerate
print("✅ Packages installed!\n")

# ============= STEP 2: IMPORTS =============
import torch
from transformers import AutoModelForCausalLM
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import json
import traceback
import time

# ============= STEP 3: CONFIG =============
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LAYER_START = 11
LAYER_END = 22
WORKER_NAME = "Worker 2"
WORKER_PORT = 5002
NGROK_TOKEN = "YOUR_WORKER_2_NGROK_TOKEN"

print("="*60)
print(f"⚙️  {WORKER_NAME} - CONFIGURATION")
print("="*60)
print(f"Model: {MODEL_NAME}")
print(f"Layers: {LAYER_START} to {LAYER_END}")
print(f"Port: {WORKER_PORT}")
print("="*60 + "\n")

# ============= STEP 4: WORKER CLASS =============
class Worker:
    def __init__(self, model_name, layer_start, layer_end, worker_name):
        print("="*60)
        print(f"⚙️ {worker_name} INITIALIZING")
        print(f"Loading layers {layer_start}-{layer_end}...")
        print("="*60)

        self.layer_start = layer_start
        self.layer_end = layer_end
        self.worker_name = worker_name

        print("\n📦 Downloading model...")
        full_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )

        print(f"\n✂️ Extracting layers {layer_start} to {layer_end}...")
        self.layers = torch.nn.ModuleList([
            full_model.model.layers[i] for i in range(layer_start, layer_end)
        ])

        # CRITICAL: Keep a reference to the full model config
        print("🔧 Storing model configuration...")
        self.config = full_model.config
        self.model = full_model.model  # Keep the whole model for rotary embeddings

        print(f"   Extracted {len(self.layers)} layers")

        print(f"\n✅ {worker_name} ready with {len(self.layers)} layers!")
        print("="*60 + "\n")

    def process(self, hidden_states):
        """Process hidden states through transformer layers"""
        print(f"\n{'='*60}")
        print(f"🔄 PROCESSING THROUGH {len(self.layers)} LAYERS")
        print(f"{'='*60}")
        print(f"Input shape: {hidden_states.shape}")

        batch_size, seq_length, _ = hidden_states.shape

        with torch.no_grad():
            # Create position IDs
            position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)

            # Get rotary embeddings using the model's method
            # This works universally across transformers versions
            try:
                # Method 1: Try using model's rotary_emb
                if hasattr(self.model, 'rotary_emb'):
                    cos, sin = self.model.rotary_emb(hidden_states, position_ids)
                    position_embeddings = (cos, sin)
                    print(f"✓ Position embeddings from model.rotary_emb")
                else:
                    # Method 2: Use the first layer's self_attn rotary_emb
                    first_layer = self.model.layers[self.layer_start]
                    if hasattr(first_layer.self_attn, 'rotary_emb'):
                        cos, sin = first_layer.self_attn.rotary_emb(hidden_states, position_ids)
                        position_embeddings = (cos, sin)
                        print(f"✓ Position embeddings from layer.self_attn.rotary_emb")
                    else:
                        # Method 3: Compute manually using config
                        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
                        rotary_emb = LlamaRotaryEmbedding(config=self.config)
                        cos, sin = rotary_emb(hidden_states, position_ids)
                        position_embeddings = (cos, sin)
                        print(f"✓ Position embeddings computed manually")
            except Exception as e:
                print(f"⚠️ Warning: Could not compute position embeddings: {e}")
                position_embeddings = None

            # Process through layers
            for i, layer in enumerate(self.layers):
                try:
                    if position_embeddings is not None:
                        # Try with position_embeddings parameter
                        try:
                            layer_output = layer(
                                hidden_states,
                                attention_mask=None,
                                position_ids=position_ids,
                                past_key_value=None,
                                output_attentions=False,
                                use_cache=False,
                                cache_position=None,
                                position_embeddings=position_embeddings,
                            )
                        except TypeError:
                            # If position_embeddings parameter doesn't exist, try without it
                            layer_output = layer(
                                hidden_states,
                                attention_mask=None,
                                position_ids=position_ids,
                                past_key_value=None,
                                output_attentions=False,
                                use_cache=False,
                            )
                    else:
                        # Fallback: just position_ids
                        layer_output = layer(
                            hidden_states,
                            attention_mask=None,
                            position_ids=position_ids,
                        )

                    # Extract hidden states
                    if isinstance(layer_output, tuple):
                        hidden_states = layer_output[0]
                    else:
                        hidden_states = layer_output

                    if hidden_states is None:
                        raise ValueError(f"Layer {i} produced None output!")

                    if i % 3 == 0 or i == 0:
                        print(f"   ✓ Layer {self.layer_start + i} complete - shape: {hidden_states.shape}")

                except Exception as e:
                    print(f"   ❌ Error in layer {i}: {str(e)}")
                    traceback.print_exc()
                    raise

        print(f"✅ All {len(self.layers)} layers processed!")
        print(f"   Final output shape: {hidden_states.shape}")
        print(f"{'='*60}\n")

        return hidden_states

# ============= STEP 5: FLASK API =============
app = Flask(__name__)
CORS(app)

worker = None

@app.route('/')
def home():
    return f"""
    <html>
    <head><title>{WORKER_NAME}</title></head>
    <body style="font-family:Arial;padding:40px;background:#fff3e0;">
        <h1>⚙️ {WORKER_NAME} (Layers {LAYER_START}-{LAYER_END})</h1>
        <h2>Status: <span style="color:green;">ONLINE</span></h2>
        <p>Model: {MODEL_NAME}</p>
        <p>Loaded {LAYER_END - LAYER_START} layers</p>
    </body>
    </html>
    """

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "role": "worker_2",
        "layers": f"{LAYER_START}-{LAYER_END}",
        "model": MODEL_NAME
    })

@app.route('/process', methods=['POST'])
def process():
    global worker

    if worker is None:
        return jsonify({"error": "Worker not initialized"}), 500

    try:
        data = request.json
        hidden_states_list = data.get("hidden_states")

        if not hidden_states_list:
            return jsonify({"error": "No hidden states provided"}), 400

        # Convert to tensor
        hidden_states = torch.tensor(hidden_states_list, dtype=torch.float32)

        print(f"📥 Received hidden states: {hidden_states.shape}")

        # Process through layers
        output = worker.process(hidden_states)

        # Convert back to list
        output_list = output.tolist()

        print(f"📤 Sending output: shape {output.shape}\n")

        return jsonify({
            "hidden_states": output_list,
            "status": "success",
            "worker": "worker_2"
        })

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500

# ============= STEP 6: START WORKER =============
def start_worker():
    global worker

    print("\n" + "="*60)
    print(f"🚀 STARTING {WORKER_NAME}")
    print("="*60 + "\n")

    worker = Worker(MODEL_NAME, LAYER_START, LAYER_END, WORKER_NAME)

    print("\n🌐 Setting up ngrok...")

    # CRITICAL FIX: Kill any existing ngrok tunnels first
    try:
        print("🔧 Cleaning up old ngrok tunnels...")
        ngrok.kill()
        time.sleep(2)  # Wait for cleanup
    except Exception as e:
        print(f"   (No existing tunnels to clean: {e})")

    # Set auth token
    print("🔑 Setting ngrok auth token...")
    ngrok.set_auth_token(NGROK_TOKEN)

    # Create new tunnel with unique port
    print(f"🌐 Creating public URL on port {WORKER_PORT}...")
    public_url = ngrok.connect(WORKER_PORT)

    print(f"\n{'='*60}")
    print(f"✅ {WORKER_NAME} URL: {public_url}")
    print(f"{'='*60}")
    print(f"\n📋 COPY THIS URL TO ORCHESTRATOR (WORKER_2_URL)\n")

    app.run(port=WORKER_PORT, host='0.0.0.0', debug=False)

# ============= STEP 7: RUN =============
if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   WORKER 2 STARTUP                           ║
╚══════════════════════════════════════════════════════════════╝

Starting Worker 2 (Layers 11-22)...
    """)

    start_worker()