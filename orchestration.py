"""
COLAB 1: ORCHESTRATOR (FIXED - Proper sampling + chat format)
Fixes the EOS token issue and adds proper chat formatting
"""

# ============= INSTALLATION =============
!pip install -q flask flask-cors pyngrok transformers torch accelerate

# ============= IMPORTS =============
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import requests
import time

# ============= CONFIGURATION =============
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# IMPORTANT: Paste Worker URLs here
WORKER_1_URL = "YOUR_WORKER_1_NGROK_URL_HERE"
WORKER_2_URL = "YOUR_WORKER_2_NGROK_URL_HERE"

# ============= ORCHESTRATOR CLASS =============
class Orchestrator:
    def __init__(self):
        print("="*60)
        print("🎭 ORCHESTRATOR INITIALIZING")
        print("="*60)

        print("📦 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("📦 Loading model components...")
        full_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        self.embed_tokens = full_model.model.embed_tokens
        self.norm = full_model.model.norm
        self.lm_head = full_model.lm_head

        # Store EOS token ID
        self.eos_token_id = self.tokenizer.eos_token_id
        print(f"   EOS token ID: {self.eos_token_id}")

        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ Orchestrator ready!")
        print("="*60)

    def format_chat_prompt(self, user_message):
        """
        Format prompt in TinyLlama chat format
        This is CRITICAL for proper responses!
        """
        # TinyLlama-Chat uses this specific format
        prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{user_message}</s>\n<|assistant|>\n"
        return prompt

    def generate_with_sampling(self, prompt, max_new_tokens=20, temperature=0.7, top_k=50, top_p=0.9):
        """
        Generate with proper sampling (not just argmax)
        Uses temperature, top-k, and top-p sampling
        """
        print(f"\n{'='*60}")
        print(f"💬 User Query: {prompt}")
        print(f"{'='*60}")

        # Format as chat
        formatted_prompt = self.format_chat_prompt(prompt)
        print(f"📝 Formatted prompt:\n{formatted_prompt[:100]}...")

        start_time = time.time()

        if not WORKER_1_URL or not WORKER_2_URL:
            return {
                "error": "Workers not configured!",
                "status": "failed"
            }

        try:
            headers = {
                'ngrok-skip-browser-warning': 'true',
                'Content-Type': 'application/json'
            }

            # Tokenize input
            print(f"\n🔄 Step 1: Tokenizing input...")
            input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
            print(f"   Input tokens: {input_ids.shape[1]}")

            # Generate tokens
            generated_ids = input_ids.clone()
            generated_tokens = []

            print(f"🔄 Step 2: Generating up to {max_new_tokens} tokens...")
            print(f"   Temperature: {temperature}")
            print(f"   Top-k: {top_k}, Top-p: {top_p}")

            for step in range(max_new_tokens):
                # Embed the full sequence
                hidden_states = self.embed_tokens(generated_ids)

                # Send through Worker 1
                response1 = requests.post(
                    f"{WORKER_1_URL}/process",
                    json={"hidden_states": hidden_states.tolist()},
                    headers=headers,
                    timeout=30
                )

                if response1.status_code != 200:
                    raise Exception(f"Worker 1 failed: {response1.text}")

                hidden_states = torch.tensor(response1.json()["hidden_states"])

                # Send through Worker 2
                response2 = requests.post(
                    f"{WORKER_2_URL}/process",
                    json={"hidden_states": hidden_states.tolist()},
                    headers=headers,
                    timeout=30
                )

                if response2.status_code != 200:
                    raise Exception(f"Worker 2 failed: {response2.text}")

                hidden_states = torch.tensor(response2.json()["hidden_states"])

                # Final processing
                hidden_states = self.norm(hidden_states)
                logits = self.lm_head(hidden_states)

                # Get logits for last position
                next_token_logits = logits[0, -1, :]

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

                # Debug: Show top 5 predictions
                if step < 3:
                    top_probs, top_ids = torch.topk(probs, 5)
                    print(f"\n   Step {step+1} - Top 5 predictions:")
                    for prob, token_id in zip(top_probs, top_ids):
                        token_text = self.tokenizer.decode([token_id.item()])
                        print(f"      '{token_text}' ({prob.item()*100:.1f}%)")
                    print(f"   Selected: '{self.tokenizer.decode([next_token_id])}'")

                # Check for EOS
                if next_token_id == self.eos_token_id:
                    print(f"\n   ✓ EOS token reached at step {step+1}")
                    break

                # Append to sequence
                generated_ids = torch.cat([
                    generated_ids,
                    torch.tensor([[next_token_id]])
                ], dim=1)

                generated_tokens.append(next_token_id)

                # Show progress
                if (step + 1) % 5 == 0:
                    partial = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    print(f"   Progress: '{partial}'")

            # Decode generated tokens
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            elapsed = time.time() - start_time
            tokens_per_sec = len(generated_tokens) / elapsed if elapsed > 0 else 0

            print(f"\n✅ Generation complete!")
            print(f"   Generated: '{generated_text}'")
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Speed: {tokens_per_sec:.2f} tokens/sec")
            print(f"   Tokens: {len(generated_tokens)}")
            print(f"{'='*60}\n")

            return {
                "prompt": prompt,
                "response": generated_text,
                "status": "success",
                "time_taken": f"{elapsed:.2f}s",
                "tokens_generated": len(generated_tokens),
                "tokens_per_sec": f"{tokens_per_sec:.2f}"
            }

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                "error": error_msg,
                "status": "failed"
            }

# ============= FLASK API =============
app = Flask(__name__)
CORS(app)

orchestrator = None

@app.route('/')
def home():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Distributed LLM API</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 50px auto;
                padding: 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            .container {{
                background: rgba(255, 255, 255, 0.1);
                padding: 40px;
                border-radius: 20px;
                backdrop-filter: blur(10px);
            }}
            h1 {{ margin: 0 0 10px 0; }}
            .status {{
                display: inline-block;
                background: #4CAF50;
                padding: 8px 20px;
                border-radius: 25px;
                font-weight: bold;
            }}
            .section {{
                background: rgba(0,0,0,0.2);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Distributed LLM API</h1>
            <h2>Status: <span class="status">ONLINE</span></h2>

            <div class="section">
                <h3>✅ Fixed Issues:</h3>
                <p>• Proper chat formatting</p>
                <p>• Temperature-based sampling</p>
                <p>• Top-k and top-p filtering</p>
                <p>• No more immediate EOS!</p>
            </div>

            <div class="section">
                <h3>🔧 Workers:</h3>
                <p>Worker 1: {'✅ ' + WORKER_1_URL if WORKER_1_URL else '❌ Not configured'}</p>
                <p>Worker 2: {'✅ ' + WORKER_2_URL if WORKER_2_URL else '❌ Not configured'}</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "role": "orchestrator",
        "model": MODEL_NAME,
        "version": "sampling_fixed"
    })

@app.route('/workers', methods=['GET'])
def check_workers():
    results = {}
    headers = {'ngrok-skip-browser-warning': 'true'}

    try:
        if WORKER_1_URL:
            resp = requests.get(f"{WORKER_1_URL}/health", headers=headers, timeout=5)
            results["worker_1"] = "online" if resp.status_code == 200 else "error"
        else:
            results["worker_1"] = "not_configured"
    except:
        results["worker_1"] = "offline"

    try:
        if WORKER_2_URL:
            resp = requests.get(f"{WORKER_2_URL}/health", headers=headers, timeout=5)
            results["worker_2"] = "online" if resp.status_code == 200 else "error"
        else:
            results["worker_2"] = "not_configured"
    except:
        results["worker_2"] = "offline"

    return jsonify(results)

@app.route('/generate', methods=['POST'])
def generate():
    global orchestrator

    if orchestrator is None:
        return jsonify({"error": "Orchestrator not initialized"}), 500

    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 20)
    temperature = data.get('temperature', 0.7)

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Limit tokens
    max_tokens = min(max_tokens, 30)

    result = orchestrator.generate_with_sampling(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=50,
        top_p=0.9
    )
    return jsonify(result)

# ============= STARTUP =============
def start_server():
    global orchestrator

    print("\n🚀 STARTING ORCHESTRATOR (SAMPLING FIXED)\n")
    orchestrator = Orchestrator()

    ngrok.set_auth_token("YOUR-NGROK-TOKEN")
    public_url = ngrok.connect(5000)

    print(f"\n{'='*60}")
    print(f"✅ PUBLIC API URL: {public_url}")
    print(f"{'='*60}\n")

    app.run(host='0.0.0.0', port=5000, debug=False)

# ============= RUN =============
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║         ORCHESTRATOR - EOS ISSUE FIXED!                 ║
╚══════════════════════════════════════════════════════════╝

✅ Changes:
   • Added proper TinyLlama chat formatting
   • Temperature-based sampling (not greedy!)
   • Top-k and top-p filtering
   • Shows top predictions for debugging

This should fix the immediate EOS problem!
    """)

    time.sleep(2)
    start_server()