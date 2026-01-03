"""
TEST CLIENT - With longer timeout for distributed system
"""

import requests
import json

class DistributedLLMClient:
    def __init__(self, api_url):
        self.api_url = api_url.rstrip('/')
        self.headers = {
            'ngrok-skip-browser-warning': 'true',
            'User-Agent': 'DistributedLLMClient/1.0'
        }
        print(f"🔗 Connected to: {self.api_url}")
        self.check_health()

    def check_health(self):
        try:
            response = requests.get(
                f"{self.api_url}/health",
                headers=self.headers,
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API Status: {data.get('status', 'unknown')}")
                print(f"   Role: {data.get('role', 'unknown')}")
                return True
            return False
        except Exception as e:
            print(f"❌ Cannot connect: {str(e)}")
            return False

    def check_workers(self):
        try:
            response = requests.get(
                f"{self.api_url}/workers",
                headers=self.headers,
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                print("\n🔧 Worker Status:")
                for worker, status in data.items():
                    emoji = "✅" if status == "online" else "❌"
                    print(f"   {emoji} {worker}: {status}")
                return data
            return None
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None

    def generate(self, prompt, max_tokens=20):
        """
        Generate text - optimized for 20 tokens (faster!)
        """
        print(f"\n{'='*60}")
        print(f"💬 Prompt: {prompt}")
        print(f"{'='*60}")
        print("⏳ Generating... (100-125 seconds expected)")

        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens
                },
                headers=self.headers,
                timeout=200  # Increased from 60 to 90 seconds
            )

            if response.status_code == 200:
                data = response.json()

                if "error" in data:
                    print(f"\n❌ Error: {data['error']}")
                    return None

                print(f"\n🤖 Response: {data.get('response', 'No response')}")

                # Show performance stats if available
                if 'time_taken' in data:
                    print(f"\n📊 Performance:")
                    print(f"   Time: {data.get('time_taken', 'N/A')}")
                    print(f"   Tokens: {data.get('tokens_generated', 'N/A')}")
                    print(f"   Speed: {data.get('tokens_per_sec', 'N/A')} tok/s")

                print(f"{'='*60}\n")
                return data
            else:
                print(f"\n❌ HTTP Error {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            print(f"\n❌ Request timed out after 90 seconds!")
            print(f"💡 The distributed system is still processing...")
            print(f"💡 Check the Colab 1 logs to see progress")
            return None
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            return None

    def interactive_chat(self):
        print("\n" + "="*60)
        print("💬 INTERACTIVE CHAT MODE")
        print("="*60)
        print("Commands:")
        print("  - Type your question and press Enter")
        print("  - 'quit' / 'exit' / 'q' - Exit")
        print("  - 'workers' - Check worker status")
        print("  - 'health' - Check API health")
        print("\n⚡ Tip: Shorter prompts = faster responses!")
        print("="*60 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == 'workers':
                    self.check_workers()
                    continue

                if user_input.lower() == 'health':
                    self.check_health()
                    continue

                # Generate response
                # Use fewer tokens for faster responses
                self.generate(user_input, max_tokens=15)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║     Distributed LLM Client (Optimized for Speed)        ║
╚══════════════════════════════════════════════════════════╝

⚡ Optimizations:
   • 90 second timeout (instead of 60)
   • Default 15-20 tokens (faster than 50)
   • Progress indicators
    """)

    api_url = input("Enter the API URL from Colab 1: ").strip()

    if not api_url:
        print("❌ No URL provided!")
        return

    client = DistributedLLMClient(api_url)
    client.check_workers()

    print("\n" + "="*60)
    print("What would you like to do?")
    print("1. Interactive chat (recommended)")
    print("2. Single query")
    print("3. Quick test")
    print("="*60)

    choice = input("\nChoice (1/2/3): ").strip()

    if choice == "1":
        client.interactive_chat()
    elif choice == "2":
        query = input("\nYour question: ").strip()
        if query:
            client.generate(query, max_tokens=20)
    elif choice == "3":
        print("\n🧪 Quick test with short prompt...\n")
        client.generate("Hello", max_tokens=15)
    else:
        client.interactive_chat()


if __name__ == "__main__":
    main()