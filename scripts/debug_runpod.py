import os
import sys
import runpod
from dotenv import load_dotenv

def test_connection():
    # .envを読み込み
    load_dotenv()
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    
    if not api_key or api_key == "your_api_key_here":
        print("Error: RUNPOD_API_KEY is not set correctly in .env file.")
        print("Please edit the .env file and replace 'your_api_key_here' with your actual RunPod API key.")
        return

    print(f"API Key found: {api_key[:4]}...{api_key[-4:] if len(api_key) > 4 else ''}")
    
    runpod.api_key = api_key
    
    try:
        print("Checking credentials...")
        # APIキーの有効性を確認
        # 注意: check_credentials() が例外を投げない場合もあるため、返り値も確認
        try:
            runpod.check_credentials()
            print("Credentials Check: OK")
        except Exception as ce:
            print(f"Credentials Check Failed: {ce}")
            return

        print("\nFetching GPU types...")
        # get_gpus() を使用
        gpus = runpod.get_gpus()
        print(f"Connection Successful! Found {len(gpus)} GPU types.")
        
        print("\nAvailable GPUs (Top 5):")
        for i, gpu in enumerate(gpus[:5]):
            # 構造に合わせて表示（id と displayName が一般的）
            gpu_id = gpu.get('id', 'N/A')
            gpu_name = gpu.get('displayName', 'N/A')
            print(f" - {gpu_id}: {gpu_name}")
            
    except Exception as e:
        print(f"Error during connection test: {e}")

if __name__ == "__main__":
    test_connection()
