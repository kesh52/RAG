import os
import sys
import subprocess
from dotenv import load_dotenv

def start_proxy():
    # Load environment variables from .env
    load_dotenv()
    
    connection_name = os.getenv("INSTANCE_CONNECTION_NAME")
    port = os.getenv("DB_PORT", "5432")
    
    if not connection_name:
        print("Error: INSTANCE_CONNECTION_NAME not found in .env file.", file=sys.stderr)
        print("Please configure it in your .env file like this:", file=sys.stderr)
        print("INSTANCE_CONNECTION_NAME=your-project:your-region:your-instance", file=sys.stderr)
        sys.exit(1)
        
    print(f"Starting Cloud SQL Auth Proxy for: {connection_name} on port: {port}")
    print("Press Ctrl+C to stop the proxy.")
    
    try:
        # Run the proxy as a subprocess
        subprocess.run(["cloud-sql-proxy", connection_name, "--port", port], check=True)
    except FileNotFoundError:
        print("\nError: 'cloud-sql-proxy' command not found.", file=sys.stderr)
        print("Please install the Cloud SQL Auth Proxy binary first.", file=sys.stderr)
        print("On macOS, you can install it using Homebrew:", file=sys.stderr)
        print("  brew install cloud-sql-proxy", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCloud SQL Auth Proxy stopped.")
    except Exception as e:
        print(f"\nError running Cloud SQL Auth Proxy: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    start_proxy()

